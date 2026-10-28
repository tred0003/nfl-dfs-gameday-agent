"""Audio -> transcript, five minutes at a time, by Gemini with the roster in the prompt.

    python podcast_rag/transcribe.py ringer-fantasy 1      # 2nd-newest episode
    python podcast_rag/transcribe.py --backfill            # every show's last N
    python podcast_rag/transcribe.py --backfill simmons

Why Gemini and not Speech-to-Text (tested 2026-10-28 on the same 55 s of the
Ringer fantasy show):

  Chirp 3              "Kellyn Black ... Genty's so banged up ... Seth McGallon"
  Chirp 3 + 788-name   worse: "Kylin moeglicherweise to be first" -- a big
    phrase set           inline phrase set confuses it instead of helping
  Gemini 2.5 Flash     "Kaelon Black ... Ashton Jeanty ... Seth McGowan"
    + roster in prompt

The names are the point of the archive -- player_keys, the player filter,
the citations all hang on them -- and Gemini reads the roster as context.
It is also cheaper: ~$0.15 per audio hour against ~$1 for Chirp 3 standard.

What an LLM transcriber can do that Chirp cannot is skip or invent text, so
every clip is checked before it is kept:
  words per minute   speech is ~130-200; under MIN_WPM means a dropped
                     stretch, over MAX_WPM means made-up text
  repetition         the same 8 words more than LOOP_REPEATS times is a
                     decoding loop
A clip that fails is retried once on the same model, then kept but marked
"suspect", so the pilot report counts it instead of hiding it.

Clips are CLIP_SECONDS long so a citation can point at a five-minute window,
and so a bad clip costs five minutes, not the episode. Writes
data/podcasts/transcripts/<episode>.jsonl, one row per clip.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from podcast_rag.feed import SHOWS, Episode, episodes  # noqa: E402
from podcast_rag import segments  # noqa: E402

DATA = ROOT / "data" / "podcasts"
AUDIO, CLIPS, OUT = DATA / "audio", DATA / "clips", DATA / "transcripts"
ROSTER = DATA / "roster.json"

MODEL = "gemini-2.5-flash"
CLIP_SECONDS = 300
MIN_WPM, MAX_WPM = 90, 260
LOOP_REPEATS = 3
WORKERS = 4

PROMPT = """Transcribe this NFL podcast audio verbatim, as plain text with no speaker
labels or timestamps. Spell a player's name exactly as it appears in the roster
below when the speaker means that player. Do not summarize, skip, or add
anything that was not said. Leave out ad reads.

ROSTER:
{roster}"""


def roster_text() -> str:
    if not ROSTER.exists():
        from news_rag.collect import skill_players
        ROSTER.parent.mkdir(parents=True, exist_ok=True)
        ROSTER.write_text(json.dumps(skill_players()), encoding="utf-8")
    ps = json.loads(ROSTER.read_text(encoding="utf-8"))
    return "\n".join(sorted({f"{p['name']} ({p['position']}, {p['team']})" for p in ps}))


def download(ep: Episode) -> Path:
    path = AUDIO / f"{ep.episode_id.replace(':', '_')}.mp3"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Streamed to disk: an episode is ~300 MB, too much to hold in memory.
        # Written to .part and renamed, so a killed run never leaves a
        # truncated file that the next run would take as complete.
        part = path.with_suffix(".part")
        with requests.get(ep.audio_url, timeout=600, stream=True) as r:
            r.raise_for_status()
            with part.open("wb") as f:
                for block in r.iter_content(chunk_size=1 << 20):
                    f.write(block)
        part.replace(path)
    return path


def ranges(ep: Episode, client) -> list[tuple[int, int, str]]:
    """What to transcribe: the whole episode, or a mixed show's NFL chapters."""
    if SHOWS[ep.show]["filter"] == "all" or not ep.chapters:
        return [(0, ep.duration, "nfl")]
    return segments.keep_ranges(ep, segments.label(ep, client))


def clip(src: Path, start: int, end: int) -> bytes:
    """16 kHz mono FLAC of [start, end): speech needs no more, and it keeps
    a five-minute request around 5 MB, well under the 20 MB inline limit."""
    out = CLIPS / f"{src.stem}_{start}.flac"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(end - start),
                        "-i", str(src), "-ac", "1", "-ar", "16000", str(out)], check=True)
    return out.read_bytes()


def problems(text: str, seconds: int) -> list[str]:
    words = text.split()
    wpm = len(words) / (seconds / 60)
    out = []
    if wpm < MIN_WPM:
        out.append(f"low wpm {wpm:.0f}")
    if wpm > MAX_WPM:
        out.append(f"high wpm {wpm:.0f}")
    grams = Counter(" ".join(words[i:i + 8]).lower() for i in range(len(words) - 7))
    if grams and grams.most_common(1)[0][1] > LOOP_REPEATS:
        out.append("repetition loop")
    return out


def transcribe_clip(client, audio: bytes, seconds: int, roster: str) -> dict:
    from google.genai import types
    best = None
    for attempt in (1, 2):
        resp = client.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=audio, mime_type="audio/flac"),
                      PROMPT.format(roster=roster)],
            config=types.GenerateContentConfig(
                temperature=0, thinking_config=types.ThinkingConfig(thinking_budget=0)))
        text = re.sub(r"\s+", " ", resp.text or "").strip()
        u = resp.usage_metadata
        got = {"text": text, "problems": problems(text, seconds), "attempts": attempt,
               "tokens_in": u.prompt_token_count, "tokens_out": u.candidates_token_count}
        if not got["problems"]:
            return got
        best = best or got
    return best | {"attempts": 2}


def transcribe(ep: Episode, client=None) -> Path:
    from news_rag.index import _client
    client = client or _client()
    src, roster = download(ep), roster_text()
    # The feed's <itunes:duration> can run past the file (8,129 s listed,
    # 8,067 s real): a clip that starts after the audio ends is an empty FLAC,
    # which Gemini rejects as malformed. The file's own length is the limit.
    real = int(float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True, check=True).stdout))
    jobs = []
    for start, end, lab in ranges(ep, client):
        end = min(end, real)
        for s in range(start, end, CLIP_SECONDS):
            e = min(s + CLIP_SECONDS, end)
            if e - s >= 20:                     # a sliver at the end of a range
                jobs.append((s, e, lab))

    def run(job):
        s, e, lab = job
        try:
            r = transcribe_clip(client, clip(src, s, e), e - s, roster)
        except Exception as ex:  # noqa: BLE001 -- one bad clip should not lose the episode
            r = {"text": "", "problems": [f"failed: {type(ex).__name__}: {str(ex)[:120]}"],
                 "attempts": 1, "tokens_in": 0, "tokens_out": 0}
        return {"episode_id": ep.episode_id, "show": ep.show, "title": ep.title,
                "published": ep.published.isoformat(), "start": s, "end": e,
                "label": lab, **r}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(run, jobs))
    path = out_path(ep)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    # The feeds serve 320 kbps MP3s, ~290 MB for a two-hour episode; the
    # transcript is what is kept. The audio can be fetched again from the feed.
    for f in [src, *CLIPS.glob(f"{src.stem}_*.flac")]:
        f.unlink(missing_ok=True)
    return path


def out_path(ep: Episode) -> Path:
    return OUT / f"{ep.episode_id.replace(':', '_')}.jsonl"


def backfill(show: str, client=None) -> None:
    """The show's last `last` episodes with NFL content, skipping any already
    transcribed. For a mixed show that walks past the NBA-only episodes, so
    it reads up to four times as many feed items as it keeps."""
    from news_rag.index import _client
    client = client or _client()
    want, kept = SHOWS[show]["last"], 0
    for ep in episodes(show, limit=SHOWS[show]["last"] * 4):
        if kept == want:
            break
        if out_path(ep).exists():
            kept += 1
            continue
        if not ranges(ep, client):
            print(f"  skip (no NFL): {ep.title[:70]}", flush=True)
            continue
        report(ep, transcribe(ep, client))
        kept += 1


def report(ep: Episode, path: Path) -> None:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    secs = sum(r["end"] - r["start"] for r in rows)
    tin, tout = sum(r["tokens_in"] for r in rows), sum(r["tokens_out"] for r in rows)
    print(f"{ep.show} {ep.published:%m-%d} {ep.title[:60]}\n  {len(rows)} clips, "
          f"{secs / 60:.0f} of {ep.duration / 60:.0f} min, "
          f"{sum(len(r['text'].split()) for r in rows):,} words, "
          f"{sum(bool(r['problems']) for r in rows)} suspect, "
          f"{sum(r['attempts'] > 1 for r in rows)} retried, "
          f"{tin:,} tokens in / {tout:,} out", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--backfill":
        for show in sys.argv[2:] or SHOWS:
            backfill(show)
        sys.exit()
    show, idx = sys.argv[1], int(sys.argv[2])
    ep = episodes(show, limit=idx + 1)[idx]
    report(ep, transcribe(ep))
