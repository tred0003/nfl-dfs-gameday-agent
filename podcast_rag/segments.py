"""Label each chapter of a mixed show NFL / mixed / other, before any audio is paid for.

A chapter title alone is often not enough -- "Ringer 107 picks, Week 5" is the
NFL picks segment, "Mailbag questions" could be anything -- so Gemini reads
the episode's title and summary with the list. One call per episode, a few
hundred tokens; the whole backfill is cents. Transcription is the expensive
step, so this is the same pattern as the injection screen: filter before it.

  nfl     the NFL pro game, including fantasy and betting on it
  mixed   NFL plus other things (a mailbag): transcribed, and the chunks that
          name no NFL player or team are dropped after
  other   NBA, MLB, college football, movies, life advice, ads, the intro.
          The NFL draft and college prospects count as other: no help for
          this week's lineups, and the names match no NFL roster.
"""

from __future__ import annotations

import json

from podcast_rag.feed import Episode

MODEL = "gemini-2.5-flash"
KEEP = {"nfl", "mixed"}
MIN_SECONDS = 90            # an NFL chapter shorter than this is an intro or a tease

PROMPT = """You are labelling the chapters of a sports podcast episode so that only NFL
discussion gets transcribed.

Episode: {title}
Summary: {summary}

Chapters:
{chapters}

Label every chapter:
- "nfl": about the NFL (pro football) -- games, players, teams, coaches, trades,
  injuries, fantasy football, NFL betting picks.
- "mixed": a mailbag or grab-bag likely to include NFL among other topics.
- "other": anything else -- NBA, MLB, WNBA, soccer, college football (including
  the NFL draft and college prospects), high school football, movies, books
  and TV (including football movies and shows), life advice, the intro.

Use the summary to resolve vague titles: a "picks" segment is NFL if the
summary says the picks are for an NFL week."""

SCHEMA = {
    "type": "ARRAY",
    "items": {"type": "OBJECT",
              "properties": {"n": {"type": "INTEGER"},
                             "label": {"type": "STRING", "enum": ["nfl", "mixed", "other"]}},
              "required": ["n", "label"]},
}


def label(ep: Episode, client) -> list[str]:
    """One label per chapter, in order."""
    from google.genai import types
    if not ep.chapters:
        return []
    listing = "\n".join(
        f"[{i}] {c.section + ' / ' if c.section and c.section != c.title else ''}{c.title}"
        f" ({(c.end - c.start) // 60} min)"
        for i, c in enumerate(ep.chapters, 1))
    summary = ep.description.split("\n")[0][:1500]
    resp = client.models.generate_content(
        model=MODEL, contents=PROMPT.format(title=ep.title, summary=summary, chapters=listing),
        config=types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json", response_schema=SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))
    got = {g["n"]: g["label"] for g in json.loads(resp.text)}
    missing = [i for i in range(1, len(ep.chapters) + 1) if i not in got]
    if missing:
        raise ValueError(f"{ep.episode_id}: no label for chapters {missing}")
    return [got[i] for i in range(1, len(ep.chapters) + 1)]


def keep_ranges(ep: Episode, labels: list[str]) -> list[tuple[int, int, str]]:
    """(start, end, label) ranges to transcribe, adjacent kept chapters merged."""
    out: list[list] = []
    for c, lab in zip(ep.chapters, labels):
        if lab not in KEEP or c.end - c.start < MIN_SECONDS:
            continue
        if out and out[-1][1] == c.start and out[-1][2] == lab:
            out[-1][1] = c.end
        else:
            out.append([c.start, c.end, lab])
    return [tuple(r) for r in out]
