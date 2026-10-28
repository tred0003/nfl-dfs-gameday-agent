"""Podcast feeds: which shows, their episodes, and each episode's chapter list.

Every show is found through Apple Podcasts, but Apple only lists the show --
its in-app transcripts have no API. The lookup API hands back the show's RSS
feed, and the feed has what the archive needs: the audio file, the publish
time, and the description.

  python podcast_rag/feed.py              # last few episodes of every show

None of the four feeds carries a <podcast:transcript> tag (checked
2026-10-28), so every episode is transcribed here. `filter` says how much:

  all       the whole show is NFL (the two Ringer shows): transcribe it all.
  segments  a mixed show (Simmons, Russillo): the description lists chapters
            with start times, segments.py labels each one NFL or not, and only
            the NFL ranges are cut out and transcribed. `last` then counts
            episodes that HAVE NFL content, not the newest N episodes.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime

import requests

SHOWS = {
    "ringer-fantasy": {"name": "The Ringer Fantasy Football Show", "apple_id": 1523722173,
                       "feed": "https://feeds.megaphone.fm/ringer-fantasy-football-show",
                       "filter": "all", "last": 20, "lens": "fantasy"},
    "ringer-nfl": {"name": "The Ringer NFL Show", "apple_id": 1109282822,
                   "feed": "https://feeds.megaphone.fm/the-ringer-nfl-show",
                   "filter": "all", "last": 20, "lens": "nfl"},
    "simmons": {"name": "The Bill Simmons Podcast", "apple_id": 1043699613,
                "feed": "https://feeds.megaphone.fm/the-bill-simmons-podcast",
                "filter": "segments", "last": 20, "lens": "nfl"},
    "russillo": {"name": "The Ryen Russillo Show", "apple_id": 1845980031,
                 "feed": "https://mcsorleys.barstoolsports.com/feed/ryen-russillo-show",
                 "filter": "segments", "last": 20, "lens": "nfl"},
}

ITUNES = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
LOOKUP = "https://itunes.apple.com/lookup?id={id}"
# "(17:02) The Pick Six", "(01:36:18) Ringer 107 picks" -- Simmons. Russillo
# also puts bare times on their own line under a section heading:
#   CFB TOP 12
#   (1:35:15)
STAMP = re.compile(r"^\(?(\d{1,2}(?::\d{2}){1,2})\)?\s*(.*)$")


def feed_url(apple_id: int) -> str:
    """Apple Podcasts id (the number in the podcasts.apple.com link) -> RSS feed."""
    return requests.get(LOOKUP.format(id=apple_id), timeout=30).json()["results"][0]["feedUrl"]


@dataclass
class Chapter:
    start: int              # seconds
    end: int                # seconds; the next chapter's start, or the episode's end
    title: str
    section: str = ""       # Russillo's heading over a run of chapters ("NFL WEEK 7 w/ NATE TICE")


@dataclass
class Episode:
    show: str
    guid: str
    title: str
    published: datetime
    duration: int           # seconds
    audio_url: str
    description: str
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def episode_id(self) -> str:
        return f"pod:{self.show}:{re.sub(r'[^A-Za-z0-9-]', '', self.guid)[:40]}"


def _seconds(s: str) -> int:
    t = 0
    for part in s.split(":"):
        t = t * 60 + int(part)
    return t


def _text(description: str) -> str:
    """Feed HTML -> lines. <p> and <br> become line breaks, since the chapter
    list is one stamp per line."""
    s = re.sub(r"(?i)<br\s*/?>|</p>", "\n", description or "")
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def chapters(text: str, duration: int) -> list[Chapter]:
    out, section = [], ""
    for line in text.splitlines():
        m = STAMP.match(line)
        if m:
            title = m.group(2).strip() or section        # bare "(1:35:15)" under a heading
            out.append(Chapter(_seconds(m.group(1)), 0, title, section))
        elif len(line) < 60 and not line.endswith((".", "!", "?")):
            # A short line that is not a stamp: Russillo's section heading
            # ("NFL WEEK 7 w/ NATE TICE"). The summary paragraph is longer.
            section = line
    for a, b in zip(out, out[1:]):
        a.end = b.start
    if out:
        out[-1].end = duration
    # Feeds only ever list chapters in order; anything else is a parse error.
    return [c for c in out if c.end > c.start]


def episodes(show: str, limit: int | None = None) -> list[Episode]:
    """Newest first."""
    cfg = SHOWS[show]
    r = requests.get(cfg["feed"], timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    out = []
    for item in ET.fromstring(r.content).findall("./channel/item")[:limit]:
        enc = item.find("enclosure")
        if enc is None:
            continue
        text = _text(item.findtext("description") or item.findtext(f"{ITUNES}summary") or "")
        dur = _seconds(item.findtext(f"{ITUNES}duration") or "0")
        out.append(Episode(
            show=show, guid=item.findtext("guid") or enc.get("url"),
            title=html.unescape(item.findtext("title") or "").strip(),
            published=parsedate_to_datetime(item.findtext("pubDate")),
            duration=dur, audio_url=enc.get("url"), description=text,
            chapters=chapters(text, dur)))
    return out


if __name__ == "__main__":
    for key in SHOWS:
        for ep in episodes(key, limit=3):
            print(f"{key:15} {ep.published:%m-%d} {ep.duration // 60:4}m "
                  f"{len(ep.chapters):3} ch  {ep.title[:70]}")
