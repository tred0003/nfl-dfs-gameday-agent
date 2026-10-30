"""Lineup import: a DraftKings screenshot in, this week's players out.

    browser ──image──► gameday-web ──► Document AI (OCR processor) ──► lines of text
                                                                         │
                       slots on screen ◄── user confirms ◄── match to this week's slate

Typing nine names is where wrong lineups come from. A screenshot of the
DraftKings lineup screen already has them, spelled DraftKings' way.

WHY OCR AND NOT AN EXTRACTOR OR GEMINI. Document AI's OCR processor returns
lines of text with a confidence each, and no model decides anything. The
"schema" is this week's slate: 554 known names. A line either is one of them
or it is not, so the matching is code, the same rule as the rest of the
project: when a step has one right answer, keep it out of the model.

WHY EXACT NAMES ONLY. app.js learned this one: a first-initial guess matched
"J Daniel's" to Jalon Daniels (a backup) when Jayden was not on the slate.
A confident wrong player is worse than a flagged unknown one. So a line
matches only when it contains a slate name after normalising case,
punctuation and suffixes ("A.J. Brown" = "AJ Brown", "Aaron Jones Sr." =
"Aaron Jones"). A name-shaped line that matches nothing is returned as
not_recognised, for the user to fix by hand.

NOTHING IS SAVED HERE. The route returns what it found; the page fills the
slots and the user's own save (PUT /api/lineup) is the only write.
"""

from __future__ import annotations

import os
import re

# Document AI's OCR processor in advisor-agent-2026 (multi-region "us").
PROCESSOR = os.environ.get(
    "DOCAI_PROCESSOR",
    "projects/advisor-agent-2026/locations/us/processors/6502beabda51e08f")
MAX_BYTES = 5 * 1024 * 1024     # a phone screenshot is well under 1 MB
LINEUP_SIZE = 9                 # DK Classic: QB, 2 RB, 3 WR, TE, FLEX, DST
# Magic bytes, so a file is the type it claims before it leaves the app.
SIGNATURES = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/webp": (b"RIFF",),
    "application/pdf": (b"%PDF",),
}
# Words on the DK screen that are shaped like a name but never are one.
SCREEN_WORDS = {"lineup", "salary", "classic", "nfl", "cap", "rem", "avg", "player",
                "fppg", "opp", "flex", "sun", "mon", "thu", "sat", "pm", "am",
                "edit", "export", "contest", "contests", "entries", "my", "upcoming",
                "live", "draft", "team", "projected", "points", "game", "games"}

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_NAMEISH = re.compile(r"^[A-Z][\w.'\-]+(?: [A-Z][\w.'\-]+){1,3}$")


class LineupImportError(Exception):
    """The image could not be read. The caller says so; it never guesses."""


def norm(name: str) -> str:
    n = str(name or "").lower().replace(".", "").replace("'", "").replace("’", "")
    n = n.replace("-", " ")
    return re.sub(r"\s+", " ", _SUFFIX.sub("", n)).strip()


def check_upload(data: bytes, mime: str) -> str:
    """The upload's media type, or LineupImportError for anything else."""
    mime = (mime or "").split(";")[0].strip().lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in SIGNATURES:
        raise LineupImportError("upload a PNG, JPEG or WebP screenshot, or a PDF")
    if not data:
        raise LineupImportError("the upload was empty")
    if len(data) > MAX_BYTES:
        raise LineupImportError(f"the file is over {MAX_BYTES // (1024 * 1024)} MB")
    if not data.startswith(SIGNATURES[mime]):
        raise LineupImportError(f"the file is not really {mime}")
    return mime


class DocAiReader:
    """Lines of text from Document AI's OCR processor: [(text, confidence)]."""

    def __init__(self, processor: str = PROCESSOR):
        self.processor = processor
        self._client = None

    def lines(self, data: bytes, mime: str) -> list[tuple[str, float]]:
        # Lazy import: the unit tests run without the Document AI client.
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai

        if self._client is None:
            location = self.processor.split("/locations/")[1].split("/")[0]
            self._client = documentai.DocumentProcessorServiceClient(
                client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com"))
        try:
            doc = self._client.process_document(
                request=documentai.ProcessRequest(
                    name=self.processor,
                    raw_document=documentai.RawDocument(content=data, mime_type=mime)),
                timeout=30).document
        except Exception as e:  # noqa: BLE001 -- the user gets the reason, not a stack trace
            raise LineupImportError(f"{type(e).__name__}: {str(e)[:150]}") from e

        def text(layout) -> str:
            return "".join(doc.text[int(s.start_index):int(s.end_index)]
                           for s in layout.text_anchor.text_segments).strip()
        return [(text(line.layout), round(float(line.layout.confidence), 3))
                for page in doc.pages for line in page.lines]


def match_lineup(lines: list[tuple[str, float]], players: list[dict]) -> dict:
    """Slate players named in the OCR lines, in screen order, plus what didn't match.

    One pass down the lines. Each line is checked for every slate name it
    contains, longest name first, so "Amon-Ra St. Brown" is not also read
    as a second, shorter name inside it. A name split over two lines
    ("Justin" / "Jefferson") is caught by joining a line with the next when
    neither matched alone. A defense matches by its DK name, the nickname.
    """
    by_key: dict[str, dict] = {}
    for p in players:
        by_key.setdefault(norm(p["name"]), p)
    keys = sorted((k for k in by_key if k), key=len, reverse=True)

    def find(line: str) -> list[str]:
        rest, hits = f" {norm(line)} ", []
        for k in keys:
            if f" {k} " in rest:
                hits.append(k)
                rest = rest.replace(f" {k} ", " | ")
        return hits

    found: list[tuple[str, float]] = []      # (key, confidence) in screen order
    unknown: list[str] = []
    i = 0
    while i < len(lines):
        text, conf = lines[i]
        hits = find(text)
        if not hits and i + 1 < len(lines) and not find(lines[i + 1][0]):
            joined = find(f"{text} {lines[i + 1][0]}")
            if joined:
                found += [(k, min(conf, lines[i + 1][1])) for k in joined]
                i += 2
                continue
        if hits:
            found += [(k, conf) for k in hits]
        elif _NAMEISH.match(text) and not set(norm(text).split()) & SCREEN_WORDS:
            unknown.append(text)
        i += 1

    seen, ordered = set(), []
    for k, conf in found:
        if k not in seen:
            seen.add(k)
            ordered.append((by_key[k], conf))

    warnings = []
    defenses = [p for p, _ in ordered if p["position"] == "DST"]
    if len(defenses) > 1:
        # A nickname can also appear as an opponent or in a header. Two of
        # them is a guess either way, so keep neither and say which.
        ordered = [(p, c) for p, c in ordered if p["position"] != "DST"]
        warnings.append("More than one defense was found ("
                        + ", ".join(p["name"] for p in defenses)
                        + "), so none was filled in. Pick yours by hand.")
    if len(ordered) > LINEUP_SIZE:
        warnings.append(f"Found {len(ordered)} players; kept the first {LINEUP_SIZE} on screen.")
        ordered = ordered[:LINEUP_SIZE]
    if not ordered:
        warnings.append("No player from this week's slate was found in the image.")

    return {
        "players": [{"name": p["name"], "position": p["position"], "team": p.get("team"),
                     "salary": p.get("salary"), "confidence": c} for p, c in ordered],
        "not_recognised": unknown[:6],
        "warnings": warnings,
    }
