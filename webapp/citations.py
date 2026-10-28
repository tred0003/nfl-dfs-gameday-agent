"""The Ask answer's output contract: every fact cites the passage it came from.

The model writes plain sentences with refs like [P3] -- the numbers ask.py
put on each passage the tools returned. This module turns that text into a
checked structure, with no model involved:

    "Limited Wednesday with the ankle [P2]. **Trending better** [P3][P5]."
        -> claims  [{text, refs: ["P2"]}, {text, refs: ["P3","P5"]}]
        -> problems, each one a rule a deterministic check can enforce:
             unknown_ref    a ref that was never retrieved this run
                            (removed from the answer -- a fabricated source)
             uncited        a factual sentence with no ref
             date_mismatch  a date in the sentence that none of its cited
                            passages carries ("Oct 12" citing an Oct 21 note)
        -> answer_type answered  cited facts
                       partial   "No archived news found" on the ask, plus
                                 cited related facts
                       abstained no facts claimed, nothing cited
                       unsupported facts claimed, none cited (the bad one)

Why inline refs and not a JSON answer: Gemini 2.5 does not combine function
calling with a JSON response schema in one request, so a JSON answer would
cost one more model call per question. Inline refs keep the agent's last
turn as the answer, and the structure comes from this parser, which is
cheaper and testable. Both Ask designs (agent and pipeline) share it.

What this does NOT check: that the passage actually SUPPORTS the sentence.
That is a judgment, measured offline by the grounding eval, not here.
"""

from __future__ import annotations

import re

REF = re.compile(r"\[(P\d+(?:\s*,\s*P\d+)*)\]")
_REF_AFTER_STOP = re.compile(r"([.!?])((?:\s*\[P\d+(?:\s*,\s*P\d+)*\])+)")
# A sentence ends at . ! ? before whitespace and a capital, quote or "**".
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"*(])")
# ...except after an initial or an abbreviation: "A.J. Brown", "C.J.",
# "Beckham Jr. was", "3 p.m. Sunday". A piece ending in one is glued back on.
_NOT_AN_END = re.compile(r"(?:\b[A-Z]\.|\b(?:Jr|Sr|St|vs|Mr|No|p\.m|a\.m)\.)$")
_MONTHS = ("Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
           "Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?")
# A month, then a day that is not a count ("in May 2 games ago" is no date).
_DATE = re.compile(rf"\b({_MONTHS})\.?\s+(\d{{1,2}})\b"
                   r"(?!\s*(?:games?|weeks?|days?|yards?|catches|targets|carries|snaps|points|times)\b)")
_DAYS_AGO = re.compile(r"\b(\d{1,2})\s+days?\s+ago\b", re.I)
# Sentences that need no source: the archive's own limits, and the pointer to
# live news the instruction asks for. Anything else uncited is a problem.
_CAVEAT = re.compile(r"\b(archive|live news|your report|no archived news|newer news)\b"
                     r"|\b(?:check|see|look at|watch)\b.{0,40}\b(?:ESPN|live|report|news|status|updates?)\b"
                     r"|^\**[^.!?]{1,60}:\**$",          # a header line ("**Puka Nacua:**")
                     re.I)
ABSTAIN = re.compile(r"^\W*no archived news (?:was )?found", re.I)


def number_items(result: dict, refs: dict[str, str]) -> None:
    """Give every note/passage in one tool result a ref ("P1", "P2" ...).

    `refs` maps a passage's identity to its ref and is shared across all the
    tool calls of one question, so the same passage returned by two calls
    keeps one number. Identity is the chunk id, or the text when an older
    server sent no id."""
    items = [n for p in result.get("players", []) for n in p.get("notes", [])]
    items += result.get("passages", [])
    for item in items:
        key = item.get("id") or item.get("text", "")
        if key not in refs:
            refs[key] = f"P{len(refs) + 1}"
        item["ref"] = refs[key]


def _sentences(text: str) -> list[str]:
    # "fact. [P2]" -> "fact [P2]." so a ref stays with the sentence before it.
    text = _REF_AFTER_STOP.sub(lambda m: f"{m.group(2)}{m.group(1)}", text.strip())
    out = []
    for line in text.splitlines():
        glue = False
        for piece in (x.strip() for x in _SPLIT.split(line)):
            if not piece:
                continue
            if glue:
                out[-1] = f"{out[-1]} {piece}"
            else:
                out.append(piece)
            glue = bool(_NOT_AN_END.search(out[-1]))
    return out


def _date_ok(sentence: str, cited: list[dict]) -> bool:
    """Every date the sentence states belongs to a cited passage: its publish
    date, a date written inside it, or its age in days (within one)."""
    for mon, day in _DATE.findall(sentence):
        said = re.compile(rf"\b{mon}[a-z]*\.?\s+0?{int(day)}\b")   # "Oct 2" is not "Oct 21"
        if not any(said.search(c.get("date", "")) or said.search(c.get("text", ""))
                   for c in cited):
            return False
    for n in _DAYS_AGO.findall(sentence):
        if not any(isinstance(c.get("days_ago"), int) and abs(c["days_ago"] - int(n)) <= 1
                   for c in cited):
            return False
    return True


def check(answer: str, items: list[dict]) -> dict:
    """The answer, checked against the passages it could cite.

    `items` are the evidence passages, each with its `ref`. Returns the answer
    with unknown refs removed, the claims, the problems and `abstained`."""
    by_ref = {i["ref"]: i for i in items if i.get("ref")}
    claims, problems = [], []
    for sentence in _sentences(answer or ""):
        refs = [r.strip() for group in REF.findall(sentence) for r in group.split(",")]
        known = list(dict.fromkeys(r for r in refs if r in by_ref))
        for r in dict.fromkeys(r for r in refs if r not in by_ref):
            problems.append({"kind": "unknown_ref", "ref": r, "sentence": sentence})
        text = re.sub(r"\s+([.!?,])", r"\1", REF.sub("", sentence)).strip()
        if not known and not _CAVEAT.search(text):
            problems.append({"kind": "uncited", "sentence": text})
        if known and not _date_ok(text, [by_ref[r] for r in known]):
            problems.append({"kind": "date_mismatch", "sentence": text, "refs": known})
        claims.append({"text": text, "refs": known})

    def drop_unknown(m: re.Match) -> str:
        kept = [r.strip() for r in m.group(1).split(",") if r.strip() in by_ref]
        return f"[{', '.join(kept)}]" if kept else ""

    clean = re.sub(r"[ \t]+([.!?,])", r"\1", REF.sub(drop_unknown, answer or "")).strip()
    factual = [c for c in claims if not _CAVEAT.search(c["text"]) or c["refs"]]
    cited = any(c["refs"] for c in claims)
    uncited = any(p["kind"] == "uncited" for p in problems)
    # Four outcomes, not a yes/no. Seen live on 2026-10-28: "No archived news
    # found about an ankle injury. He has a mild hamstring strain [P1]" is a
    # useful PARTIAL answer, and "This archive has NFL news only, so nothing
    # on the Yankees" is an abstention without the agreed opening words.
    if cited:
        kind = "partial" if ABSTAIN.search(clean) else "answered"
    elif ABSTAIN.search(clean) or not uncited:
        # "No archived news found ..." then advice in any wording is still a
        # refusal: nothing is claimed from a source, so nothing is made up.
        kind = "abstained"
    else:
        kind = "unsupported"
    return {"answer": clean, "claims": claims, "problems": problems,
            "answer_type": kind, "abstained": kind == "abstained",
            "cited_share": round(sum(bool(c["refs"]) for c in factual) / len(factual), 3)
            if factual else None}
