"""High / medium / low on each Ask answer, decided by code, never by the model.

A model's stated confidence does not track how often it is right, so the
label comes from what citations.check() and the evidence already record:

    answer_type   answered / partial / abstained / unsupported / empty
    problems      unknown_ref, uncited, date_mismatch
    cited_share   share of factual sentences that cite a passage
    outlets       distinct sources among the passages the answer cites
    newest        days since the newest cited passage
    model card    whether the XGBoost projection was cited, and whether a
                  named player carries a Q/D injury status (a questionable
                  player who sits scores zero, so that answer is a coin flip
                  no matter how well it is sourced)

SHADOW MODE: the label goes into the response and the weekly exam's rows,
not onto the page. The exam grades every answer against the box score, so
after a few Sundays ask_weekly can show whether high answers really are
right more often than low ones. Only then does the page show it.
"""

from __future__ import annotations

import re

MODEL_SOURCE = "nfl-dfs model"
_STATUS = re.compile(r"injury status (\w+)", re.I)
RISKY_STATUS = {"Q", "D", "QUESTIONABLE", "DOUBTFUL"}


def answer_confidence(checked: dict, evidence: list[dict]) -> tuple[str, list[str]]:
    """(label, reasons) for one checked answer and the evidence it could cite."""
    kind = checked.get("answer_type")
    if kind in ("abstained", "empty", "unsupported"):
        return "low", [f"answer is {kind}"]

    by_ref = {i["ref"]: i for e in evidence for i in e.get("items", []) if i.get("ref")}
    cited = [by_ref[r] for c in checked.get("claims", []) for r in c.get("refs", [])
             if r in by_ref]
    problems = {p["kind"] for p in checked.get("problems", [])}
    share = checked.get("cited_share")
    outlets = {i.get("source") for i in cited if i.get("source")}
    news_ages = [i["days_ago"] for i in cited
                 if isinstance(i.get("days_ago"), int) and not _is_model(i)]
    newest = min(news_ages) if news_ages else None
    model_cited = any(_is_model(i) for i in cited)
    risky = sorted({m.upper() for i in by_ref.values() if _is_model(i)
                    for m in _STATUS.findall(i.get("text", ""))} & RISKY_STATUS)

    why = []
    if problems & {"unknown_ref", "date_mismatch"}:
        why.append("citation problems: " + ", ".join(sorted(problems)))
    if share is None or share < 0.5:
        why.append(f"most sentences uncited (cited share {share})")
    if not cited:
        why.append("nothing cited")
    if why:
        return "low", why

    medium = []
    if kind == "partial":
        medium.append("partial answer")
    if "uncited" in problems:
        medium.append("an uncited sentence")
    if share < 0.9:
        medium.append(f"cited share {share}")
    if len(outlets) < 2:
        medium.append("one source")
    if newest is None or newest > 3:
        medium.append("no news from the last 3 days" if newest is None else f"newest news {newest}d old")
    if risky:
        medium.append(f"a named player is {'/'.join(risky)}")
    if medium:
        return "medium", medium
    return "high", [f"{len(outlets)} sources", f"newest news {newest}d old",
                    "model projection cited" if model_cited else "news only", "every fact cited"]


def _is_model(item: dict) -> bool:
    return str(item.get("source", "")).startswith(MODEL_SOURCE)
