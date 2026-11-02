"""High / medium / low on each news claim, decided by code, never by the model.

Why not ask Gemini how sure it is: a model's stated confidence does not track
how often it is right, and it sounds surest when it is wrong. The 2026-11-01
report stated "Breece Hall is out, which helps the Broncos DST" with full
confidence -- Hall is a Jet. Every signal below would have marked that claim
low: a league feed, a player the slate did not know, a team tag that was a
name-match guess.

The signals are facts the pipeline already has:
    placed_by_slate  the DK slate puts this player on this team (not a guess
                     from which feed the article came through)
    own_feed         ESPN team news or the team's official site, not a
                     league-wide feed filed under a team by name match
    sources          how many different outlets report something about him
    age_h            hours since the article was published
    claim_check      the second check agreed the summary says what the quote says

SHADOW MODE: the label is logged (jsonPayload.event="claim_confidence") and
not shown to the agent or the reader until graded weeks show high claims
really are right more often than low ones. CONFIDENCE_VISIBLE=1 turns it on.
"""

from __future__ import annotations

import json
import os

VISIBLE = os.environ.get("CONFIDENCE_VISIBLE", "0") == "1"
FRESH_H = 24      # high needs news from the last day
STALE_H = 36      # older than this is low: practice reports and statuses move
LABELS = ("low", "medium", "high")


def claim_confidence(claim: dict, sources: int) -> tuple[str, list[str]]:
    """(label, reasons) for one verified claim. Pure code, no model.

    `claim` carries placed_by_slate, own_feed, age_h and claim_check;
    `sources` is how many distinct outlets have a kept claim on this player."""
    why = []
    if not claim.get("placed_by_slate"):
        why.append("player not on the slate; team taken from the article")
    if claim.get("claim_check") != "supports":
        why.append(f"summary not confirmed by the quote ({claim.get('claim_check')})")
    age = claim.get("age_h")
    if age is None or age > STALE_H:
        why.append(f"news is {age:.0f}h old" if age is not None else "publish time unknown")
    if why:
        return "low", why

    if age <= FRESH_H and (claim.get("own_feed") or sources >= 2):
        return "high", ["on the slate", "summary matches quote", f"{age:.0f}h old",
                        "team's own feed" if claim.get("own_feed") else f"{sources} outlets agree"]
    return "medium", ["on the slate", "summary matches quote",
                      f"{age:.0f}h old" + ("" if age <= FRESH_H else " (over a day)"),
                      "single league-feed source" if not claim.get("own_feed") and sources < 2
                      else "corroborated"]


def label_claims(claims: list[dict], norm=lambda s: s.lower()) -> list[dict]:
    """Label every claim, log each label, and return claims for the agent.

    The scoring fields are internal: they are stripped before the agent sees
    the claim. The label itself rides along only when VISIBLE."""
    outlets: dict[str, set[str]] = {}
    for c in claims:
        outlets.setdefault(norm(c["player"]), set()).add(c.get("source", ""))
    out = []
    for c in claims:
        label, why = claim_confidence(c, len(outlets[norm(c["player"])]))
        _log(c, label, why)
        c = {k: v for k, v in c.items() if k not in ("placed_by_slate", "own_feed", "age_h")}
        if VISIBLE:
            c["confidence"], c["confidence_why"] = label, why
        out.append(c)
    return out


def _log(claim: dict, label: str, why: list[str]) -> None:
    """One line per claim. Agent Engine sends stdout to Cloud Logging, where a
    JSON line becomes jsonPayload: filter jsonPayload.event="claim_confidence"."""
    print(json.dumps({
        "severity": "INFO",
        "message": f"CLAIM_CONFIDENCE {label} {claim.get('player')} [{claim.get('source')}]",
        "event": "claim_confidence", "label": label, "why": why,
        "player": claim.get("player"), "team": claim.get("team"), "kind": claim.get("kind"),
        "claim": claim.get("claim"), "source": claim.get("source"), "url": claim.get("url"),
        "placed_by_slate": claim.get("placed_by_slate"), "own_feed": claim.get("own_feed"),
        "age_h": claim.get("age_h"), "claim_check": claim.get("claim_check"),
    }, default=str), flush=True)
