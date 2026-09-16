"""The live DraftKings slate, fetched at most once per TTL window.

This replaces the unbounded module-level `_SLATE` dict in src/agent.py.

Why a cache exists at all: one agent turn can call three or four tools, and
every one of them needs the slate. Without caching, a single question to the
agent is four round trips to DraftKings' undocumented endpoint.

Why the old one was wrong: on a laptop the process answers once and exits, so a
dict that never expires is harmless. On Agent Runtime one warm process serves
requests for hours. The slate fetched at 10:00 AM would still be sitting in that
dict at 11:30, reporting a player Questionable forty minutes after he was
downgraded to Out -- which is the entire product failing, silently, in the one
direction that matters. The bug cannot reproduce locally; it only exists once
the process outlives the request.

A short TTL keeps the within-turn saving and drops the across-turn staleness.

Concurrency note: Agent Runtime can serve requests on several threads, and there
is no lock here. The worst case is two threads fetching the same slate at once
and one overwriting the other with equivalent data -- wasteful, not incorrect.
A lock would be the fix if the fetch were expensive or the write non-atomic.
"""

from __future__ import annotations

import os
import time

import pandas as pd

from dk_client import fetch_draftables, list_slates, pick_main_slate
from predict_slate import normalize_name

# Seconds a fetched slate stays usable. Long enough to cover one multi-tool
# reasoning turn, short enough that the next question re-reads the wire.
SLATE_TTL = float(os.environ.get("SLATE_TTL_SECONDS", "90"))

_slates: dict[int, dict] = {}
_draft_group_cache: dict = {}


def _draft_group() -> int:
    """Resolve which DK draft group to read, caching the lookup.

    `list_slates()` is itself a network call, so resolving this on every tool
    call would undo most of what the slate cache buys.
    """
    pinned = int(os.environ.get("DRAFT_GROUP") or 0)
    if pinned:
        return pinned

    now = time.monotonic()
    if _draft_group_cache and now - _draft_group_cache["at"] < SLATE_TTL:
        return _draft_group_cache["dg"]

    dg = pick_main_slate(list_slates())
    _draft_group_cache.update(dg=dg, at=now)
    return dg


def slate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (players, games) for the current slate, fetching if stale."""
    dg = _draft_group()
    now = time.monotonic()

    cached = _slates.get(dg)
    if cached is not None and now - cached["at"] < SLATE_TTL:
        return cached["players"], cached["games"]

    players, games = fetch_draftables(dg)
    players["jn"] = normalize_name(players["name"])
    _slates[dg] = {"players": players, "games": games, "at": now}
    return players, games


def my_lineup_names() -> list[str]:
    """The user's rostered players, from the MY_LINEUP env var."""
    return [n.strip() for n in os.environ.get("MY_LINEUP", "").split(",") if n.strip()]
