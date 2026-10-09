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
from espn_client import fetch_injuries, fetch_scoreboard
from predict_slate import normalize_name

# Seconds a fetched slate stays usable. Long enough to cover one multi-tool
# reasoning turn, short enough that the next question re-reads the wire.
SLATE_TTL = float(os.environ.get("SLATE_TTL_SECONDS", "90"))

_slates: dict[int, dict] = {}
_draft_group_cache: dict = {}
_espn: dict = {}


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


def espn_injuries() -> pd.DataFrame:
    """League-wide injury report from ESPN, cached on the same TTL as the slate.

    DK's draftables call still supplies salary and roster structure -- this is
    consulted separately and merged in by name, not used to replace `slate()`.
    """
    now = time.monotonic()
    cached = _espn.get("injuries")
    if cached is not None and now - cached["at"] < SLATE_TTL:
        return cached["df"]

    df = fetch_injuries()
    df["jn"] = normalize_name(df["name"])
    _espn["injuries"] = {"df": df, "at": now}
    return df


def espn_scoreboard() -> pd.DataFrame:
    """This week's games from ESPN -- venue, dome flag, weather if posted."""
    now = time.monotonic()
    cached = _espn.get("scoreboard")
    if cached is not None and now - cached["at"] < SLATE_TTL:
        return cached["df"]

    df = fetch_scoreboard()
    _espn["scoreboard"] = {"df": df, "at": now}
    return df


# Session-state key the scheduled client writes the lineup under. One name,
# shared by the writer (run_report.py) and the reader (here).
LINEUP_STATE_KEY = "my_lineup"


def my_lineup_names(tool_context=None) -> list[str]:
    """The user's rostered players: this request's session state, else MY_LINEUP.

    Session state is the production path. The scheduled client reads the
    lineup from a Google Sheet and creates the session with it, so the lineup
    arrives WITH the request instead of living in the deployment.

    MY_LINEUP used to be the only path, and `adk deploy` bakes gameday_agent/.env
    into the deployment -- so the lineup was frozen at whatever it was on deploy
    day. It was last set 9/20, and every daily report for three weeks checked
    that roster while the user had moved on. Changing a lineup meant a
    four-minute redeploy from a laptop, which in practice meant never.

    The env var stays as the fallback for `adk web` and hand-run tools, where
    there is no client to supply state. It should NOT be present in the .env
    that gets deployed -- if it is, a request that somehow arrives without state
    silently checks a stale roster again instead of failing.
    """
    state = getattr(tool_context, "state", None) if tool_context is not None else None
    if state is not None:
        names = state.get(LINEUP_STATE_KEY)
        if names:
            return [str(n).strip() for n in names if str(n).strip()]
    return [n.strip() for n in os.environ.get("MY_LINEUP", "").split(",") if n.strip()]
