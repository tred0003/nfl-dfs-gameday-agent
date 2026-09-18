"""DraftKings slate ingestion.

Everything that knows about DK's undocumented JSON shape lives in THIS file.
The rest of the project consumes tidy DataFrames, so when DK changes their
payload (and they will) only this module breaks.

There is no official DraftKings API. These endpoints are unauthenticated but
undocumented and unsupported. Personal research use only -- do not redistribute
the pulled data, and keep the request rate polite.

Run: python src/dk_client.py            # fetches the next Sunday main slate
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

LOBBY_URL = "https://www.draftkings.com/lobby/getcontests?sport=NFL"
DRAFTABLES_URL = "https://api.draftkings.com/draftgroups/v1/draftgroups/{dg}/draftables"

TIMEOUT = 30

# No custom headers, on purpose -- especially no User-Agent.
#
# This endpoint 403s a request carrying a hand-set User-Agent and serves one
# that carries requests' own default (`python-requests/x.y.z`). This module
# used to send "Mozilla/5.0 (research; personal use)", which meant every call
# from this code failed while manual curl checks (which send curl's own UA)
# succeeded -- so the block looked like it was aimed at the machine rather
# than at the request. It wasn't. Same finding as espn_client.py, found the
# same way, one day apart.
#
# Don't add a User-Agent here without re-testing against the live endpoint.

SLATES_DIR = Path(__file__).resolve().parent.parent / "data" / "slates"

# DK marks each RB/WR/TE twice: once in its own roster slot and once as FLEX.
# Deduping on these keeps one row per player.
ROSTER_SLOTS = {66: "QB", 67: "RB", 68: "WR", 69: "TE", 70: "FLEX", 71: "DST"}

# draftStatAttributes ids seen on NFL slates
STAT_FPPG = 90      # DK's own fantasy-points-per-game
STAT_OPRK = -2      # opponent rank vs this position


def _get_json(url: str) -> dict:
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _parse_dotnet_date(s: str) -> datetime | None:
    """DK lobby returns '/Date(1789318800000)/' (ms since epoch, UTC)."""
    m = re.search(r"/Date\((\d+)\)/", s or "")
    if not m:
        return None
    return datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc)


def list_slates() -> pd.DataFrame:
    """All currently-offered NFL draft groups, one row per slate."""
    data = _get_json(LOBBY_URL)
    rows = {}
    for c in data.get("Contests", []):
        dg = c.get("dg")
        if dg is None:
            continue
        # keep the richest contest per draft group -- prize size is what
        # distinguishes the main slate from same-time side slates
        prior = rows.get(dg)
        if prior is not None and (c.get("po") or 0) <= prior["top_prize"]:
            continue
        rows[dg] = {
            "draft_group_id": dg,
            "game_type": c.get("gameType"),
            "slate_label": c.get("sdstring"),          # e.g. "Sun 1:00PM"
            "start_time_utc": _parse_dotnet_date(c.get("sd")),
            "example_contest": c.get("n"),
            "top_prize": c.get("po") or 0,
        }
    df = pd.DataFrame(rows.values())
    return df.sort_values("start_time_utc").reset_index(drop=True)


def pick_main_slate(slates: pd.DataFrame) -> int:
    """The Sunday main slate -- the one you actually play.

    DK does not label a "main slate", and several Classic slates share a Sun
    1:00PM start (Sun-Mon specials, afternoon-only, etc). Picking the earliest
    silently returns the wrong one. The reliable signal is prize money: the main
    slate always hosts the week's largest tournament.

    Prize money alone is not enough. DK advertises contests against draft
    groups that have no players in them yet -- a $6,000,000 "Main Event"
    outranked the real $3,000,000 Sunday slate on 2026-09-18 and returned a
    200 with zero draftables. Sorting on prize alone picks that one every
    time, and an empty slate is indistinguishable downstream from a slate
    where none of your players are listed.

    So candidates are tried richest-first and the first one that actually has
    players wins. That costs one extra request per empty candidate, which is
    cheap next to silently returning a slate with nobody on it.

    Still a heuristic. Pin PIN_DRAFT_GROUP or pass --dg for anything that
    matters, and check the printed slate before trusting it.
    """
    pinned = os.environ.get("PIN_DRAFT_GROUP")
    if pinned:
        return int(pinned)

    m = slates[
        (slates["game_type"] == "Classic")
        & (slates["slate_label"].fillna("").str.startswith("Sun"))
    ]
    if m.empty:
        raise RuntimeError("no Sunday Classic slate found; inspect list_slates()")

    skipped = []
    for _, c in m.sort_values("top_prize", ascending=False).iterrows():
        dg = int(c["draft_group_id"])
        n = len(_get_json(DRAFTABLES_URL.format(dg=dg)).get("draftables", []))
        if n == 0:
            skipped.append(f"dg{dg} (${c['top_prize']:,.0f}, no players yet)")
            continue
        if skipped:
            print(f"  skipped {', '.join(skipped)}")
        print(f"  main slate -> dg{dg} "
              f"({c['slate_label']}, top prize ${c['top_prize']:,.0f}, "
              f"{n} draftables): {c['example_contest']}")
        return dg

    raise RuntimeError(
        f"every Sunday Classic slate came back empty ({', '.join(skipped)}); "
        "salaries are probably not released yet -- inspect list_slates()"
    )


def fetch_draftables(draft_group_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (players, games) for one slate.

    players: one row per player (FLEX duplicates removed)
    games:   one row per game, incl. venue / dome / weather icon
    """
    raw = _get_json(DRAFTABLES_URL.format(dg=draft_group_id))

    games = []
    for c in raw.get("competitions", []):
        w = c.get("weather") or {}
        games.append({
            "competition_id": c.get("competitionId"),
            "game": c.get("name"),
            "home_team": (c.get("homeTeam") or {}).get("abbreviation"),
            "away_team": (c.get("awayTeam") or {}).get("abbreviation"),
            "start_time_utc": c.get("startTime"),
            "venue": c.get("venue"),
            "is_dome": w.get("isDome"),
            "weather_icon": w.get("icon"),
            "depth_charts_available": c.get("depthChartsAvailable"),
        })
    games_df = pd.DataFrame(games)

    players = []
    for d in raw.get("draftables", []):
        stats = {a["id"]: a.get("value") for a in d.get("draftStatAttributes", [])}
        comp = d.get("competition") or {}
        players.append({
            "draft_group_id": draft_group_id,
            "player_dk_id": d.get("playerDkId"),
            "name": d.get("displayName"),
            "position": d.get("position"),
            "roster_slot": ROSTER_SLOTS.get(d.get("rosterSlotId"), d.get("rosterSlotId")),
            "team": d.get("teamAbbreviation"),
            "salary": d.get("salary"),
            "status": d.get("status"),          # None / Q / OUT / IR
            "news_status": d.get("newsStatus"),  # None / Recent / Breaking
            "dk_fppg": pd.to_numeric(stats.get(STAT_FPPG), errors="coerce"),
            "dk_oprk": pd.to_numeric(
                str(stats.get(STAT_OPRK, "")).rstrip("stndrh"), errors="coerce"
            ),
            "competition_id": comp.get("competitionId"),
            "game": comp.get("name"),
            "start_time_utc": comp.get("startTime"),
            "is_disabled": d.get("isDisabled"),
        })
    df = pd.DataFrame(players)

    # Drop the FLEX shadow rows; keep the true positional row per player.
    df = (
        df.sort_values("roster_slot", key=lambda s: s.eq("FLEX"))
          .drop_duplicates(subset=["player_dk_id"], keep="first")
          .reset_index(drop=True)
    )
    return df, games_df


def snapshot(draft_group_id: int | None = None) -> Path:
    """Pull a slate and write a timestamped snapshot.

    Salaries are fixed at slate release but STATUS changes right up to kickoff.
    Snapshotting with a timestamp is what lets you later train on 'what was
    knowable at time T' instead of leaking Sunday-afternoon news into a
    Wednesday-morning decision.
    """
    slates = list_slates()
    if draft_group_id is None:
        draft_group_id = pick_main_slate(slates)

    players, games = fetch_draftables(draft_group_id)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = SLATES_DIR / f"dg{draft_group_id}"
    out.mkdir(parents=True, exist_ok=True)

    players.to_parquet(out / f"players_{ts}.parquet", index=False)
    games.to_parquet(out / f"games_{ts}.parquet", index=False)
    (out / f"meta_{ts}.json").write_text(json.dumps({
        "draft_group_id": draft_group_id,
        "pulled_at_utc": ts,
        "n_players": len(players),
        "n_games": len(games),
    }, indent=2))

    print(f"slate {draft_group_id}: {len(players)} players, {len(games)} games -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Snapshot a DraftKings NFL slate")
    p.add_argument("--dg", type=int, default=None, help="draft group id (default: Sunday main)")
    p.add_argument("--list", action="store_true", help="list available slates and exit")
    a = p.parse_args()

    if a.list:
        print(list_slates().to_string(index=False))
    else:
        snapshot(a.dg)
