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

# A Sunday main slate is the whole 1 PM + 4 PM window: 9-13 games, ~10 even on
# a heavy bye week. Side slates (afternoon only, early only) carry 2-5. On
# 2026-11-01 at 12:31 ET the main slate's contests had left the lobby and the
# richest one left was a 3-game "Afternoon Only" slate -- every report checked
# 9-player lineups against 3 games and called 6 players "not on the slate".
MIN_MAIN_GAMES = int(os.environ.get("MIN_MAIN_GAMES", "6"))

# Where the project-slate job writes the slate it projected. It runs Thu 09:00,
# Fri 19:00 and Sun 11:00 ET -- before lock, while the lobby still lists the
# main slate -- so its pick is the most trustworthy one available on Sunday.
PROJECTIONS_TABLE = os.environ.get(
    "PROJECTIONS_TABLE", "fantasy-football-ml-503718.dfs.projections")
# A saved slate counts as current until this long after its last kickoff.
SAVED_GRACE_HOURS = 5

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

    Prize money also depends on WHEN you ask. The lobby lists open contests,
    and on Sunday the main slate's big ones fill and drop out before 1 PM. So
    the slate the projection job saved this week wins while it is current,
    and the lobby candidate must carry at least MIN_MAIN_GAMES games. A side
    slate that sneaks through is a report about the wrong games that still
    looks well formed -- raising is better than that.

    Still a heuristic. Pin PIN_DRAFT_GROUP or pass --dg for anything that
    matters, and check the printed slate before trusting it.
    """
    pinned = os.environ.get("PIN_DRAFT_GROUP")
    if pinned:
        return int(pinned)

    saved = saved_main_slate()
    if saved is not None:
        print(f"  main slate -> dg{saved} (saved by the projection job)")
        return saved

    m = slates[
        (slates["game_type"] == "Classic")
        & (slates["slate_label"].fillna("").str.startswith("Sun"))
    ]
    if m.empty:
        raise RuntimeError("no Sunday Classic slate found; inspect list_slates()")

    skipped = []
    for _, c in m.sort_values("top_prize", ascending=False).iterrows():
        dg = int(c["draft_group_id"])
        raw = _get_json(DRAFTABLES_URL.format(dg=dg))
        n = len(raw.get("draftables", []))
        games = len(raw.get("competitions", []))
        if n == 0:
            skipped.append(f"dg{dg} (${c['top_prize']:,.0f}, no players yet)")
            continue
        if games < MIN_MAIN_GAMES:
            skipped.append(f"dg{dg} (${c['top_prize']:,.0f}, only {games} games: "
                           f"{c['example_contest']})")
            continue
        if skipped:
            print(f"  skipped {', '.join(skipped)}")
        print(f"  main slate -> dg{dg} "
              f"({c['slate_label']}, top prize ${c['top_prize']:,.0f}, "
              f"{n} draftables, {games} games): {c['example_contest']}")
        return dg

    raise RuntimeError(
        f"no Sunday Classic slate with players and {MIN_MAIN_GAMES}+ games "
        f"({', '.join(skipped)}); salaries may not be released yet, or the main "
        "slate has left the lobby -- inspect list_slates() or set PIN_DRAFT_GROUP"
    )


def _last_kickoff(competitions: list[dict]) -> datetime | None:
    times = pd.to_datetime([c.get("startTime") for c in competitions],
                           utc=True, errors="coerce").dropna()
    return times.max().to_pydatetime() if len(times) else None


def saved_main_slate(now: datetime | None = None) -> int | None:
    """The draft group the projection job last projected, if it is still this week's.

    None whenever it cannot say -- no BigQuery library or permission (laptop,
    CI), no run in the last 8 days, or a slate whose last game kicked off more
    than SAVED_GRACE_HOURS ago (Monday to Thursday morning: last week's slate,
    and the lobby is right again). Never raises: this is a hint in front of
    the lobby, not a new way for the report to fail.
    """
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECTIONS_TABLE.split(".")[0])
        rows = list(client.query(f"""
            SELECT draft_group_id FROM `{PROJECTIONS_TABLE}`
            WHERE run_ts > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 8 DAY)
            ORDER BY run_ts DESC LIMIT 1""").result(timeout=20))
        if not rows:
            return None
        dg = int(rows[0]["draft_group_id"])
        last = _last_kickoff(_get_json(DRAFTABLES_URL.format(dg=dg)).get("competitions", []))
    except Exception as e:  # noqa: BLE001 -- any failure falls back to the lobby
        print(f"  saved slate unavailable ({type(e).__name__}); using the lobby")
        return None
    now = now or datetime.now(timezone.utc)
    if last is None or (now - last).total_seconds() > SAVED_GRACE_HOURS * 3600:
        return None
    return dg


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
