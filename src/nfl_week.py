"""Which NFL week is it? Worked out from the calendar, never configured.

Why this exists: WEEK used to be an env var, and `adk deploy agent_engine`
bakes every line of gameday_agent/.env into the deployment. It was set to 2 on
9/20 and never touched again, so for three weeks the deployed agent quoted
Week 2 Vegas totals against Week 5 slates -- and every run exited 0. A value
that has to be remembered weekly will eventually be forgotten; a value the
code can derive should not be a setting at all.

How: nflverse's schedule lists every game's date. The week is the one whose
games span the anchor date (the slate's first kickoff, or today). Between
weeks -- Tuesday and Wednesday, after Monday night and before Thursday --
it is the NEXT week, because every caller here is preparing for the slate
that has not happened yet.

WEEK is still honoured as an explicit override, for replaying a past slate
and for the eval harness, which pins it so a run never touches the network.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# nfl_data_py's import_schedules() is a bare read_csv of this same nflverse
# file (via a mirror). Reading it directly means the scheduled-report image
# does not need nfl_data_py at all just to answer "what week is it".
SCHEDULE_CSV = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

ET = ZoneInfo("America/New_York")

_cache: dict[int, pd.DataFrame] = {}


def _schedule(season: int) -> pd.DataFrame:
    """Game dates for one season: local parquet if present, else nflverse.

    The parquet exists on a laptop that has run fetch_nflverse.py. It does not
    exist on Agent Engine or in a Cloud Run image -- data/ is gitignored and
    never staged -- which is exactly the vegas_for_slate() FileNotFoundError
    that killed get_vegas_context once already. Same guard here.
    """
    if season in _cache:
        return _cache[season]
    try:
        s = pd.read_parquet(RAW / "schedules.parquet", columns=["season", "week", "gameday"])
        s = s[s["season"].eq(season)]
    except (FileNotFoundError, OSError, ValueError):
        s = pd.DataFrame()
    if s.empty:
        s = pd.read_csv(SCHEDULE_CSV, usecols=["season", "week", "gameday", "game_type"])
        # Regular season only. Playoff weeks restart nothing on DK's main
        # slate and nflverse numbers them 19-22, which would be a surprising
        # thing to see in a report header in January.
        s = s[s["season"].eq(season) & s["game_type"].eq("REG")]
    s = s.assign(gameday=pd.to_datetime(s["gameday"]).dt.date)
    _cache[season] = s
    return s


def week_for_date(season: int, anchor: date, schedule: pd.DataFrame | None = None) -> int:
    """The week whose games span `anchor`, else the next one to start after it."""
    s = _schedule(season) if schedule is None else schedule
    if s.empty:
        raise RuntimeError(f"no {season} schedule available to derive the week from")
    spans = s.groupby("week")["gameday"].agg(["min", "max"]).sort_index()

    inside = spans[(spans["min"] <= anchor) & (spans["max"] >= anchor)]
    if not inside.empty:
        return int(inside.index[0])
    upcoming = spans[spans["min"] > anchor]
    if not upcoming.empty:
        return int(upcoming.index[0])
    # Past the last regular-season game. The last week is the honest answer;
    # there is no DK main slate to prepare for anyway.
    return int(spans.index[-1])


def current_week(season: int | None = None, kickoff_utc=None) -> int:
    """This slate's week. WEEK env wins if set; otherwise derived.

    `kickoff_utc` is the slate's first game start when the caller has the
    slate in hand -- better than "today", because the report runs daily and
    the slate is what the numbers have to describe. Converted to Eastern
    before taking the date: an 8:20 PM ET Sunday kickoff is already Monday in
    UTC, and that one-day slip is enough to land on the wrong week.
    """
    override = os.environ.get("WEEK", "").strip()
    if override:
        return int(override)

    season = season or int(os.environ.get("SEASON", "2026"))
    if kickoff_utc is not None and not pd.isna(kickoff_utc):
        ts = pd.to_datetime(kickoff_utc, utc=True)
        anchor = ts.tz_convert(ET).date()
    else:
        anchor = datetime.now(ET).date()
    return week_for_date(season, anchor)
