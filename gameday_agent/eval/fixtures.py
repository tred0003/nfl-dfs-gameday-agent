"""Synthetic DK + ESPN data for the eval harness. No real pulled data here.

This project's own .gitignore excludes DK pulls for a stated reason: DK's ToS
forbids redistributing their data, and espn_client.py carries the same
"personal research use only" note for ESPN. A frozen snapshot of either would
be exactly the kind of file that rule exists to keep out of git -- so eval
fixtures are hand-built here instead of frozen from a live pull.

This is also just the more correct choice for a test fixture: every row
exists because a scenario in scenarios.py needs it (a sub-$3000 TE, a null
FPPG, a specific OUT player), not because it happened to be true on some
Tuesday. Real player names are used only because they read naturally in a
demo -- none of the salaries, stats, or statuses attached to them are real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from predict_slate import normalize_name  # noqa: E402

# The lineup used by every scenario in scenarios.py. Corey Whitlock's null
# dk_fppg and Grant Iverson's $2900 salary are deliberate -- they are what the
# clean_lineup and emergency_out scenarios exist to guard.
LINEUP_PLAYERS = [
    {"name": "Deshawn Carver", "position": "QB", "team": "PHI", "game": "WAS @ PHI", "salary": 6600, "dk_fppg": 19.1, "dk_oprk": 29},
    {"name": "Marlon Petty", "position": "RB", "team": "DET", "game": "NO @ DET", "salary": 8000, "dk_fppg": 22.3, "dk_oprk": 13},
    {"name": "Julian Vance", "position": "RB", "team": "MIA", "game": "MIA @ LV", "salary": 7000, "dk_fppg": 20.9, "dk_oprk": 21},
    {"name": "Trevon Ashby", "position": "WR", "team": "PIT", "game": "ATL @ PIT", "salary": 5400, "dk_fppg": 12.5, "dk_oprk": 21},
    {"name": "Rashad Feeney", "position": "WR", "team": "TEN", "game": "NYJ @ TEN", "salary": 4500, "dk_fppg": 7.2, "dk_oprk": 15},
    {"name": "Corey Whitlock", "position": "WR", "team": "PHI", "game": "WAS @ PHI", "salary": 5100, "dk_fppg": None, "dk_oprk": 24},
    {"name": "Grant Iverson", "position": "TE", "team": "LV", "game": "MIA @ LV", "salary": 2900, "dk_fppg": 5.7, "dk_oprk": 30},
    {"name": "Deion Marsh", "position": "WR", "team": "MIN", "game": "GB @ MIN", "salary": 7300, "dk_fppg": 12.4, "dk_oprk": 18},
    {"name": "Coastal", "position": "DST", "team": "BAL", "game": "BAL @ IND", "salary": 3000, "dk_fppg": 5.7, "dk_oprk": 7},
]

# Extra players so find_replacements() and get_notable_scratches() have a
# real pool to search, spanning the same positions/salary bands.
POOL_PLAYERS = [
    {"name": "Marquis Boyd", "position": "WR", "team": "TEN", "game": "NYJ @ TEN", "salary": 4800, "dk_fppg": 14.2, "dk_oprk": 19},
    {"name": "Antoine Reyes", "position": "WR", "team": "PIT", "game": "ATL @ PIT", "salary": 4900, "dk_fppg": 12.1, "dk_oprk": 21},
    {"name": "Devon Larkspur", "position": "WR", "team": "LAC", "game": "ARI @ LAC", "salary": 4700, "dk_fppg": 12.0, "dk_oprk": 11},
    {"name": "Kellan Fitch", "position": "WR", "team": "BUF", "game": "BUF @ HOU", "salary": 4800, "dk_fppg": 11.4, "dk_oprk": 8},
    {"name": "Preston Oyelaran", "position": "TE", "team": "LV", "game": "MIA @ LV", "salary": 2500, "dk_fppg": 8.6, "dk_oprk": 27},
    {"name": "Nate Ferriday", "position": "TE", "team": "BUF", "game": "BUF @ HOU", "salary": 3100, "dk_fppg": 6.0, "dk_oprk": 22},
    {"name": "Deandre Stovall", "position": "RB", "team": "NO", "game": "NO @ DET", "salary": 5900, "dk_fppg": 15.7, "dk_oprk": 13},
    {"name": "Marcus Whitfeld", "position": "RB", "team": "TB", "game": "TB @ CIN", "salary": 5700, "dk_fppg": 14.3, "dk_oprk": 16},
    # High salary, ruled OUT: exercises get_notable_scratches without being
    # on the user's own roster.
    {"name": "Damarion Vess", "position": "RB", "team": "GB", "game": "GB @ MIN", "salary": 6300, "dk_fppg": 16.1, "dk_oprk": 9},
]


# The games these players are in. This IS load-bearing: the weather and Vegas
# tools scope themselves to the teams on the slate, so an empty games frame
# would scope them to nothing. Every game here is a 1pm or 4pm Sunday kickoff,
# because that is the only window the user's contest covers.
SLATE_GAMES = [
    {"game": "WAS @ PHI", "home_team": "PHI", "away_team": "WAS"},
    {"game": "NO @ DET", "home_team": "DET", "away_team": "NO"},
    {"game": "MIA @ LV", "home_team": "LV", "away_team": "MIA"},
    {"game": "ATL @ PIT", "home_team": "PIT", "away_team": "ATL"},
    {"game": "NYJ @ TEN", "home_team": "TEN", "away_team": "NYJ"},
    {"game": "GB @ MIN", "home_team": "MIN", "away_team": "GB"},
    {"game": "BAL @ IND", "home_team": "IND", "away_team": "BAL"},
    {"game": "ARI @ LAC", "home_team": "LAC", "away_team": "ARI"},
    {"game": "BUF @ HOU", "home_team": "HOU", "away_team": "BUF"},
    {"game": "TB @ CIN", "home_team": "CIN", "away_team": "TB"},
]


def dk_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(players, games) in the shape slate() returns."""
    players = pd.DataFrame(LINEUP_PLAYERS + POOL_PLAYERS)
    players["jn"] = normalize_name(players["name"])
    games = pd.DataFrame(SLATE_GAMES)
    return players, games


def espn_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(injuries, scoreboard) in the shape espn_injuries()/espn_scoreboard() return.

    Only the pool players carry a report here -- scenarios.py clears and
    overrides the lineup players itself, per scenario, so this baseline
    never leaks an assumption about lineup health into a test.
    """
    injuries = pd.DataFrame([
        {"team": "Las Vegas Raiders", "name": "Damarion Vess", "position": "RB",
         "status": "OUT", "body_part": "Undisclosed", "return_date": None,
         "comment": "Vess will not play Sunday.", "report_date": "2026-09-17"},
        {"team": "Buffalo Bills", "name": "Kellan Fitch", "position": "WR",
         "status": "Q", "body_part": "Ankle", "return_date": "2026-09-20",
         "comment": "Fitch (ankle) is questionable for Sunday.", "report_date": "2026-09-17"},
    ])
    injuries["jn"] = normalize_name(injuries["name"])

    # ESPN returns every game in the week, including the primetime ones that
    # are NOT on the user's slate. The third row exists so the scoping filter
    # in get_slate_weather is actually exercised: it is a Sunday-night game in
    # a blizzard, and it must never reach the report, because no player in it
    # can be rostered. ESPN also spells Washington `WSH` where DK says `WAS`,
    # so the first row keeps that normalization honest too.
    scoreboard = pd.DataFrame([
        {"game": "Washington Commanders at Philadelphia Eagles", "home_team": "PHI", "away_team": "WSH",
         "start_time_utc": "2026-09-20T20:25Z", "venue": "Lincoln Financial Field", "is_dome": False,
         "weather_posted": True, "weather_temp_f": 68, "weather_condition": "Rain"},
        {"game": "New Orleans Saints at Detroit Lions", "home_team": "DET", "away_team": "NO",
         "start_time_utc": "2026-09-20T17:00Z", "venue": "Ford Field", "is_dome": True,
         "weather_posted": False, "weather_temp_f": None, "weather_condition": None},
        {"game": "New York Giants at Los Angeles Rams", "home_team": "LAR", "away_team": "NYG",
         "start_time_utc": "2026-09-21T00:20Z", "venue": "SoFi Stadium", "is_dome": False,
         "weather_posted": True, "weather_temp_f": 21, "weather_condition": "Heavy Snow"},
    ])
    return injuries, scoreboard
