"""ESPN's unofficial site API -- injury reports and per-game venue/weather.

ESPN publishes no official public API, but this is the same JSON endpoint
their own site and app consume, and -- as of September 2026 -- it is far more
stable than DraftKings': the DK draftables endpoint (dk_client.py) started
hard-blocking this project's requests with a 403 (Akamai) mid-September,
while this one has not. DK is still the only source for salaries, so it is
not being replaced, just no longer trusted for injury status or news.

Everything that knows about ESPN's undocumented JSON shape lives in THIS
file, same isolation rule as dk_client.py: when ESPN changes their payload,
exactly one file breaks.

Personal research use only -- do not redistribute the pulled data.

No custom headers, on purpose. site.api.espn.com's WAF 403s a request that
sends ANY User-Agent that looks hand-set -- a browser string, a custom
identifying one, even an empty one -- but passes requests' own default
(`python-requests/x.y.z`). Confirmed by testing each of those explicitly:
only the unmodified default got through, consistently, across repeated
tries. It looked exactly like random flakiness until isolated. Don't add a
User-Agent override here without re-testing against the live endpoint.

Run: python src/espn_client.py --injuries
     python src/espn_client.py --scoreboard
"""

from __future__ import annotations

import argparse

import pandas as pd
import requests

INJURIES_URL = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

TIMEOUT = 30

# ESPN's designation strings -> the short codes the rest of this project uses
# (see dk_client.py's `status` column: None/Q/D/OUT/IR). "Active" is ESPN's
# baseline for anyone who has ever had an injury-report entry but is
# currently unrestricted -- it means the same as no designation at all.
STATUS_MAP = {
    "Active": None,
    "Questionable": "Q",
    "Doubtful": "D",
    "Out": "OUT",
    "Injured Reserve": "IR",
}


def _get_json(url: str, **params) -> dict:
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_injuries() -> pd.DataFrame:
    """One row per player who has ever had an injury-report entry, league-wide.

    Rows where `status` is None (ESPN's "Active") mean the designation was
    lifted -- keep them rather than dropping, since `get_lineup_status` needs
    to tell "no report" apart from "was hurt, now cleared".

    `body_part` and `return_date` come from ESPN's own structured detail, not
    parsed out of free text -- when present, `return_date` is a real signal
    DK's endpoint never provided at all.
    """
    data = _get_json(INJURIES_URL)
    rows = []
    for team in data.get("injuries", []):
        team_name = team.get("displayName")
        for inj in team.get("injuries", []):
            athlete = inj.get("athlete") or {}
            details = inj.get("details") or {}
            rows.append({
                "team": team_name,
                "name": athlete.get("displayName"),
                "position": (athlete.get("position") or {}).get("abbreviation"),
                "status": STATUS_MAP.get(inj.get("status"), inj.get("status")),
                "body_part": details.get("type"),
                "return_date": details.get("returnDate"),
                "comment": inj.get("shortComment") or inj.get("longComment"),
                "report_date": inj.get("date"),
            })
    return pd.DataFrame(rows)


def fetch_scoreboard() -> pd.DataFrame:
    """One row per NFL game this week: venue, dome flag, and weather if posted.

    `weather` hangs off the EVENT, not the competition. Reading it off `comp`
    returns None for every game in every week, which made `weather_posted`
    permanently False -- and the earlier docstring here explained that away as
    "ESPN doesn't post until close to kickoff." It was never an ESPN
    limitation; the forecast was in the payload the whole time, one level up.
    Checked Friday 2026-09-18: 15 of 16 events carried a Sunday forecast.

    `weather_posted` still distinguishes "not available yet" from "checked, no
    weather issue" -- treat a false as unknown, not as clear skies. It is
    legitimately false for a game already in progress.

    No wind: ESPN's scoreboard weather is displayValue, temperature,
    highTemperature and conditionId only. `gust` is absent on every event.
    Any wind-based reasoning needs a different source.
    """
    data = _get_json(SCOREBOARD_URL)
    rows = []
    for e in data.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        venue = comp.get("venue") or {}
        weather = e.get("weather") or {}
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        rows.append({
            "game": e.get("name"),
            "home_team": (home.get("team") or {}).get("abbreviation"),
            "away_team": (away.get("team") or {}).get("abbreviation"),
            "start_time_utc": e.get("date"),
            "venue": venue.get("fullName"),
            "is_dome": venue.get("indoor"),
            "weather_posted": bool(weather),
            "weather_temp_f": weather.get("temperature"),
            "weather_condition": weather.get("displayValue"),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch ESPN NFL injuries/scoreboard")
    p.add_argument("--injuries", action="store_true")
    p.add_argument("--scoreboard", action="store_true")
    a = p.parse_args()

    if a.injuries or not a.scoreboard:
        print(fetch_injuries().to_string(index=False))
    if a.scoreboard:
        print(fetch_scoreboard().to_string(index=False))
