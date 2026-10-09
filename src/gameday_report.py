"""Sunday-morning gameday report -> Discord.

Runs headless on a schedule (Cloud Run job + Cloud Scheduler). Pulls the live
public DraftKings slate and reports three things about YOUR lineup: availability,
weather, and breaking news. Then it gets out of the way.

Deliberately STATELESS. A scheduled report does not need to remember previous
runs, and not needing state is what lets this run as a plain container with no
database, no volume and no bucket. The diffing watcher (watch_slate.py) is the
stateful one; this is the thing you actually want at noon on a Sunday.

Configuration is entirely environment variables, so the deployed job can be
retargeted at a new week's lineup without rebuilding the image:

  DISCORD_WEBHOOK_URL   required to actually send; prints to stdout otherwise
  MY_LINEUP             comma-separated player names, when no lineup is passed in
  DRAFT_GROUP           pin a slate id; otherwise picks the largest Sunday prize
  WEEK                  override the week; otherwise derived from the slate date

When gameday_agent/run_report.py calls this as its fallback it passes the
lineup it read from the Google Sheet, so the fallback checks the same roster
the agent would have.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dk_client import fetch_draftables, list_slates, pick_main_slate  # noqa: E402
from nfl_week import current_week  # noqa: E402
from predict_slate import normalize_name  # noqa: E402
from watch_slate import load_env, send_discord  # noqa: E402

# Availability tags, worst first. "D" (Doubtful) is nearer to Out than to
# Questionable -- those players usually do not suit up.
SEVERITY = {"OUT": 0, "IR": 0, "D": 1, "Q": 2}
FLAG = {"OUT": "🔴", "IR": "🔴", "D": "🟠", "Q": "🟡"}

# DK weather icons that actually change how a game is played. Dome games are
# excluded before this is consulted -- the feed still reports the outdoor
# forecast for a covered stadium, which is how you end up warning someone about
# rain in Minneapolis.
BAD_WEATHER = {"rain", "snow", "sleet", "wind", "thunderstorm", "fog"}


def lineup_names() -> list[str]:
    raw = os.environ.get("MY_LINEUP", "")
    return [n.strip() for n in raw.split(",") if n.strip()]


def build_report(week: int | None = None, dg: int | None = None,
                 names: list[str] | None = None) -> str:
    dg = dg or int(os.environ.get("DRAFT_GROUP") or 0) or pick_main_slate(list_slates())
    players, games = fetch_draftables(dg)

    if week is None:
        # The week is only the header here, and this is the fallback -- the
        # thing that runs when something else already broke. A schedule
        # lookup that fails must not take the whole report down with it.
        try:
            week = current_week(kickoff_utc=games["start_time_utc"].min())
        except Exception:  # noqa: BLE001
            week = None

    names = lineup_names() if names is None else names
    label = f"Week {week} " if week else ""
    lines = [f"**🏈 NFL DFS — {label}gameday report**"]

    # ---------------------------------------------------------- your lineup
    if names:
        want = {normalize_name(pd.Series([n]))[0]: n for n in names}
        players["jn"] = normalize_name(players["name"])
        mine = players[players["jn"].isin(want)].copy()

        missing = set(want) - set(mine["jn"])
        flagged = mine[mine["status"].isin(SEVERITY)].copy()

        if flagged.empty:
            lines.append(f"\n✅ **All {len(mine)} of your players are active.**")
        else:
            flagged["sev"] = flagged["status"].map(SEVERITY)
            lines.append("\n**⚠️ AVAILABILITY**")
            for _, r in flagged.sort_values("sev").iterrows():
                lines.append(
                    f"{FLAG.get(r['status'], '•')} **{r['name']}** "
                    f"({r['position']}, {r['team']}) — **{r['status']}**"
                )
            if (flagged["sev"] <= 1).any():
                lines.append("_You need a swap._")

        news = mine[mine["news_status"].eq("Breaking")]
        if not news.empty:
            lines.append("\n**📰 BREAKING NEWS ON YOUR PLAYERS**")
            for _, r in news.iterrows():
                lines.append(f"• {r['name']} ({r['position']}, {r['team']})")

        if missing:
            lines.append(f"\n_Not found on this slate: {', '.join(want[m] for m in missing)}_")

        my_teams = set(mine["team"].dropna())
    else:
        lines.append("\n_No lineup provided._")
        my_teams = set()

    # -------------------------------------------------------------- weather
    outdoor = games[~games["is_dome"].fillna(False)]
    rough = outdoor[outdoor["weather_icon"].fillna("").str.lower().apply(
        lambda i: any(b in i for b in BAD_WEATHER))]

    if rough.empty:
        lines.append("\n🌤️ **Weather:** nothing notable in any outdoor game.")
    else:
        lines.append("\n**🌧️ WEATHER**")
        for _, g in rough.iterrows():
            involved = {g["home_team"], g["away_team"]} & my_teams
            mark = "  ← **your player**" if involved else ""
            lines.append(f"• {g['game']} — {g['weather_icon']} ({g['venue']}){mark}")

    # ------------------------------------------------- slate-wide scratches
    hurt = players[players["status"].isin(["OUT", "IR", "D"]) & players["salary"].ge(6000)]
    if not hurt.empty:
        lines.append("\n**🚑 NOTABLE SCRATCHES (slate-wide, $6k+)**")
        for _, r in hurt.nlargest(6, "salary").iterrows():
            lines.append(
                f"• {r['name']} ({r['position']}, {r['team']}) "
                f"${r['salary']:,} — {r['status']}"
            )

    lines.append(f"\n_slate dg{dg} · {len(games)} games · "
                 f"{pd.Timestamp.now(tz='US/Eastern'):%a %b %d %I:%M %p ET}_")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--week", type=int, default=None,
                   help="override; otherwise derived from the slate date")
    p.add_argument("--dg", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    load_env()
    report = build_report(a.week, a.dg)

    if a.dry_run:
        print(report)
    else:
        send_discord(report)


if __name__ == "__main__":
    main()
