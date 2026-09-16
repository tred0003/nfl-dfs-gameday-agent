"""Gameday agent -- Gemini with tools, deciding what to investigate.

This is the layer that is genuinely an agent, as distinct from gameday_report.py,
which is a scheduled script with fixed rules. The difference is real and worth
being able to defend:

  gameday_report.py   Fixed sequence. Fetch slate -> apply `if status in
                      BAD_STATUSES` -> format template. Deterministic, free,
                      cannot hallucinate. Correct choice for the things rules
                      handle well.

  agent.py            The model chooses which tools to call and when. It may
                      look at the lineup, notice breaking news on a running
                      back, go find replacements at that salary, check the
                      Vegas total for the replacement's game, and only then
                      decide whether any of it is worth an interruption. That
                      sequence is not hardcoded anywhere.

Use the agent where JUDGMENT over messy inputs is required -- how bad is this
injury really, do these two sources actually conflict, is this worth waking
someone up for. Use rules where a lookup suffices. Being able to say which is
which is the point.

  python src/agent.py --week 1 --dry-run
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
from predict_slate import normalize_name, vegas_for_slate  # noqa: E402
from watch_slate import load_env, send_discord  # noqa: E402

PROJECT = os.environ.get("GCP_PROJECT", "advisor-agent-2026")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# module-level cache so repeated tool calls in one run hit the API once
_SLATE: dict = {}


def _slate() -> tuple[pd.DataFrame, pd.DataFrame]:
    if "players" not in _SLATE:
        dg = int(os.environ.get("DRAFT_GROUP") or 0) or pick_main_slate(list_slates())
        players, games = fetch_draftables(dg)
        players["jn"] = normalize_name(players["name"])
        _SLATE.update(players=players, games=games, dg=dg)
    return _SLATE["players"], _SLATE["games"]


def _my_names() -> list[str]:
    return [n.strip() for n in os.environ.get("MY_LINEUP", "").split(",") if n.strip()]


# --------------------------------------------------------------------- tools
# Docstrings and type hints below become the function declarations the model
# sees, so they are prompt surface, not just documentation.

def get_lineup_status() -> list[dict]:
    """Get the user's current DFS lineup with each player's live availability.

    Returns one record per rostered player with position, team, opponent, salary,
    injury status (None/Q/D/OUT/IR) and whether DraftKings is flagging breaking
    news on them.
    """
    players, _ = _slate()
    want = {normalize_name(pd.Series([n]))[0] for n in _my_names()}
    mine = players[players["jn"].isin(want)]
    out = []
    for _, r in mine.iterrows():
        out.append({
            "name": r["name"], "position": r["position"], "team": r["team"],
            "game": r["game"], "salary": int(r["salary"]),
            "injury_status": r["status"] or "None",
            "news_flag": r["news_status"] or "None",
            "dk_points_per_game": None if pd.isna(r["dk_fppg"]) else float(r["dk_fppg"]),
            "opponent_rank_vs_position": None if pd.isna(r["dk_oprk"]) else int(r["dk_oprk"]),
        })
    return out


def get_slate_weather() -> list[dict]:
    """Get weather for every game on the slate.

    `is_dome` matters: DraftKings reports the outdoor forecast even for covered
    stadiums, so a dome game showing rain is not actually affected.
    """
    _, games = _slate()
    return [{
        "game": g["game"], "venue": g["venue"],
        "is_dome": bool(g["is_dome"]), "conditions": g["weather_icon"],
    } for _, g in games.iterrows()]


def get_vegas_context() -> list[dict]:
    """Get each team's Vegas implied point total for this week.

    Implied team total is how many points the betting market expects a team to
    score. It was the strongest non-form predictor in backtesting -- a player in
    a 28-point offense is in a materially better spot than one in a 19-point
    offense, regardless of talent.
    """
    v = vegas_for_slate(2026, int(os.environ.get("WEEK", 1)))
    v = v.dropna(subset=["implied_team_total"]).drop_duplicates("team")
    return [{"team": r["team"],
             "implied_team_total": round(float(r["implied_team_total"]), 2),
             "opponent_implied_total": round(float(r["implied_opp_total"]), 2)}
            for _, r in v.iterrows()]


def find_replacements(position: str, max_salary: int, min_salary: int = 3000) -> list[dict]:
    """Find available players at a position within a salary range.

    Use this when a rostered player is OUT or Doubtful and needs replacing.
    Excludes anyone who is themselves OUT or on IR. Returns the top options by
    DraftKings points per game.
    """
    players, _ = _slate()
    p = players[
        players["position"].eq(position.upper())
        & players["salary"].between(min_salary, max_salary)
        & ~players["status"].isin(["OUT", "IR"])
    ].copy()
    p = p.sort_values("dk_fppg", ascending=False).head(8)
    return [{
        "name": r["name"], "team": r["team"], "game": r["game"],
        "salary": int(r["salary"]),
        "dk_points_per_game": None if pd.isna(r["dk_fppg"]) else float(r["dk_fppg"]),
        "injury_status": r["status"] or "None",
    } for _, r in p.iterrows()]


def get_notable_scratches(min_salary: int = 6000) -> list[dict]:
    """Get expensive players across the whole slate who are OUT, IR or Doubtful.

    These reshape the slate even if the user does not roster them -- their
    backups become viable and their teammates absorb the vacated workload.
    """
    players, _ = _slate()
    p = players[players["status"].isin(["OUT", "IR", "D"])
                & players["salary"].ge(min_salary)]
    return [{"name": r["name"], "position": r["position"], "team": r["team"],
             "salary": int(r["salary"]), "injury_status": r["status"]}
            for _, r in p.nlargest(10, "salary").iterrows()]


TOOLS = [get_lineup_status, get_slate_weather, get_vegas_context,
         find_replacements, get_notable_scratches]

SYSTEM_PROMPT = """\
You are a daily fantasy sports analyst monitoring one user's DraftKings NFL
lineup on Sunday morning, about an hour before kickoff.

Your report goes to Discord and interrupts someone spending time with family.
That is the constraint that should shape everything you write. Earn the
interruption or keep it short.

Investigate in whatever order makes sense. Start with the lineup. If a player is
OUT or Doubtful, find replacements before you report the problem -- a warning
without an option is not useful. If something looks fine, verify it rather than
assuming.

What matters, roughly in order:
1. A rostered player who cannot play. This is the only true emergency.
2. A rostered player who is Questionable or has breaking news -- real, but not
   necessarily actionable.
3. Weather that actually affects a game the user has exposure to. Dome games are
   not affected regardless of the forecast; check is_dome before mentioning
   weather at all.
4. Slate-wide scratches that change the landscape.

Known analytical findings from backtesting this user's own data, which you may
use but should not recite:
- Implied team total is the strongest non-form signal available.
- Wind shifts play-calling toward the run but does NOT raise RB fantasy points.
- Buying low on a TE after a bad game works; on an RB it is a trap.

Format for Discord using markdown. Be direct and concrete -- name players,
numbers and games. No preamble, no restating the question, no hedging language.
If everything is genuinely fine, say so in two lines and stop. A short report is
a good outcome, not a failure to find something.
"""


def run_agent(week: int, model: str = MODEL) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    resp = client.models.generate_content(
        model=model,
        contents=(
            f"It is Sunday morning of NFL Week {week}. Check my lineup and tell me "
            f"anything I need to know before kickoff."
        ),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=TOOLS,               # SDK handles the call/response loop
            temperature=0.2,           # a status report should not be creative
        ),
    )
    return (resp.text or "").strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--week", type=int, default=int(os.environ.get("WEEK", 1)))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-fallback", action="store_true",
                   help="fail loudly instead of falling back to the rule-based report")
    a = p.parse_args()

    load_env()
    try:
        report = run_agent(a.week, a.model)
        if not report:
            raise RuntimeError("agent returned an empty report")
    except Exception as e:
        print(f"agent failed: {type(e).__name__}: {e}")
        if a.no_fallback:
            raise
        # A scheduled Sunday job that silently sends nothing is worse than one
        # that sends the deterministic version.
        print("falling back to the rule-based report")
        from gameday_report import build_report
        report = build_report(a.week)

    if a.dry_run:
        print(report)
    else:
        send_discord(report)


if __name__ == "__main__":
    main()
