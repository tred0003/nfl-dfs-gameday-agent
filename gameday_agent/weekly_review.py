"""Weekly review: grade last week's REAL reports and compare them with the weeks before.

Everything else here checks one report at a time, or checks the agent against
made-up scenarios:

    report_check      every daily report, before it posts (deterministic)
    canary            Thursdays: do the tools work? (not: was the report good?)
    run_eval + judge  on demand: synthetic fixtures, scored against baseline.json

None of them ever goes back to what was actually sent and asks whether it was
any good, or whether this week was worse than last. This does, once a week,
after Monday night, on the reports already saved in Firestore
(users/{uid}/reports/{date}, which carry tool calls and results since 10/16):

  1. JUDGE. Each distinct agent report of the week goes to the same LLM judge
     the eval uses, with the tool results it was written from as ground truth.
     The judge grades `agent_text` -- what the model wrote -- not `text`, which
     may carry report_check's correction block on top. The block is code; the
     question is how good the model was.
  2. OUTCOMES. The Sunday report said who was OUT, Doubtful, Questionable or
     fine. nflverse snap counts say who actually played. Warnings that came
     true, and "fine" players who then sat, are the only ground truth this
     project gets for free -- the judge can only check a report against its
     own tool results, never against what happened.
  3. TREND. The week's numbers are saved to eval_weekly/{season}-w{NN} and
     compared with up to four weeks before. A regression exits 1, which trips
     the alert that already covers every Cloud Run job in the project.

    python gameday_agent/weekly_review.py                  # last completed week
    python gameday_agent/weekly_review.py --week 7
    python gameday_agent/weekly_review.py --dry-run        # print; save/post nothing
    python gameday_agent/weekly_review.py --no-judge       # free: checks, cost, outcomes
    python gameday_agent/weekly_review.py --no-discord

WHAT IS A REGRESSION. Thresholds are deliberately loose: one week is a handful
of reports, and the eval measured identical runs swinging ~±5 quality points.

  quality    judge mean more than QUALITY_TOLERANCE (10) below the prior weeks' mean
  fallback   more than a quarter of the week's reports were the rule-based fallback
  core tools any agent report skipped a core tool (prefetch requests all of them)
  checks     report_check failure rate up more than 20 points on the prior weeks
  cost       cost per agent report more than 1.5x the prior weeks' mean
  judge      every judge call failed -- no quality number at all is not a pass

Outcome misses are REPORTED, not gated, for now: a player ruled inactive at
11:30 after the 12:30 report went out is not the agent's miss, and with nine
players a week there is not enough data to tell the two apart. Once a few
weeks exist the precision numbers will say whether a gate is worth having.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from user_store import FirestoreStore  # noqa: E402

ET = ZoneInfo("America/New_York")
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"

# Must match run_report.QUERY and prefetch.CORE_TOOLS. Copied, not imported:
# both modules import the agent stack at the top, and this file's logic is
# tested in CI without it (tests/test_weekly_review.py checks they match).
QUERY = "Check my lineup for this Sunday and tell me if anything needs my attention."
CORE_TOOLS = ["get_lineup_status", "get_cb_matchups", "get_slate_weather",
              "get_vegas_context", "get_notable_scratches", "get_value_picks",
              "get_team_news", "get_player_history"]
# A core tool added mid-season is required only of reports made on or after
# the day it shipped; the old agent cannot have called it.
CORE_SINCE = {"get_player_history": "2026-10-22"}

RULED_OUT = {"O", "OUT", "IR", "D", "DOUBTFUL"}
QUESTIONABLE = {"Q", "QUESTIONABLE"}

QUALITY_TOLERANCE = 10.0      # same number as eval/scoring.py
FALLBACK_LIMIT = 0.25
CHECK_FAIL_RISE = 0.20
COST_RATIO = 1.5
HISTORY_WEEKS = 4
MAX_JUDGED = 30               # ~3.5c each with Pro; a cap, not a target
JUDGE_RETRIES = 3


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------ the week

def week_days(season: int, week: int, schedule: pd.DataFrame | None = None) -> tuple[date, date, date]:
    """(first day, last day, main Sunday) of a week's report window.

    The window runs from the day after the previous week's last game to this
    week's last game -- the same rule nfl_week uses, where Tuesday and
    Wednesday already belong to the coming week. The Sunday is the one the
    reports were preparing for.
    """
    from nfl_week import _schedule

    s = _schedule(season) if schedule is None else schedule
    spans = s.groupby("week")["gameday"].agg(["min", "max"]).sort_index()
    if week not in spans.index:
        raise ValueError(f"no week {week} in the {season} schedule")
    last = spans.loc[week, "max"]
    prev = spans[spans.index < week]
    first = (prev["max"].max() + pd.Timedelta(days=1)) if not prev.empty else spans.loc[week, "min"]
    days = s[s["week"].eq(week)]["gameday"]
    sundays = sorted(d for d in days if d.weekday() == 6)
    sunday = sundays[0] if sundays else spans.loc[week, "min"]
    return pd.Timestamp(first).date(), pd.Timestamp(last).date(), sunday


def last_completed_week(season: int, today: date | None = None,
                        schedule: pd.DataFrame | None = None) -> int:
    """The latest week whose last game is before today."""
    from nfl_week import _schedule

    s = _schedule(season) if schedule is None else schedule
    today = today or datetime.now(ET).date()
    spans = s.groupby("week")["gameday"].max()
    done = spans[spans < today]
    if done.empty:
        raise ValueError(f"no {season} week has finished before {today}")
    return int(done.index.max())


# --------------------------------------------------------------- the reports

def collect(store, first: date, last: date) -> list[dict]:
    """Every saved report in [first, last], one row per user per day."""
    rows = []
    for user in store.all_users():
        for rid in store.report_ids(user.uid):
            try:
                day = date.fromisoformat(rid)
            except ValueError:
                continue
            if first <= day <= last:
                doc = store.get_report(user.uid, rid) or {}
                rows.append({"uid": user.uid, "date": rid, "doc": doc})
    return sorted(rows, key=lambda r: (r["date"], r["uid"]))


def judge_inputs(doc: dict) -> tuple[list, list] | None:
    """(tool_calls, tool_results) in the shape judge() takes, or None.

    Reports saved before 10/16 have no tool calls stored, and without the
    tool results the judge has no ground truth -- it would be grading prose.
    Results capped at save time come back as their JSON prefix, a string;
    the judge truncates evidence long before that cap anyway.
    """
    stored = doc.get("tool_calls")
    if not stored:
        return None
    calls, results = [], []
    for c in stored:
        try:
            args = json.loads(c.get("args_json") or "{}")
        except ValueError:
            args = {}
        raw = c.get("result_json") or "null"
        try:
            result = json.loads(raw)
        except ValueError:
            result = raw
        calls.append((c.get("name", ""), args))
        results.append((c.get("name", ""), result))
    return calls, results


def lineup_result(doc: dict) -> dict | None:
    """get_lineup_status's result from a stored report, if it parses."""
    for c in doc.get("tool_calls") or []:
        if c.get("name") == "get_lineup_status":
            try:
                r = json.loads(c.get("result_json") or "")
            except ValueError:
                return None
            return r if isinstance(r, dict) else None
    return None


def _is_retryable(err: str | None) -> bool:
    return bool(err) and any(t in err for t in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))


def grade(rows: list[dict], judge_fn: Callable | None, max_judged: int = MAX_JUDGED,
          sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """One result row per report: the numbers this week is summarised from.

    Reused reports (same lineup as an earlier user that run) are counted for
    checks and fallback but judged once, through the original.
    """
    out, judged = [], 0
    for r in rows:
        doc = r["doc"]
        source = doc.get("source", "?")
        tools = doc.get("tools") or []
        row = {"uid": r["uid"], "date": r["date"], "source": source,
               "reused_from": doc.get("reused_from"),
               "check_failed": bool(doc.get("problems")),
               "problems": list(doc.get("problems") or []),
               "missing_core": ([t for t in CORE_TOOLS if t not in tools
                                 and r["date"][:10] >= CORE_SINCE.get(t, "")]
                                if source == "agent" else []),
               # total_cost_usd (agent + in-tool calls) since 10/19; before
               # that only the agent's own cost_usd was stored.
               "cost_usd": (None if doc.get("reused_from") else
                            doc.get("total_cost_usd", doc.get("cost_usd"))),
               "quality": None, "scores": {}, "reasons": {}, "judge_error": None}

        if judge_fn is None or source != "agent" or doc.get("reused_from"):
            out.append(row)
            continue
        inputs = judge_inputs(doc)
        if inputs is None:
            row["judge_error"] = "no tool calls stored (report predates 10/16)"
            out.append(row)
            continue
        if judged >= max_judged:
            row["judge_error"] = f"over the {max_judged}-report judge cap"
            out.append(row)
            continue

        calls, results = inputs
        text = doc.get("agent_text") or doc.get("text") or ""
        verdict = None
        for attempt in range(JUDGE_RETRIES):
            verdict = judge_fn(QUERY, calls, results, text)
            if not _is_retryable(verdict.error):
                break
            sleep(20 * (attempt + 1) ** 2)
        judged += 1
        if verdict.error:
            row["judge_error"] = verdict.error[:300]
        else:
            row["quality"] = round(verdict.percent, 1)
            row["scores"] = dict(verdict.scores)
            row["reasons"] = dict(verdict.reasons)
        out.append(row)
    return out


# --------------------------------------------------------------- outcomes

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str) -> str:
    n = str(name or "").lower().replace(".", "").replace("'", "").replace("-", " ")
    n = _SUFFIX.sub("", n)
    return re.sub(r"\s+", " ", n).strip()


def load_snaps(season: int) -> pd.DataFrame:
    """nflverse snap counts for the season. Same file cb_coverage.py reads."""
    import requests

    r = requests.get(f"{NFLVERSE}/snap_counts/snap_counts_{season}.parquet", timeout=120)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def sunday_reports(rows: list[dict], sunday: date) -> dict[str, dict]:
    """Each user's last report on or before the Sunday: what they set their lineup by."""
    best: dict[str, dict] = {}
    for r in rows:
        if date.fromisoformat(r["date"]) <= sunday:
            best[r["uid"]] = r  # rows are date-sorted, so the last one wins
    return best


def outcomes(rows: list[dict], sunday: date, snaps: pd.DataFrame, week: int,
             expected_teams: int) -> dict:
    """Did the warnings come true? One entry per distinct (player, verdict).

    played = more than zero offensive snaps that week. DST is skipped (a team
    always plays). Snap counts list only players who took a snap, so no row
    means "did not play" -- but only once every team that played that week
    has rows; before that, it may just mean the data is not in yet. A no-row
    player is marked `no_row`, because a name the two sources spell
    differently looks exactly like a player who sat.
    """
    wk = snaps[snaps["week"].eq(week)] if not snaps.empty else snaps
    if wk.empty:
        return {"status": "pending", "detail": f"no week {week} snap counts published yet"}
    teams = wk["team"].nunique() if "team" in wk else 0
    complete = teams >= expected_teams

    played: dict[str, bool] = {}
    for _, s in wk.iterrows():
        key = norm_name(s["player"])
        played[key] = played.get(key, False) or float(s.get("offense_snaps") or 0) > 0

    seen: dict[tuple[str, str], dict] = {}
    for uid, r in sunday_reports(rows, sunday).items():
        res = lineup_result(r["doc"])
        if not res or res.get("status") != "ok":
            continue
        for p in res.get("lineup") or []:
            if str(p.get("position", "")).upper() in ("DST", "DEF"):
                continue
            status = str(p.get("injury_status") or "").upper()
            verdict = ("out" if status in RULED_OUT
                       else "questionable" if status in QUESTIONABLE else "fine")
            key = norm_name(p.get("name"))
            no_row = key not in played
            did = played.get(key, False if complete else None)
            seen.setdefault((key, verdict), {
                "name": p.get("name"), "team": p.get("team"), "verdict": verdict,
                "status": status or None, "played": did, "no_row": no_row,
                "users": []})["users"].append(uid)

    entries = list(seen.values())
    known = [e for e in entries if e["played"] is not None]
    out_ = [e for e in known if e["verdict"] == "out"]
    q = [e for e in known if e["verdict"] == "questionable"]
    fine = [e for e in known if e["verdict"] == "fine"]
    return {
        "status": "ok" if complete else "partial",
        "detail": ("" if complete else
                   f"snap counts for {teams}/{expected_teams} teams so far"),
        "players": len(entries),
        "unknown": len(entries) - len(known),
        # Of the players the report said would not play, how many sat.
        "out_warned": len(out_),
        "out_correct": sum(1 for e in out_ if not e["played"]),
        "questionable": len(q),
        "questionable_sat": sum(1 for e in q if not e["played"]),
        # Players the report raised no flag on who then did not play: the miss.
        "fine": len(fine),
        "surprise_sits": [e["name"] + (" (no snap row)" if e["no_row"] else "")
                          for e in fine if not e["played"]],
        "wrong_outs": [e["name"] for e in out_ if e["played"]],
        "entries": entries,
    }


# ------------------------------------------------------------- the summary

def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def summarise(graded: list[dict]) -> dict:
    n = len(graded)
    agent = [g for g in graded if g["source"] == "agent"]
    judged = [g for g in graded if g["quality"] is not None]
    attempted = judged + [g for g in graded if g["judge_error"]
                          and "predates" not in g["judge_error"]
                          and "cap" not in g["judge_error"]]
    criteria: dict[str, list[int]] = {}
    for g in judged:
        for k, v in g["scores"].items():
            criteria.setdefault(k, []).append(v)
    return {
        "reports": n,
        "agent_reports": len(agent),
        "fallback_rate": round((n - len(agent)) / n, 4) if n else None,
        "check_fail_rate": round(sum(g["check_failed"] for g in agent) / len(agent), 4) if agent else None,
        "missing_core": sorted({t for g in agent for t in g["missing_core"]}),
        "missing_core_reports": sum(1 for g in agent if g["missing_core"]),
        "cost_per_report": _mean(g["cost_usd"] for g in agent),
        "judged": len(judged),
        "judge_attempted": len(attempted),
        "quality": _mean(g["quality"] for g in judged),
        "criteria": {k: round(statistics.mean(v), 2) for k, v in sorted(criteria.items())},
    }


def regressions(now: dict, history: list[dict]) -> list[str]:
    """Plain sentences; empty = no regression. History is prior weeks' summaries."""
    flags = []

    def prior(key):
        return _mean(h.get(key) for h in history)

    if now["reports"] == 0:
        return ["no reports were saved all week -- the daily job may not be running"]
    if now["fallback_rate"] is not None and now["fallback_rate"] > FALLBACK_LIMIT:
        flags.append(f"fallback on {now['fallback_rate']:.0%} of reports "
                     f"(limit {FALLBACK_LIMIT:.0%}) -- the agent was down a lot this week")
    if now["missing_core_reports"]:
        flags.append(f"{now['missing_core_reports']} agent report(s) skipped core tools: "
                     + ", ".join(now["missing_core"]))
    if now["judge_attempted"] and not now["judged"]:
        flags.append(f"all {now['judge_attempted']} judge calls failed -- no quality score this week")

    pq = prior("quality")
    if now["quality"] is not None and pq is not None and now["quality"] < pq - QUALITY_TOLERANCE:
        flags.append(f"judge quality {now['quality']:.1f}, down from {pq:.1f} "
                     f"(tolerance {QUALITY_TOLERANCE:g})")
    pc = prior("check_fail_rate")
    if now["check_fail_rate"] is not None and pc is not None and now["check_fail_rate"] > pc + CHECK_FAIL_RISE:
        flags.append(f"report_check failed on {now['check_fail_rate']:.0%} of agent reports, "
                     f"up from {pc:.0%}")
    pcost = prior("cost_per_report")
    if now["cost_per_report"] and pcost and now["cost_per_report"] > pcost * COST_RATIO:
        flags.append(f"cost per report ${now['cost_per_report']:.4f}, "
                     f"{now['cost_per_report'] / pcost:.1f}x the prior ${pcost:.4f}")
    return flags


def _fmt(x, spec=".1f", none="--"):
    return none if x is None else format(x, spec)


def render(season: int, week: int, now: dict, history: list[tuple[str, dict]],
           outcome: dict, graded: list[dict], flags: list[str]) -> str:
    """The Discord message. Short; the detail lives in Firestore."""
    head = "❌" if flags else "✅"
    lines = [f"{head} **Weekly agent review -- {season} Week {week}**"]
    prior = [h.get("summary", {}) for _, h in history]
    pq = _mean(p.get("quality") for p in prior)
    lines.append(
        f"Reports {now['reports']} ({now['agent_reports']} agent, "
        f"fallback {_fmt(now['fallback_rate'], '.0%')}) · "
        f"check fails {_fmt(now['check_fail_rate'], '.0%')} · "
        f"cost/report ${_fmt(now['cost_per_report'], '.4f')}")
    lines.append(
        f"Judge quality {_fmt(now['quality'])} on {now['judged']} report(s)"
        + (f" · prior {len(prior)} wk {_fmt(pq)}" if pq is not None else " · no prior weeks yet"))
    if now["criteria"]:
        lines.append("  " + " · ".join(f"{k} {v:.1f}" for k, v in now["criteria"].items()))
    if history:
        trend = [f"W{h_id.rsplit('w', 1)[-1]} {_fmt(h.get('summary', {}).get('quality'))}"
                 for h_id, h in history]
        lines.append("Trend: " + " → ".join(trend + [f"W{week:02d} {_fmt(now['quality'])}"]))

    if outcome.get("status") in ("ok", "partial"):
        lines.append(
            f"Outcomes: OUT/D warnings {outcome['out_correct']}/{outcome['out_warned']} sat · "
            f"Q {outcome['questionable_sat']}/{outcome['questionable']} sat · "
            f"unflagged sits {len(outcome['surprise_sits'])}/{outcome['fine']}"
            + (f" ({', '.join(outcome['surprise_sits'])})" if outcome["surprise_sits"] else "")
            + (f" · {outcome['unknown']} not known yet ({outcome['detail']})"
               if outcome["unknown"] else ""))
        if outcome["wrong_outs"]:
            lines.append(f"  Said out but played: {', '.join(outcome['wrong_outs'])}")
    else:
        lines.append(f"Outcomes: {outcome.get('detail', outcome.get('status'))}")

    low = sorted((g for g in graded if g["quality"] is not None), key=lambda g: g["quality"])[:2]
    for g in low:
        worst = min(g["scores"], key=g["scores"].get) if g["scores"] else None
        reason = g["reasons"].get(worst, "") if worst else ""
        lines.append(f"Lowest: {g['uid']} {g['date']} {g['quality']:.1f}"
                     + (f" -- {worst}: {reason[:160]}" if worst else ""))
    if flags:
        lines.append("**Regressions:**")
        lines += [f"• {f}" for f in flags]
    return "\n".join(lines)


# -------------------------------------------------------------------- main

def week_id(season: int, week: int) -> str:
    return f"{season}-w{week:02d}"


def review(store, season: int, week: int, judge_fn: Callable | None,
           snaps_fn: Callable[[int], pd.DataFrame], schedule: pd.DataFrame | None = None,
           max_judged: int = MAX_JUDGED) -> dict:
    """The whole review for one week, minus saving and posting. Returns everything."""
    first, last, sunday = week_days(season, week, schedule)
    log(f"week {week}: reports {first} .. {last}, Sunday {sunday}")
    rows = collect(store, first, last)
    log(f"{len(rows)} report(s) from {len({r['uid'] for r in rows})} user(s)")

    graded = grade(rows, judge_fn, max_judged=max_judged)
    now = summarise(graded)
    log(f"summary: {json.dumps(now, default=str)}")

    try:
        from nfl_week import _schedule
        sched = _schedule(season) if schedule is None else schedule
        expected = 2 * int(sched["week"].eq(week).sum())
        outcome = outcomes(rows, sunday, snaps_fn(season), week, expected)
    except Exception as e:  # noqa: BLE001 -- nflverse being down is not a regression
        outcome = {"status": "error", "detail": f"snap counts unavailable: {type(e).__name__}: {e}"}
    log(f"outcomes: {outcome.get('status')} {outcome.get('detail', '')}")

    wid = week_id(season, week)
    history = store.weekly_history(before=wid, n=HISTORY_WEEKS)
    flags = regressions(now, [h.get("summary", {}) for _, h in history])
    return {"week_id": wid, "season": season, "week": week, "first": str(first),
            "last": str(last), "sunday": str(sunday), "summary": now,
            "outcome": outcome, "flags": flags, "graded": graded, "history": history}


def main() -> int:
    p = argparse.ArgumentParser(description="Grade and compare last week's real reports")
    p.add_argument("--season", type=int, default=int(os.environ.get("SEASON", "2026")))
    p.add_argument("--week", type=int, help="default: the last completed week")
    p.add_argument("--dry-run", action="store_true", help="print; save and post nothing")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge (free)")
    p.add_argument("--no-discord", action="store_true", help="save, but post nothing")
    p.add_argument("--max-judged", type=int, default=MAX_JUDGED)
    a = p.parse_args()

    from watch_slate import load_env, send_discord
    load_env()

    week = a.week or last_completed_week(a.season)
    judge_fn = None
    if not a.no_judge:
        # Lazy: the judge pulls in the agent package (and ADK) through its
        # imports; --no-judge runs without either.
        from gameday_agent.eval.judge import JUDGE_MODEL, judge
        judge_fn = judge
        log(f"judge: {JUDGE_MODEL}")

    try:
        store = FirestoreStore()
        result = review(store, a.season, week, judge_fn, load_snaps, max_judged=a.max_judged)
    except Exception as e:  # noqa: BLE001 -- anything here means no review at all
        log(f"REVIEW FAILED: {type(e).__name__}: {e}")
        return 1

    msg = render(a.season, week, result["summary"], result["history"],
                 result["outcome"], result["graded"], result["flags"])
    print("\n" + msg + "\n", flush=True)

    if a.dry_run:
        log("dry run -- not saving or posting")
    else:
        doc = {k: result[k] for k in ("season", "week", "first", "last", "sunday",
                                      "summary", "outcome", "flags", "graded")}
        doc["judge_model"] = os.environ.get("JUDGE_MODEL", "gemini-2.5-pro") if judge_fn else None
        store.save_weekly(result["week_id"], doc)
        log(f"saved eval_weekly/{result['week_id']}")
        if not a.no_discord and not send_discord(msg):
            log("NOT DELIVERED: no DISCORD_WEBHOOK_URL")
            return 1

    for f in result["flags"]:
        log(f"WEEKLY REVIEW REGRESSION: {f}")
    return 1 if result["flags"] else 0


if __name__ == "__main__":
    sys.exit(main())
