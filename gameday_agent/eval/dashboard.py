"""Eval dashboard: every report, opened up -- plus blind grading and the numbers.

    python gameday_agent/eval/dashboard.py          # then open http://localhost:8765

Five views, one local server:

  Overview   the headline numbers from the latest eval run, the judge
             validation table, and the run history.
  Reports    every report there is: live per-user reports from Firestore
             and every scenario of every eval run. Click one to open it.
  Report     one report, end to end: the lineup that was submitted and how
             the slate matched it, every model call with its tokens and cost,
             every tool call with its arguments and the full JSON it
             returned, the five report_check rules and what each looked at,
             the scenario's own checks (eval), and the report as delivered.
  Tower      the Control Tower: production SLA, business-rule (fact-check)
             pass rates, policy hits, cost, data freshness, every scheduled
             job's recent runs, and the alert policies with what fired.
             Reads Firestore, Cloud Run, Cloud Logging, Cloud Monitoring and
             BigQuery; Cloud Monitoring is what pages (deploy/MONITORING.md).
  Grade      blind grading. The judge's scores are not sent to the browser
             until yours are saved (human_scores.json, the file align.py reads).
             For the same reason the Report view hides the judge's scores on
             any eval report you have not graded yet.

Local only, on purpose: it reads every user's reports (an admin view), so it
is not something to put on the public web app.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gameday_agent"))
sys.path.insert(0, str(ROOT))

from flask import Flask, abort, jsonify, request, send_file  # noqa: E402

from gameday_agent.eval.align import (  # noqa: E402
    MIN_USEFUL_SAMPLE, _all_judged_results, _load_human, _save_human, alignment)
from gameday_agent.eval.judge import JUDGE_MODEL  # noqa: E402
from gameday_agent.eval.rubric import CRITERIA, to_percent  # noqa: E402
from gameday_agent.eval.scoring import QUALITY_TOLERANCE, RUNS_DIR, load_baseline  # noqa: E402
from report_check import audit  # noqa: E402
from token_cost import PRICES_CHECKED, PRICES_PER_M  # noqa: E402

HERE = Path(__file__).resolve().parent
PORT = 8765
PROJECT = "advisor-agent-2026"
AGENT_MODEL = "gemini-2.5-flash"

# Every tool in gameday_agent/tools.py TOOLS. The first eight are expected on
# every full report; the last three are conditional (find_replacements only
# when a player is OUT, the two searches only when the agent goes looking).
TOOLS = ["get_lineup_status", "get_cb_matchups", "get_slate_weather",
         "get_vegas_context", "get_team_news", "get_player_history",
         "get_notable_scratches", "get_value_picks", "find_replacements",
         "search_news", "search_news_archive"]
CORE_TOOLS = TOOLS[:8]

TOOL_ABOUT = {
    "get_lineup_status": "Your players: salary, injury status, and any name the slate didn't recognise",
    "get_cb_matchups": "The cornerback each of your receivers and tight ends is likely to face",
    "get_slate_weather": "Weather and roof status for each game you have a player in",
    "get_vegas_context": "Betting lines: how many points each team is expected to score",
    "get_team_news": "Recent team-site news, screened for injection, with quoted claims",
    "get_notable_scratches": "Big-salary players ruled out anywhere on the slate",
    "get_value_picks": "Cheap players the projection model likes this week",
    "find_replacements": "Swap-in options at a position, under a salary cap",
    "search_news": "A web search the agent can run when it wants more",
    "get_player_history": "The last four weeks of archived news on each of your players",
    "search_news_archive": "A question the agent asks the news archive (BigQuery vector search)",
}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def _tok_total(t: dict | None) -> int | None:
    if not t:
        return None
    return t.get("input", 0) + t.get("output", 0) + t.get("thinking", 0)


def _graded_ids() -> set[str]:
    return set(_load_human())


# ------------------------------------------------------------------ eval runs

def _runs() -> list[tuple[str, dict]]:
    out = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            out.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            continue
    return out


def _load_run(stem: str) -> dict:
    path = RUNS_DIR / f"{stem}.json"
    if not path.exists() or path.parent != RUNS_DIR:
        abort(404, "no such run")
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_responses(trace: dict) -> list[dict]:
    return [{"name": n, "response": r} for n, r in trace.get("tool_results", [])]


def _eval_tool_calls(trace: dict) -> list[dict]:
    """Each call with its own result, paired by name and order."""
    by_name: dict[str, list] = {}
    for n, r in trace.get("tool_results", []):
        by_name.setdefault(n, []).append(r)
    out = []
    for n, args in trace.get("tool_calls", []):
        queue = by_name.get(n) or []
        out.append({"name": n, "args": args, "result": queue.pop(0) if queue else None})
    return out


def _eval_row(stem: str, payload: dict, r: dict) -> dict:
    trace = (payload.get("traces") or {}).get(r["scenario"]) or {}
    called = {n for n, _ in trace.get("tool_calls", [])}
    rows = audit(r.get("report", ""), _eval_responses(trace)) if trace else []
    return {
        "kind": "eval", "id": f"{stem}/{r['scenario']}", "date": payload.get("at", "")[:16],
        "who": r["scenario"], "label": r.get("description", ""), "source": "eval",
        "checks": f"{r['checks_passed']}/{r['checks_total']}",
        "checks_ok": r["checks_passed"] == r["checks_total"],
        "rc_passed": sum(x["passed"] for x in rows), "rc_total": len(rows),
        "quality": r.get("quality"), "tool_count": r.get("tool_count"),
        "tools_called": sorted(called) if trace else None,
        "core_used": sum(t in called for t in CORE_TOOLS),
        "tokens_total": _tok_total(r.get("tokens")), "cost_usd": r.get("cost_usd"),
        "total_cost_usd": r.get("total_cost_usd"),
        "has_trace": bool(trace),
    }


# --------------------------------------------------------------- production

def _db():
    from google.cloud import firestore
    return firestore.Client(project=PROJECT)


def _prod_rows() -> tuple[list[dict], str | None]:
    """Every stored report in users/*/reports/*. The admin view."""
    try:
        rows = []
        for snap in _db().collection_group("reports").stream():
            d = snap.to_dict() or {}
            uid = snap.reference.parent.parent.id
            checks = d.get("checks") or []
            tools = d.get("tools") or []
            rows.append({
                "kind": "prod", "id": f"{uid}/{snap.id}", "date": snap.id, "who": uid,
                "label": f"{len(d.get('lineup') or [])} players" if d.get("lineup") else "",
                "source": d.get("source", ""),
                "checks": "", "checks_ok": not (d.get("problems") or []),
                "rc_passed": sum(c.get("passed", False) for c in checks) if checks else None,
                "rc_total": len(checks) if checks else None,
                "problems": len(d.get("problems") or []),
                "quality": None, "tool_count": len(tools), "tools_called": sorted(set(tools)),
                "core_used": sum(t in set(tools) for t in CORE_TOOLS),
                "tokens_total": _tok_total(d.get("tokens")), "cost_usd": d.get("cost_usd"),
                "total_cost_usd": d.get("total_cost_usd"),
                "has_trace": bool(d.get("tool_calls")),
            })
        rows.sort(key=lambda r: (r["date"], r["who"]), reverse=True)
        return rows, None
    except Exception as e:  # noqa: BLE001 -- the page still works without it
        return [], f"{type(e).__name__}: {str(e)[:200]}"


# ------------------------------------------------------------- report detail

def _lineup_view(submitted: list[str], responses: list[dict]) -> dict:
    """What was submitted, and what the slate made of it (get_lineup_status)."""
    lu = next((r.get("response") or {} for r in responses
               if r.get("name") == "get_lineup_status"), None)
    if lu is None:
        return {"submitted": submitted, "status": None, "players": [],
                "not_found": [], "name_guesses": {}}
    return {"submitted": submitted, "status": lu.get("status"),
            "message": lu.get("message"),
            "players": [{k: p.get(k) for k in ("name", "position", "team", "game",
                                                "salary", "injury_status", "injury_detail")}
                        for p in lu.get("lineup") or []],
            "not_found": lu.get("not_found_on_slate") or [],
            "name_guesses": lu.get("name_guesses") or {}}


def _detail_eval(stem: str, scenario: str) -> dict:
    payload = _load_run(stem)
    r = next((x for x in payload.get("results", []) if x["scenario"] == scenario), None)
    if r is None:
        abort(404, "no such scenario in that run")
    trace = (payload.get("traces") or {}).get(scenario) or {}
    responses = _eval_responses(trace)
    item_id = f"{stem}:{scenario}"
    graded = item_id in _graded_ids()
    judged = bool(r.get("judge_scores"))
    return {
        "kind": "eval", "title": scenario, "subtitle": r.get("description", ""),
        "when": payload.get("at", ""), "who": "eval harness (fixtures)",
        "source": "eval", "model": AGENT_MODEL, "elapsed_s": r.get("elapsed_s"),
        "lineup": _lineup_view(trace.get("lineup") or [], responses),
        "tokens": r.get("tokens") or None, "cost_usd": r.get("cost_usd"),
        "judge_cost_usd": r.get("judge_cost_usd"),
        "subcalls": r.get("subcalls") or [], "subcall_cost_usd": r.get("subcall_cost_usd"),
        "total_cost_usd": r.get("total_cost_usd"),
        "model_calls": trace.get("model_calls") or [],
        "tool_calls": _eval_tool_calls(trace),
        "report_check": audit(r.get("report", ""), responses) if trace else [],
        "scenario_checks": {"passed": r["checks_passed"], "total": r["checks_total"],
                            "failures": r.get("failures", [])},
        "judge": ({"scores": r["judge_scores"], "reasons": r.get("judge_reasons", {}),
                   "quality": r.get("quality")} if judged and graded else None),
        "judge_hidden": judged and not graded, "grade_id": item_id,
        "text": r.get("report", ""), "agent_text": None, "agent_error": r.get("judge_error"),
    }


def _detail_prod(uid: str, date: str) -> dict:
    snap = _db().collection("users").document(uid).collection("reports").document(date).get()
    if not snap.exists:
        abort(404, "no such report")
    d = snap.to_dict() or {}
    calls = []
    for c in d.get("tool_calls") or []:
        try:
            result = json.loads(c.get("result_json") or "null")
        except json.JSONDecodeError:
            result = c.get("result_json")  # truncated: show the raw text
        calls.append({"name": c.get("name"), "args": json.loads(c.get("args_json") or "{}"),
                      "result": result, "truncated": c.get("truncated", False),
                      "result_chars": c.get("result_chars")})
    responses = [{"name": c["name"], "response": c["result"]} for c in calls
                 if isinstance(c["result"], dict)]
    return {
        "kind": "prod", "title": f"{uid} · {date}", "subtitle": "Daily report (12:30 job)",
        "when": date, "who": uid, "source": d.get("source", ""),
        "model": d.get("model"), "agent": d.get("agent"), "elapsed_s": d.get("elapsed_s"),
        "lineup": _lineup_view(d.get("lineup") or [], responses),
        "tokens": d.get("tokens"), "cost_usd": d.get("cost_usd"), "judge_cost_usd": None,
        "subcalls": d.get("subcalls") or [], "subcall_cost_usd": d.get("subcall_cost_usd"),
        "total_cost_usd": d.get("total_cost_usd"),
        "model_calls": d.get("model_calls") or [], "tool_calls": calls,
        "report_check": d.get("checks") or [],
        "scenario_checks": None, "judge": None, "judge_hidden": False, "grade_id": None,
        "text": d.get("text", ""), "agent_text": d.get("agent_text"),
        "agent_error": d.get("agent_error"),
        "legacy": not d.get("tool_calls"),
    }


# ---------------------------------------------------------------- grading

def _grading_queue() -> list[dict]:
    """Judged reports you have not graded, newest run first.

    Newest RUN first, all of its scenarios -- not the newest few reports. A
    run covers every scenario once, so grading one run end to end samples
    the whole suite instead of whichever scenario happened to run last.
    """
    human = _load_human()
    items = [i for i in _all_judged_results() if i["id"] not in human]
    items.sort(key=lambda i: i["id"].split(":", 1)[0], reverse=True)
    return items


# ----------------------------------------------------------- control tower
#
# The operator's view: is production healthy, is it obeying its rules, what
# does it cost, is its data fresh. Same signals Cloud Monitoring alerts on
# (deploy/MONITORING.md) -- Monitoring pages, this explains. Every source is
# read independently and fails on its own: an expired gcloud login blanks one
# panel, not the page.

ML_PROJECT = "fantasy-football-ml-503718"
REGION = "us-central1"
TOWER_DAYS = 14
COST_ALERT_USD = 0.04          # same line as deploy/monitoring/report-cost.json
JOBS = [  # (project, job, what it is)
    (PROJECT, "gameday-report", "Daily report, 12:30 ET"),
    (PROJECT, "gameday-canary", "Canary, Thu 10:00 ET"),
    (ML_PROJECT, "news-archive", "News archive refresh, daily 11:00 ET"),
    (ML_PROJECT, "project-slate", "Projections, Thu / Fri / Sun"),
]
# Freshness limits, hours. Each is the longest gap the schedule allows plus slack.
FRESH = {"archive": 36, "projections": 100, "report": 30, "canary": 24 * 8}


def _gcp():
    import google.auth
    import google.auth.transport.requests as tr
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return tr.AuthorizedSession(creds)


def _hours_since(ts) -> float | None:
    if ts is None:
        return None
    from datetime import timezone
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - ts).total_seconds() / 3600, 1)


def _tower_days() -> list[str]:
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("America/New_York")).date()
    return [(today - timedelta(days=i)).isoformat() for i in range(TOWER_DAYS - 1, -1, -1)]


def _warned(tool_calls: list[dict]) -> list[str]:
    """source_warnings carried by any tool result in a stored report."""
    out = []
    for c in tool_calls or []:
        try:
            res = json.loads(c.get("result_json") or "null")
        except json.JSONDecodeError:
            continue
        if isinstance(res, dict):
            out += [str(w) for w in res.get("source_warnings") or []]
    return out


def _t_reports() -> dict:
    """SLA, business rules, FinOps and policy hits, from the stored reports."""
    db = _db()
    days = _tower_days()
    active = [s.id for s in db.collection("users").where("active", "==", True).stream()]
    by_day = {d: {"date": d, "agent": 0, "fallback": 0, "missing": 0, "reports": 0,
                  "agent_cost": 0.0, "tool_cost": 0.0, "costed": 0, "floor": False} for d in days}
    seen: dict[str, set] = {d: set() for d in days}
    rules: dict[str, dict] = {}
    fails, warnings, checked, latest = [], [], 0, None
    for snap in db.collection_group("reports").stream():
        date, uid = snap.id, snap.reference.parent.parent.id
        if date not in by_day:
            continue
        d = snap.to_dict() or {}
        row = by_day[date]
        seen[date].add(uid)
        row["reports"] += 1
        row["fallback" if d.get("source") == "fallback" else "agent"] += 1
        latest = max(latest or date, date)
        # Before 10/19 only the agent's own calls were metered: that cost is a
        # floor (no in-tool calls), flagged so the chart can say so.
        if d.get("cost_usd") is not None and not d.get("reused_from"):
            row["agent_cost"] += d.get("cost_usd") or 0
            row["tool_cost"] += d.get("subcall_cost_usd") or 0
            row["costed"] += 1
            row["floor"] = row["floor"] or d.get("total_cost_usd") is None
        if d.get("source") == "agent":
            checked += 1
            for c in d.get("checks") or []:
                r = rules.setdefault(c.get("rule", "?"), {"rule": c.get("rule", "?"), "runs": 0, "failed": 0})
                r["runs"] += 1
                r["failed"] += 0 if c.get("passed") else 1
            if d.get("problems"):
                fails.append({"date": date, "uid": uid, "problems": d["problems"]})
        for w in _warned(d.get("tool_calls")):
            warnings.append({"date": date, "uid": uid, "warning": w})
    for d in days:
        # A user active today with no report that day: no lineup, or a failure.
        # Measured against TODAY's active users -- a user added mid-window
        # shows as missing on the days before they existed.
        by_day[d]["missing"] = len([u for u in active if u not in seen[d]]) if by_day[d]["reports"] else 0
    fails.sort(key=lambda f: f["date"], reverse=True)
    return {"days": list(by_day.values()), "active_users": len(active),
            "rules": sorted(rules.values(), key=lambda r: -r["failed"]),
            "checked": checked, "failures": fails[:20], "warnings": warnings[-20:],
            "latest_report": latest}


def _t_jobs() -> list[dict]:
    s = _gcp()
    out = []
    for project, job, about in JOBS:
        url = (f"https://run.googleapis.com/v2/projects/{project}/locations/{REGION}"
               f"/jobs/{job}/executions?pageSize=8")
        r = s.get(url, timeout=20)
        if not r.ok:
            out.append({"job": job, "project": project, "about": about,
                        "error": f"HTTP {r.status_code}: {r.text[:150]}"})
            continue
        runs = []
        for e in r.json().get("executions", []):
            done = e.get("completionTime")
            status = ("running" if not done else "failed" if e.get("failedCount") else
                      "cancelled" if e.get("cancelledCount") else "ok")
            dur = None
            if done and e.get("startTime"):
                dur = round((datetime.fromisoformat(done.replace("Z", "+00:00"))
                             - datetime.fromisoformat(e["startTime"].replace("Z", "+00:00"))
                             ).total_seconds())
            runs.append({"name": e["name"].rsplit("/", 1)[-1], "created": e.get("createTime"),
                         "status": status, "seconds": dur})
        out.append({"job": job, "project": project, "about": about, "runs": runs,
                    "console": f"https://console.cloud.google.com/run/jobs/details/{REGION}/{job}/executions?project={project}"})
    return out


def _t_alerts() -> dict:
    s = _gcp()
    policies = []
    for project in (PROJECT, ML_PROJECT):
        r = s.get(f"https://monitoring.googleapis.com/v3/projects/{project}/alertPolicies", timeout=20)
        r.raise_for_status()
        policies += [{"project": project, "name": p.get("displayName"), "enabled": p.get("enabled", False)}
                     for p in r.json().get("alertPolicies", [])]
    # The events those policies fire on, last TOWER_DAYS: warning report
    # lines, scheduler failures. Failed job runs are in the jobs panel.
    since = f'timestamp>="{_tower_days()[0]}T00:00:00Z"'
    flt = (f'{since} AND ((jsonPayload.event="report" AND severity>=WARNING '
           f'AND jsonPayload.dry_run=false) OR (resource.type="cloud_scheduler_job" AND severity>=ERROR))')
    events = []
    for project in (PROJECT, ML_PROJECT):
        r = s.post("https://logging.googleapis.com/v2/entries:list", timeout=30, json={
            "resourceNames": [f"projects/{project}"], "filter": flt,
            "orderBy": "timestamp desc", "pageSize": 50})
        r.raise_for_status()
        for e in r.json().get("entries", []):
            p = e.get("jsonPayload") or {}
            events.append({"time": e.get("timestamp"), "project": project, "severity": e.get("severity"),
                           "kind": "report" if p.get("event") == "report" else "scheduler",
                           "uid": p.get("uid"), "source": p.get("source"),
                           "text": p.get("message") or e.get("textPayload")
                                   or json.dumps(p.get("status") or p)[:200]})
    events.sort(key=lambda e: e["time"] or "", reverse=True)
    return {"policies": policies, "events": events[:40],
            "console": f"https://console.cloud.google.com/monitoring/alerting?project={PROJECT}"}


def _t_data() -> dict:
    from google.cloud import bigquery
    bq = bigquery.Client(project=ML_PROJECT)
    q = f"""
      SELECT
        (SELECT MAX(published) FROM `{ML_PROJECT}.dfs.news_chunks`
         WHERE published >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
           AND screen = 'clean') AS archive_newest,
        (SELECT MAX(run_ts) FROM `{ML_PROJECT}.dfs.projections`) AS projections_newest,
        (SELECT COUNT(*) FROM `{ML_PROJECT}.dfs.news_chunks`) AS chunks,
        (SELECT COUNTIF(screen = 'clean') FROM `{ML_PROJECT}.dfs.news_chunks`) AS clean,
        (SELECT COUNTIF(screen = 'instruction') FROM `{ML_PROJECT}.dfs.news_chunks`) AS instruction,
        (SELECT COUNTIF(screen IS NULL OR screen = 'unchecked') FROM `{ML_PROJECT}.dfs.news_chunks`) AS unchecked
    """
    r = dict(next(iter(bq.query(q).result())))
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}


def _tower() -> dict:
    from concurrent.futures import ThreadPoolExecutor
    parts = {"reports": _t_reports, "jobs": _t_jobs, "alerts": _t_alerts, "data": _t_data}
    out, errors = {}, {}
    with ThreadPoolExecutor(max_workers=len(parts)) as ex:
        futs = {k: ex.submit(f) for k, f in parts.items()}
        for k, f in futs.items():
            try:
                out[k] = f.result()
            except Exception as e:  # noqa: BLE001 -- one source down is one panel down
                out[k], errors[k] = None, f"{type(e).__name__}: {str(e)[:200]}"

    rep, jobs, data = out["reports"], out["jobs"] or [], out["data"]
    canary = next((j for j in jobs if j["job"] == "gameday-canary"), {})
    last_ok = next((r["created"] for r in canary.get("runs", []) if r["status"] == "ok"), None)
    fresh = [
        {"what": "News archive (newest clean item)", "limit_h": FRESH["archive"],
         "hours": _hours_since(data and data.get("archive_newest"))},
        {"what": "Projections (newest run)", "limit_h": FRESH["projections"],
         "hours": _hours_since(data and data.get("projections_newest"))},
        {"what": "Production report (newest day stored)", "limit_h": FRESH["report"],
         "hours": _hours_since(f"{rep['latest_report']}T16:30:00+00:00")
                  if rep and rep.get("latest_report") else None},
        {"what": "Canary (last passing run)", "limit_h": FRESH["canary"], "hours": _hours_since(last_ok)},
    ]
    for f in fresh:
        f["ok"] = f["hours"] is not None and f["hours"] <= f["limit_h"]

    days = (rep or {}).get("days", [])
    costed = [d for d in days[-7:] if d["costed"] and not d["floor"]]
    per_report = (sum(d["agent_cost"] + d["tool_cost"] for d in costed)
                  / sum(d["costed"] for d in costed)) if costed else None
    reports_per_day = (sum(d["costed"] for d in costed) / len(costed)) if costed else None
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reports": rep, "jobs": jobs, "alerts": out["alerts"], "data": data,
        "freshness": fresh, "errors": errors,
        "finops": {"per_report_7d": per_report, "reports_per_day_7d": reports_per_day,
                   "month_projection": per_report * reports_per_day * 30
                   if per_report and reports_per_day else None,
                   "alert_usd": COST_ALERT_USD},
    }


# --------------------------------------------------------------------- app

app = Flask(__name__)


@app.get("/")
def index():
    return send_file(HERE / "dashboard.html")


@app.get("/api/meta")
def meta():
    return jsonify(
        criteria=[{"key": c.key, "name": c.name, "question": c.question,
                   "anchor_1": c.anchor_1, "anchor_5": c.anchor_5, "weight": c.weight}
                  for c in CRITERIA],
        tools=TOOLS, core_tools=CORE_TOOLS, tool_about=TOOL_ABOUT,
        prices=PRICES_PER_M, prices_checked=PRICES_CHECKED,
        judge_model=JUDGE_MODEL, agent_model=AGENT_MODEL, min_useful=MIN_USEFUL_SAMPLE)


@app.get("/api/overview")
def overview():
    runs = _runs()
    latest_stem, latest = runs[-1] if runs else (None, {"results": []})
    human = _load_human()
    hist = []
    for stem, p in runs[::-1]:
        res = p.get("results", [])
        hist.append({"run": stem, "at": p.get("at", ""), "reports": len(res),
                     "checks_passed": sum(r["checks_passed"] for r in res),
                     "checks_total": sum(r["checks_total"] for r in res),
                     "clean": sum(r["checks_passed"] == r["checks_total"] for r in res),
                     "quality": _mean([r.get("quality") for r in res]),
                     "avg_tools": _mean([r.get("tool_count") for r in res]),
                     "avg_seconds": _mean([r.get("elapsed_s") for r in res]),
                     "avg_tokens": _mean([_tok_total(r.get("tokens")) for r in res]),
                     "avg_cost": _mean([r.get("cost_usd") for r in res]),
                     "avg_total_cost": _mean([r.get("total_cost_usd") for r in res])})
    prod, prod_err = _prod_rows()
    return jsonify(
        latest={"run": latest_stem, "at": latest.get("at", ""),
                "rows": [_eval_row(latest_stem, latest, r) for r in latest.get("results", [])],
                "judge_costs": [r.get("judge_cost_usd") for r in latest.get("results", [])]},
        history=hist, alignment=alignment(human), graded=len(human),
        to_grade=len(_grading_queue()),
        baseline_quality=(load_baseline() or {}).get("suite_quality"),
        tolerance=QUALITY_TOLERANCE, prod=prod[:10], prod_error=prod_err)


@app.get("/api/tower")
def tower():
    return jsonify(_tower())


@app.get("/api/reports")
def reports():
    prod, err = _prod_rows()
    evals = []
    for stem, payload in _runs()[::-1]:
        evals += [_eval_row(stem, payload, r) for r in payload.get("results", [])]
    return jsonify(prod=prod, prod_error=err, eval=evals)


@app.get("/api/report/eval/<stem>/<scenario>")
def report_eval(stem, scenario):
    return jsonify(_detail_eval(stem, scenario))


@app.get("/api/report/prod/<uid>/<date>")
def report_prod(uid, date):
    if "/" in uid or "/" in date:
        abort(400, "bad id")
    return jsonify(_detail_prod(uid, date))


@app.get("/api/grade/next")
def grade_next():
    queue = _grading_queue()
    if not queue:
        return jsonify(item=None, remaining=0)
    item = queue[0]
    stem, _, scenario = item["id"].partition(":")
    trace = (_load_run(stem).get("traces") or {}).get(scenario) or {}
    # Deliberately NOT included: judge_scores and judge_reasons. Blind grading.
    return jsonify(remaining=len(queue), item={
        "id": item["id"], "scenario": item["scenario"], "desc": item["desc"],
        "report": item["report"], "tool_calls": _eval_tool_calls(trace),
        "detail_link": f"#report/eval/{stem}/{scenario}"})


@app.post("/api/grade")
def grade():
    body = request.get_json(silent=True) or {}
    item_id = body.get("id", "")
    scores = body.get("scores") or {}
    judged = {i["id"]: i for i in _all_judged_results()}
    if item_id not in judged:
        abort(400, "unknown report id")
    keys = {c.key for c in CRITERIA}
    if set(scores) != keys or not all(isinstance(v, int) and 1 <= v <= 5
                                      for v in scores.values()):
        abort(400, "score every criterion 1-5")

    human = _load_human()
    human[item_id] = {"scenario": judged[item_id]["scenario"], "scores": scores,
                      "note": str(body.get("note", ""))[:2000],
                      "graded_at": datetime.now().isoformat(timespec="seconds"),
                      "via": "dashboard"}
    _save_human(human)

    # Now -- and only now -- the judge's side.
    item = judged[item_id]
    return jsonify(
        yours=scores, yours_percent=to_percent(scores),
        judge=item["judge_scores"], judge_percent=to_percent(item["judge_scores"]),
        reasons=item["judge_reasons"], graded=len(human))


@app.errorhandler(400)
@app.errorhandler(404)
def json_error(e):
    return jsonify(error=e.description), e.code


if __name__ == "__main__":
    print(f"Eval dashboard: http://localhost:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)
