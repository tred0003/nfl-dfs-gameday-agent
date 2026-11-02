"""Ask's weekly exam: 50 start/sit questions on Sunday, graded on Tuesday by the box score.

    python webapp/ask_weekly.py ask              # Sunday 11:15 ET: ask this week's 50
    python webapp/ask_weekly.py ask --dry-run    # print the questions; ask nothing
    python webapp/ask_weekly.py grade            # Tuesday: grade every week not yet graded
    python webapp/ask_weekly.py report           # season to date, per week

Every other Ask eval (ask_eval/) grades answers against the archive: did it
cite, did it stay grounded, did a judge like it. None asks whether the advice
was RIGHT. This does. "Start A or B?" has an answer on Tuesday -- whoever
scored more DraftKings points -- so the grade needs no judge model at all.

THE QUESTIONS are made from the week's projection run, never written by hand:

  close         40 pairs at one position whose projections are within
                CLOSE_POINTS. A 10-point gap teaches nothing; close calls are
                where advice can earn its keep.
  questionable  10 pairs where one player is listed Questionable. News should
                matter most here: a Q player who sits scores zero.

THE TWO BASELINES. A coin flip is 50%. "Start the higher projection" is the
XGBoost model alone, no news, no Gemini -- saved beside every question at ask
time. Ask is only worth its cost if it beats the second one. One week is 50
noisy coin-flip-sized outcomes; trust the season line, not the week.

THE ASKING is the production code path: webapp/ask.ArchiveAgent, the same
news-mcp, the same model card, run from the web app's own image. Ask's
answer is prose, so a small model reads it and names the player it says to
start ("no pick" when it will not choose -- counted as a miss, not skipped).

STORAGE: dfs.ask_weekly (one row per question, written Sunday before kickoff)
and dfs.ask_weekly_grades (written Tuesday). Asking twice in one week writes
nothing the second time; grading only touches weeks with no grades yet.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

BQ_PROJECT = "fantasy-football-ml-503718"
ANSWERS = os.environ.get("ASK_WEEKLY_TABLE", f"{BQ_PROJECT}.dfs.ask_weekly")
GRADES = os.environ.get("ASK_WEEKLY_GRADES", f"{BQ_PROJECT}.dfs.ask_weekly_grades")
STATS = ("https://github.com/nflverse/nflverse-data/releases/download/"
         "stats_player/stats_player_week_{season}.parquet")
PICK_MODEL = "gemini-2.5-flash-lite"
UID = "ask-weekly"          # in the archive_ask log line, so Tower can tell it from users

N_CLOSE = 40
N_QUESTIONABLE = 10
CLOSE_POINTS = 3.0
# Close pairs per position; the Q pairs take any position. A short position
# hands its unused share to the next (WR, RB, TE, QB order).
QUOTA = {"WR": 14, "RB": 12, "TE": 8, "QB": 6}
# Below these the "decision" is between two players nobody would start.
FLOOR = {"QB": 12.0, "RB": 8.0, "WR": 8.0, "TE": 6.0}
RULED_OUT = {"O", "OUT", "IR", "D", "DOUBTFUL", "SUSP", "PUP", "NFI"}
TEMPLATES = [
    "Should I start {a} or {b} this week?",
    "{a} or {b} -- who's the better start this week?",
    "I have one {pos} spot. {a} or {b}?",
    "Who should I play this week, {a} or {b}?",
]
WORKERS = 4                 # Ask calls at once; news-mcp runs at most 2 instances
NO_PICK = "no pick"


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%H:%M:%S} {msg}", flush=True)


# ------------------------------------------------------------- the questions

def _eligible(players: list[dict]) -> list[dict]:
    return [p for p in players
            if p.get("position") in FLOOR and p.get("projection") is not None
            and p["projection"] >= FLOOR[p["position"]]
            and str(p.get("status") or "").upper() not in RULED_OUT]


def _partner(p: dict, pool: list[dict], used: set[str], rng: random.Random,
             healthy_only: bool = False) -> dict | None:
    near = [q for q in pool if q["name"] not in used and q["name"] != p["name"]
            and q["position"] == p["position"]
            and abs(q["projection"] - p["projection"]) <= CLOSE_POINTS
            and not (healthy_only and q.get("status"))]
    return rng.choice(near) if near else None


def make_questions(pool: dict) -> list[dict]:
    """The week's questions from one projection run. Same run, same questions."""
    rng = random.Random(int(hashlib.sha1(pool["run_id"].encode()).hexdigest()[:8], 16))
    players = _eligible(pool["players"])
    rng.shuffle(players)
    used: set[str] = set()
    pairs: list[tuple[str, dict, dict]] = []

    for p in [p for p in players if str(p.get("status") or "").upper() == "Q"]:
        if sum(k == "questionable" for k, *_ in pairs) >= N_QUESTIONABLE:
            break
        if p["name"] in used:
            continue
        q = _partner(p, players, used, rng, healthy_only=True)
        if q:
            pairs.append(("questionable", p, q))
            used |= {p["name"], q["name"]}

    want = N_CLOSE + N_QUESTIONABLE - len(pairs)
    spare = 0
    for pos, n in QUOTA.items():
        n = round(n * want / N_CLOSE) + spare
        got = 0
        for p in [p for p in players if p["position"] == pos]:
            if got >= n:
                break
            if p["name"] in used:
                continue
            q = _partner(p, players, used, rng)
            if q:
                pairs.append(("close", p, q))
                used |= {p["name"], q["name"]}
                got += 1
        spare = n - got

    out = []
    for i, (kind, a, b) in enumerate(pairs[:N_CLOSE + N_QUESTIONABLE]):
        if rng.random() < 0.5:
            a, b = b, a                       # the model's pick is not always named first
        out.append({
            "qid": f"{pool['season']}-w{pool['week']:02d}-{i:02d}",
            "season": pool["season"], "week": pool["week"], "run_id": pool["run_id"],
            "kind": kind, "position": a["position"],
            "question": rng.choice(TEMPLATES).format(a=a["name"], b=b["name"], pos=a["position"]),
            "player_a": a["name"], "player_b": b["name"],
            "proj_a": a["projection"], "proj_b": b["projection"],
            "status_a": a.get("status"), "status_b": b.get("status"),
            "model_pick": a["name"] if a["projection"] >= b["projection"] else b["name"],
        })
    return out


# --------------------------------------------------------------- the asking

def read_pick(client, question: dict, answer: str) -> str:
    """The player Ask's answer says to start, or NO_PICK."""
    from google.genai import types
    names = [question["player_a"], question["player_b"]]
    prompt = (f"Question: {question['question']}\n\nAnswer:\n{answer}\n\n"
              f"Which ONE of {names[0]} or {names[1]} does the answer recommend starting? "
              f"If it does not clearly choose one, say \"{NO_PICK}\".")
    resp = client.models.generate_content(
        model=PICK_MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0, response_mime_type="text/x.enum",
            response_schema={"type": "STRING", "enum": names + [NO_PICK]}))
    pick = (resp.text or "").strip()
    return pick if pick in names else NO_PICK


def ask_all(questions: list[dict], agent, client) -> list[dict]:
    def one(q: dict) -> dict:
        row = q | {"asked_at": datetime.now(timezone.utc).isoformat()}
        try:
            out = agent.ask(q["question"], uid=UID)
        except Exception as e:  # noqa: BLE001 -- a failed answer is a row, and a miss
            return row | {"pick": NO_PICK, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        try:
            pick = read_pick(client, q, out["answer"])
        except Exception as e:  # noqa: BLE001
            pick, row["error"] = NO_PICK, f"pick: {type(e).__name__}: {str(e)[:200]}"
        return row | {
            "answer": out["answer"], "pick": pick, "answer_type": out.get("answer_type"),
            "cited_share": out.get("cited_share"),
            "confidence": out.get("confidence"),
            "confidence_why": json.dumps(out.get("confidence_why", [])),
            "problems": json.dumps([p["kind"] for p in out.get("problems", [])]),
            "model_card_used": any(e["tool"] == "xgboost_projection" for e in out.get("evidence", [])),
            "versions": json.dumps(out.get("versions", {})),
            "cost_usd": out.get("cost_usd"), "elapsed_s": out.get("elapsed_s"),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(one, questions), 1):
            rows.append(r)
            if i % 10 == 0:
                log(f"  {i}/{len(questions)} asked")
    return rows


# --------------------------------------------------------------- the grading

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(name: str) -> str:
    n = str(name or "").lower().replace(".", "").replace("'", "").replace("-", " ")
    return re.sub(r"\s+", " ", _SUFFIX.sub("", n)).strip()


def load_points(season: int):
    """DK points per (normalized name, week), from nflverse's box scores."""
    import pandas as pd
    import requests
    from scoring import add_dk_points

    r = requests.get(STATS.format(season=season), timeout=120)
    r.raise_for_status()
    df = pd.read_parquet(io.BytesIO(r.content))
    df = df[df["season_type"].eq("REG") | df["season_type"].eq("POST")]
    # Same renames as fetch_nflverse._load_player_stats, which scoring.py expects.
    df = add_dk_points(df.rename(columns={"passing_interceptions": "interceptions"}))
    df["key"] = df["player_display_name"].map(norm)
    return df[["key", "week", "dk_points"]]


def grade_rows(answers: list[dict], points) -> list[dict]:
    """One grade per answer. A player with no box-score row that week but one
    in another week scored 0 (inactive, or no stats); a player in no week at
    all is a name the two sources spell differently -- ungradable, not 0."""
    seen = set(points["key"])
    out = []
    for a in answers:
        wk = points[points["week"].eq(a["week"])]
        pts = dict(zip(wk["key"], wk["dk_points"]))
        row = {"qid": a["qid"], "season": a["season"], "week": a["week"], "kind": a["kind"],
               "position": a["position"], "player_a": a["player_a"], "player_b": a["player_b"],
               "pick": a["pick"], "model_pick": a["model_pick"],
               "graded_at": datetime.now(timezone.utc).isoformat()}
        ka, kb = norm(a["player_a"]), norm(a["player_b"])
        if ka not in seen or kb not in seen:
            out.append(row | {"pts_a": None, "pts_b": None, "winner": None,
                              "ask_correct": None, "model_correct": None,
                              "note": "unmatched name"})
            continue
        pa, pb = float(pts.get(ka, 0.0)), float(pts.get(kb, 0.0))
        winner = None if pa == pb else (a["player_a"] if pa > pb else a["player_b"])
        out.append(row | {"pts_a": pa, "pts_b": pb, "winner": winner,
                          "ask_correct": None if winner is None else a["pick"] == winner,
                          "model_correct": None if winner is None else a["model_pick"] == winner,
                          "note": "tie" if winner is None else None})
    return out


def summarise(grades: list[dict]) -> dict:
    g = [r for r in grades if r["ask_correct"] is not None]

    def acc(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else None
    q = [r for r in g if r["kind"] == "questionable"]
    return {"graded": len(g), "ask": acc(g, "ask_correct"), "model": acc(g, "model_correct"),
            "ask_questionable": acc(q, "ask_correct"), "model_questionable": acc(q, "model_correct"),
            "no_pick": sum(r["pick"] == NO_PICK for r in g),
            "disagree": sum(r["pick"] != r["model_pick"] and r["pick"] != NO_PICK for r in g),
            "ask_when_disagree": acc([r for r in g if r["pick"] != r["model_pick"]
                                      and r["pick"] != NO_PICK], "ask_correct")}


# ----------------------------------------------------------------- BigQuery

def _bq():
    from google.cloud import bigquery
    return bigquery.Client(project=BQ_PROJECT)


def _schema(kind: str):
    from google.cloud import bigquery as b
    S, F, I, B, T = "STRING", "FLOAT64", "INT64", "BOOL", "TIMESTAMP"
    common = [("qid", S), ("season", I), ("week", I), ("kind", S), ("position", S),
              ("player_a", S), ("player_b", S), ("pick", S), ("model_pick", S)]
    extra = {
        "answers": [("run_id", S), ("question", S), ("proj_a", F), ("proj_b", F),
                    ("status_a", S), ("status_b", S), ("asked_at", T), ("answer", S),
                    ("answer_type", S), ("cited_share", F), ("problems", S),
                    ("model_card_used", B), ("versions", S), ("cost_usd", F),
                    ("elapsed_s", F), ("error", S), ("confidence", S), ("confidence_why", S)],
        "grades": [("pts_a", F), ("pts_b", F), ("winner", S), ("ask_correct", B),
                   ("model_correct", B), ("note", S), ("graded_at", T)],
    }[kind]
    return [b.SchemaField(n, t) for n, t in common + extra]


def _write(bq, table: str, kind: str, rows: list[dict]) -> None:
    from google.cloud import bigquery
    schema = _schema(kind)
    names = {f.name for f in schema}
    cfg = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND",
                                 create_disposition="CREATE_IF_NEEDED")
    bq.load_table_from_json([{k: v for k, v in r.items() if k in names} for r in rows],
                            table, job_config=cfg).result()


def _query(bq, sql: str) -> list[dict]:
    from google.api_core.exceptions import NotFound
    try:
        return [dict(r) for r in bq.query(sql).result()]
    except NotFound:
        return []


# ---------------------------------------------------------------- commands

def cmd_ask(dry_run: bool) -> int:
    from slate import BigQuerySlate

    pool = BigQuerySlate().pool()
    qs = make_questions(pool)
    log(f"week {pool['week']} run {pool['run_id']}: {len(qs)} questions "
        f"({sum(q['kind'] == 'questionable' for q in qs)} with a Questionable player)")
    if dry_run:
        for q in qs:
            log(f"  {q['qid']} [{q['kind']}] {q['question']}  model: {q['model_pick']} "
                f"({q['proj_a']} v {q['proj_b']})")
        return 0
    bq = _bq()
    done = _query(bq, f"SELECT COUNT(*) AS n FROM `{ANSWERS}` "
                      f"WHERE season = {pool['season']} AND week = {pool['week']}")
    if done and done[0]["n"]:
        log(f"week {pool['week']} already asked ({done[0]['n']} rows); nothing to do")
        return 0
    if len(qs) < N_CLOSE:
        log(f"ASK WEEKLY FAILED: only {len(qs)} questions -- is the projection run complete?")
        return 1

    import ask
    from google import genai
    started = time.monotonic()
    rows = ask_all(qs, ask.ArchiveAgent(slate=BigQuerySlate()),
                   genai.Client(vertexai=True, project=ask.PROJECT, location=ask.LOCATION))
    _write(bq, ANSWERS, "answers", rows)
    errors = sum(bool(r.get("error")) for r in rows)
    agree = sum(r["pick"] == r["model_pick"] for r in rows)
    # One line, fixed prefix, for a log-based metric.
    log(f"ASK WEEKLY asked={len(rows)} errors={errors} no_pick="
        f"{sum(r['pick'] == NO_PICK for r in rows)} agrees_with_model={agree} "
        f"cost_usd={sum(r.get('cost_usd') or 0 for r in rows):.3f} "
        f"minutes={(time.monotonic() - started) / 60:.1f}")
    return 1 if errors > len(rows) // 4 else 0


def cmd_grade() -> int:
    bq = _bq()
    todo = _query(bq, f"SELECT * FROM `{ANSWERS}` a" + (
        f" WHERE NOT EXISTS (SELECT 1 FROM `{GRADES}` g WHERE g.qid = a.qid)"
        if _exists(bq, GRADES) else ""))
    if not todo:
        log("nothing to grade")
        return 0
    status = 0
    for season in sorted({r["season"] for r in todo}):
        points = load_points(season)
        have = set(points["week"])
        for week in sorted({r["week"] for r in todo if r["season"] == season}):
            if week not in have:
                log(f"{season} week {week}: box scores not published yet; next run tries again")
                continue
            rows = grade_rows([r for r in todo if r["season"] == season and r["week"] == week],
                              points)
            _write(bq, GRADES, "grades", rows)
            s = summarise(rows)
            log(f"ASK WEEKLY GRADE season={season} week={week} " +
                " ".join(f"{k}={v}" for k, v in s.items()))
            if sum(r["note"] == "unmatched name" for r in rows) > len(rows) // 5:
                log("ASK WEEKLY GRADE FAILED: over a fifth of the names did not match the box scores")
                status = 1
    return status


def _exists(bq, table: str) -> bool:
    from google.api_core.exceptions import NotFound
    try:
        bq.get_table(table)
        return True
    except NotFound:
        return False


def cmd_report() -> int:
    bq = _bq()
    rows = _query(bq, f"SELECT * FROM `{GRADES}` ORDER BY season, week")
    if not rows:
        log("no grades yet")
        return 0
    for (season, week) in sorted({(r["season"], r["week"]) for r in rows}):
        s = summarise([r for r in rows if (r["season"], r["week"]) == (season, week)])
        print(f"{season} week {week:>2}: Ask {s['ask']}  model {s['model']}  "
              f"(n={s['graded']}, Q pairs Ask {s['ask_questionable']} / model "
              f"{s['model_questionable']}, no pick {s['no_pick']})")
    s = summarise(rows)
    print(f"season to date: Ask {s['ask']}  model {s['model']}  coin 0.5  (n={s['graded']}); "
          f"when Ask overrode the model it was right {s['ask_when_disagree']} of "
          f"{s['disagree']} times")
    # Calibration: the labels are worth showing only if high beats low here.
    cal = _query(bq, f"""
        SELECT a.confidence, COUNT(*) n, ROUND(AVG(IF(g.ask_correct, 1, 0)), 3) acc
        FROM `{GRADES}` g JOIN `{ANSWERS}` a USING (qid)
        WHERE g.ask_correct IS NOT NULL AND a.confidence IS NOT NULL
        GROUP BY 1 ORDER BY 1""")
    if cal:
        print("confidence: " + "  ".join(f"{r['confidence']} {r['acc']} (n={r['n']})"
                                         for r in cal))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["ask", "grade", "report"])
    p.add_argument("--dry-run", action="store_true", help="ask: print the questions only")
    a = p.parse_args()
    return {"ask": lambda: cmd_ask(a.dry_run), "grade": cmd_grade, "report": cmd_report}[a.command]()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import logging
    logging.disable(logging.INFO)
    sys.exit(main())
