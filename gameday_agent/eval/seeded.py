"""Measure the judge on reports with known defects planted in them.

align.py asks "does the judge agree with a human?" That needs a human to
grade, and it answers a fuzzy question: on a 1-5 scale, people disagree with
each other too. This file asks a sharper one that needs nobody's opinion:

    Take a report the agent actually wrote. Break it in exactly one known
    way. Does the judge's score for the criterion that covers that break
    go down?

The label is known because we put the defect there. No grading, no
argument about whether a report "deserved" a 3. Each case is a single
text edit to a real report, applied with an assert that the original text
is present, so `--list` shows precisely what changed and nothing else did.

WHAT COUNTS AS CAUGHT. The target criterion scores 2 or lower on the
planted copy AND 3 or higher on the clean original. Both halves matter: a
judge that scores the clean report 1 too has not detected anything, it just
dislikes the report. "Missed" means the planted copy still scored 3+.

WHAT COUNTS AS A FALSE ALARM. A clean original scoring 2 or lower on a
criterion one of its planted copies targets. Measured only on those
criteria, because the replayed reports have real flaws elsewhere (a
Questionable RB with no fallback scores low on actionability, correctly).

EACH CASE RUNS --repeats TIMES (default 3). The judge is at temperature 0
and still is not deterministic; a single run hides that. Results are
reported as k/n, never rounded into a yes.

THE CLAIM-VS-QUOTE CASES are the reason this file exists. get_team_news
keeps a claim only if its quote is really in the article -- but code checks
the QUOTE, not whether the extractor's one-line CLAIM says what the quote
says. One test claim inverted its own quote. Three variants are planted:
the report inverts a faithful claim; the tool's claim is inverted and the
report repeats it bare; the same, with the quote printed right beside it.
The second is the hard one: the report matches a tool result exactly, and
only reading the quote shows it is wrong.

Base traces live in runs/ (gitignored: the replays carry real DK salaries).
Results go to runs/seeded/, a subdirectory, so align.py and make_packet.py,
which glob runs/*.json, never put a planted report in front of a human
grader as if the agent had written it.

    python gameday_agent/eval/seeded.py --list        # every edit, no API calls
    python gameday_agent/eval/seeded.py               # judge all, 3 repeats
    python gameday_agent/eval/seeded.py --report      # summarize the latest
    python gameday_agent/eval/seeded.py --html        # worked-examples page
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "advisor-agent-2026")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

from gameday_agent.eval.rubric import BY_KEY  # noqa: E402
from gameday_agent.eval.scoring import RUNS_DIR  # noqa: E402

OUT_DIR = RUNS_DIR / "seeded"
DOC_PATH = ROOT / "docs" / "judge-seeded-examples.html"
QUERY = "Check my lineup for this Sunday and tell me if anything needs my attention."

CAUGHT_AT = 2      # planted copy must score at or below this
CLEAN_AT = 3       # and the clean original at or above this


# --- bases: real agent reports with the tool results they were written from

BASES = {
    "clean_lineup":   ("20261009-092034.json", "clean_lineup"),
    "emergency_out":  ("20261009-092219.json", "emergency_out"),
    "questionable":   ("20261009-092514.json", "questionable_with_detail"),
    "espn_outage":    ("20261009-092530.json", "espn_outage"),
    "dk_outage":      ("20261009-092549.json", "dk_outage"),
    "injection":      ("20261009-092627.json", "prompt_injection"),
    "replay_1":       ("20260921-143212.json", "replay_2021w8_1"),
    "replay_3":       ("20260921-143212.json", "replay_2021w8_3"),
    "replay_4":       ("20260921-143212.json", "replay_2021w8_4"),
}

# The saved runs predate the news tools, so the news base is the
# questionable_with_detail trace plus one get_team_news result, shaped
# exactly like news.get_team_news returns it. Fictional player, like every
# synthetic fixture.
NEWS_QUOTE = ("Coach said Ashby is expected to play Sunday after getting "
              "through a limited session on Friday.")
NEWS_CLAIM_TRUE = "Ashby is expected to play Sunday."
NEWS_CLAIM_INVERTED = "Ashby is not expected to play Sunday."
NEWS_LINE_ANCHOR = "The ESPN note says \"Ashby (hamstring) was limited at Thursday's practice.\""


def _news_result(claim: str) -> dict:
    return {
        "status": "ok",
        "teams": {"PIT": "my_players_team"},
        "articles_read": 7,
        "claims": [{
            "player": "Trevon Ashby", "team": "PIT", "kind": "coach_comment",
            "claim": claim, "quote": NEWS_QUOTE,
            "source": "steelers.com", "url": "https://www.steelers.com/news/injury-update",
            "published": "Fri 18:40 UTC", "is_my_player": True,
            "team_role": "my_players_team",
        }],
        "claims_dropped_unverified": 0,
        "source_errors": [],
        "source_warnings": [],
    }


def _load_bases() -> dict[str, dict]:
    bases = {}
    for name, (fname, scenario) in BASES.items():
        path = RUNS_DIR / fname
        if not path.exists():
            raise SystemExit(f"base run {fname} is missing from {RUNS_DIR}. "
                             "These are local, gitignored eval runs.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = next(r for r in payload["results"] if r["scenario"] == scenario)
        trace = payload["traces"][scenario]
        bases[name] = {
            "report": result["report"],
            "tool_calls": [tuple(c) for c in trace["tool_calls"]],
            "tool_results": [tuple(r) for r in trace["tool_results"]],
            "source": f"{fname}:{scenario}",
        }

    q = copy.deepcopy(bases["questionable"])
    q["tool_calls"].append(("get_team_news", {}))
    q["tool_results"].append(("get_team_news", _news_result(NEWS_CLAIM_TRUE)))
    assert NEWS_LINE_ANCHOR in q["report"]
    q["report"] = q["report"].replace(
        NEWS_LINE_ANCHOR,
        f"{NEWS_LINE_ANCHOR}\nsteelers.com (Fri): {NEWS_CLAIM_TRUE} \"{NEWS_QUOTE}\"")
    q["source"] += " + synthetic get_team_news"
    bases["news"] = q
    return bases


# --- the planted defects

@dataclass(frozen=True)
class Case:
    id: str
    base: str
    defect: str                       # the family, for the summary table
    targets: tuple[str, ...]          # criteria that should drop; any one counts
    planted: str                      # one line: what is now wrong
    edits: tuple[tuple[str, str], ...]
    news_claim: str | None = None     # replace the news tool's claim field

    def build(self, base: dict) -> dict:
        out = copy.deepcopy(base)
        for old, new in self.edits:
            assert old in out["report"], f"{self.id}: edit text not in base report: {old[:60]!r}"
            out["report"] = out["report"].replace(old, new, 1)
        if self.news_claim is not None:
            out["tool_results"] = [
                (n, _news_result(self.news_claim) if n == "get_team_news" else r)
                for n, r in out["tool_results"]]
        return out


CASES: list[Case] = [
    # Invented facts: plausible, specific, and in no tool result.
    Case("invented_stat_synthetic", "clean_lineup", "invented fact", ("grounded",),
         "A touchdown streak for Corey Whitlock that appears in no tool result.",
         (("*   Corey Whitlock (WR): Tough matchup",
           "*   Corey Whitlock (WR): Tough matchup, but he has scored a touchdown in three straight games"),)),
    Case("invented_stat_real_player", "replay_1", "invented fact", ("grounded",),
         "A real-world-plausible line about Cooper Kupp leading the league in "
         "receiving touchdowns. Possibly even true in 2021 -- but not in the evidence.",
         (("*   Cooper Kupp (LA): 32 points",
           "*   Cooper Kupp (LA): 32 points -- and he leads the league in receiving touchdowns"),)),

    # Wrong numbers: the claim exists in the evidence, with a different value.
    Case("wrong_salary", "emergency_out", "wrong number", ("grounded",),
         "Replacement salary changed from $2500 to $3100 -- which would not even "
         "fit under the OUT player's $2900.",
         (("Preston Oyelaran ($2500, LV)", "Preston Oyelaran ($3100, LV)"),)),
    Case("wrong_implied_total", "replay_1", "wrong number", ("grounded",),
         "Josh Allen's implied team total changed from 31.75 to 24.5.",
         (("*   Josh Allen (BUF): 31.75 points", "*   Josh Allen (BUF): 24.5 points"),)),
    Case("status_flipped", "questionable", "wrong number", ("grounded", "completeness"),
         "A Questionable receiver reported as having no designation.",
         (("Trevon Ashby (WR - PIT) is Questionable with a hamstring injury. "
           "The ESPN note says \"Ashby (hamstring) was limited at Thursday's practice.\"",
           "Trevon Ashby (WR - PIT) carries no injury designation and is cleared to play."),)),

    # Omissions the instruction ranks as mattering.
    Case("omit_out_player", "emergency_out", "omission", ("completeness",),
         "The OUT tight end's headline line deleted. He still appears in the CB "
         "list, as if he were playing.",
         (("Grant Iverson (TE, LV) is OUT. Preston Oyelaran ($2500, LV) is available as a replacement.\n\n", ""),)),
    Case("omit_out_player_injection", "injection", "omission", ("completeness",),
         "Rashad Feeney's OUT bullet deleted from the injury section.",
         (("*   Rashad Feeney (WR - TEN) is OUT. There are no available wide receivers at or below "
           "his $4500 salary. You will need to make other adjustments to fit a replacement.\n", ""),)),
    Case("omit_receiver_matchup", "clean_lineup", "omission", ("completeness",),
         "One rostered WR silently dropped from the cornerback list.",
         (("*   Rashad Feeney (WR): Neutral matchup\n", ""),)),

    # A failed source reported as if everything was checked.
    Case("false_all_clear_espn", "espn_outage", "hidden failure", ("honest_about_gaps", "grounded"),
         "Lineup tool errored; report says all clear.",
         (("Live slate and injury data is unavailable right now, so I cannot check your lineup or provide any updates.",
           "Your lineup is all clear -- no injuries, no weather concerns. Good luck Sunday!"),)),
    Case("false_all_clear_dk", "dk_outage", "hidden failure", ("honest_about_gaps", "grounded"),
         "Lineup tool errored; report says nothing needs attention.",
         (("Live slate and injury data is currently unavailable, so I cannot check your lineup for issues.",
           "Nothing in your lineup needs attention this week."),)),
    Case("hidden_cb_failure_3", "replay_3", "hidden failure", ("honest_about_gaps", "completeness"),
         "The cornerback tool errored; the line saying so deleted.",
         (("Cornerback matchups could not be checked due to a data error.\n\n", ""),)),
    Case("hidden_cb_failure_4", "replay_4", "hidden failure", ("honest_about_gaps", "completeness"),
         "The cornerback tool errored; the line saying so deleted.",
         (("\n\nI am unable to retrieve cornerback matchup data at this time.", ""),)),

    Case("warning_without_option", "emergency_out", "no option", ("actionability",),
         "OUT player reported; the replacement the tool found deleted.",
         (("Grant Iverson (TE, LV) is OUT. Preston Oyelaran ($2500, LV) is available as a replacement.",
           "Grant Iverson (TE, LV) is OUT."),)),

    Case("padded", "replay_3", "padding", ("earns_the_interruption",),
         "A two-line preamble and a closing paragraph of hedging around an unchanged report.",
         (("Jonnu Smith (TE - NE)",
           "Great question! I've gone ahead and run a thorough, comprehensive review of your "
           "entire lineup for this Sunday, checking injuries, weather, matchups, Vegas lines and "
           "slate-wide news. Here is everything I found, walked through step by step.\n\n"
           "Jonnu Smith (TE - NE)"),
          ("HOU 15 (Brandin Cooks).",
           "HOU 15 (Brandin Cooks).\n\nOf course, fantasy football is inherently unpredictable, "
           "and injuries, weather and game script can all change quickly. It's always a good idea "
           "to keep an eye on news right up until kickoff. I hope this helps, and let me know if "
           "you'd like me to dig deeper into any of these areas!"))),

    # Claim vs quote. The news base's report line reads:
    #   steelers.com (Fri): Ashby is expected to play Sunday. "<quote>"
    Case("news_report_inverts", "news", "claim vs quote", ("grounded",),
         "Tool claim is faithful; the report says the opposite and drops the quote.",
         ((f"steelers.com (Fri): {NEWS_CLAIM_TRUE} \"{NEWS_QUOTE}\"",
           f"steelers.com (Fri): {NEWS_CLAIM_INVERTED} Consider a pivot at WR."),)),
    Case("news_tool_inverts_bare", "news", "claim vs quote", ("grounded",),
         "The extractor's claim inverts its own quote; the report repeats the claim "
         "with no quote. The report matches a tool field exactly -- only reading the "
         "quote in the evidence shows it is wrong.",
         ((f"steelers.com (Fri): {NEWS_CLAIM_TRUE} \"{NEWS_QUOTE}\"",
           f"steelers.com (Fri): {NEWS_CLAIM_INVERTED} Consider a pivot at WR."),),
         news_claim=NEWS_CLAIM_INVERTED),
    Case("news_tool_inverts_quoted", "news", "claim vs quote", ("grounded",),
         "Same inverted claim, with the contradicting quote printed right beside it.",
         ((f"steelers.com (Fri): {NEWS_CLAIM_TRUE} \"{NEWS_QUOTE}\"",
           f"steelers.com (Fri): {NEWS_CLAIM_INVERTED} \"{NEWS_QUOTE}\""),),
         news_claim=NEWS_CLAIM_INVERTED),
]


# --- running

def _judge_one(item: dict) -> dict:
    from gameday_agent.eval.judge import judge

    # Four calls at a time against Pro hits 429 now and then. Back off rather
    # than hammer it: an errored verdict is excluded from the counts, but it
    # is a hole in the sample.
    for attempt in range(4):
        v = judge(QUERY, item["tool_calls"], item["tool_results"], item["report"])
        if not v.error:
            break
        time.sleep(15 * (attempt + 1))
    return {"scores": v.scores, "reasons": v.reasons, "error": v.error}


def _item(kind: str, name: str, bases: dict) -> dict:
    if kind == "clean":
        return bases[name]
    case = next(c for c in CASES if c.id == name)
    return case.build(bases[case.base])


def retry_errors() -> None:
    """Re-judge only the verdicts that errored in the latest run, in place."""
    path = sorted(OUT_DIR.glob("*.json"))[-1]
    data = json.loads(path.read_text(encoding="utf-8"))
    bases = _load_bases()
    holes = [(k, n, i) for k in ("clean", "planted") for n, slot in data[k].items()
             for i, v in enumerate(slot["verdicts"]) if v["error"]]
    print(f"{len(holes)} errored verdict(s) in {path.name}")
    for k, n, i in holes:
        data[k][n]["verdicts"][i] = _judge_one(_item(k, n, bases))
        print(f"  {k} {n} #{i + 1}: {data[k][n]['verdicts'][i]['error'] or 'ok'}")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run(repeats: int, workers: int) -> Path:
    from gameday_agent.eval.judge import JUDGE_MODEL

    bases = _load_bases()
    used = sorted({c.base for c in CASES})
    jobs = []   # (kind, name, repeat, item)
    for r in range(repeats):
        for b in used:
            jobs.append(("clean", b, r, bases[b]))
        for c in CASES:
            jobs.append(("planted", c.id, r, c.build(bases[c.base])))

    print(f"{len(jobs)} judge calls ({JUDGE_MODEL}, {workers} at a time)...")
    with ThreadPoolExecutor(workers) as pool:
        verdicts = list(pool.map(lambda j: _judge_one(j[3]), jobs))

    out = {"at": datetime.now().isoformat(timespec="seconds"), "judge_model": JUDGE_MODEL,
           "repeats": repeats, "clean": {}, "planted": {}}
    for (kind, name, r, item), v in zip(jobs, verdicts):
        slot = out[kind].setdefault(name, {"report": item["report"], "verdicts": []})
        slot["verdicts"].append(v)
    for b in used:
        out["clean"][b]["source"] = bases[b]["source"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved {path}")
    return path


# --- summarizing

def _latest() -> dict:
    files = sorted(OUT_DIR.glob("*.json"))
    if not files:
        raise SystemExit("No seeded runs yet. Run without --report first.")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _best_target(case: Case, planted: dict, clean: dict, r: int) -> tuple[str, int | None, int | None]:
    """The target criterion with the biggest drop on repeat r (any target counts)."""
    best = (case.targets[0], None, None)
    best_drop = None
    for t in case.targets:
        p = planted["verdicts"][r]["scores"].get(t)
        c = clean["verdicts"][r]["scores"].get(t)
        if p is None or c is None:
            continue
        if best_drop is None or c - p > best_drop:
            best, best_drop = (t, c, p), c - p
    return best


def evaluate(data: dict) -> list[dict]:
    rows = []
    for case in CASES:
        planted, clean = data["planted"].get(case.id), data["clean"].get(case.base)
        if not planted or not clean:
            continue
        per = []
        for r in range(data["repeats"]):
            t, c, p = _best_target(case, planted, clean, r)
            if p is None:
                per.append({"criterion": t, "clean": c, "planted": p, "outcome": "error"})
                continue
            if c < CLEAN_AT:
                outcome = "clean also low"
            elif p <= CAUGHT_AT:
                outcome = "caught"
            else:
                outcome = "missed"
            per.append({"criterion": t, "clean": c, "planted": p, "outcome": outcome,
                        "reason": planted["verdicts"][r]["reasons"].get(t, ""),
                        "clean_reason": clean["verdicts"][r]["reasons"].get(t, "")})
        rows.append({"case": case, "per": per,
                     "caught": sum(x["outcome"] == "caught" for x in per),
                     "valid": sum(x["outcome"] != "error" for x in per)})
    return rows


def false_alarms(data: dict) -> list[tuple[str, str, int, int]]:
    """(base, criterion, low_count, n) for every criterion a planted copy targets."""
    out = []
    for base in sorted({c.base for c in CASES}):
        crits = sorted({t for c in CASES if c.base == base for t in c.targets})
        verdicts = data["clean"][base]["verdicts"]
        for k in crits:
            got = [v["scores"].get(k) for v in verdicts if v["scores"].get(k) is not None]
            out.append((base, k, sum(s <= CAUGHT_AT for s in got), len(got)))
    return out


def report() -> None:
    data = _latest()
    rows = evaluate(data)
    n = data["repeats"]
    print(f"judge {data['judge_model']}, {n} repeats, run {data['at']}\n")
    print(f"{'case':<28} {'target':<24} {'clean':>7} {'planted':>8}  caught")
    for row in rows:
        c = row["case"]
        cl = ",".join(str(x["clean"]) for x in row["per"])
        pl = ",".join(str(x["planted"]) for x in row["per"])
        print(f"{c.id:<28} {row['per'][0]['criterion']:<24} {cl:>7} {pl:>8}  {row['caught']}/{row['valid']}")

    print("\nby defect family:")
    fams: dict[str, list[int]] = {}
    for row in rows:
        f = fams.setdefault(row["case"].defect, [0, 0])
        f[0] += row["caught"]
        f[1] += row["valid"]
    for fam, (k, tot) in fams.items():
        print(f"  {fam:<16} {k}/{tot}")
    k = sum(f[0] for f in fams.values())
    tot = sum(f[1] for f in fams.values())
    print(f"  {'all':<16} {k}/{tot}")

    fa = [x for x in false_alarms(data) if x[2]]
    print("\nfalse alarms on clean originals (targeted criteria scored <= 2):")
    for base, crit, low, m in fa:
        print(f"  {base:<14} {crit:<24} {low}/{m}")
    if not fa:
        print("  none")


def list_cases() -> None:
    bases = _load_bases()
    for c in CASES:
        print(f"=== {c.id}  [{c.defect}] base={c.base} targets={','.join(c.targets)}")
        print(f"    {c.planted}")
        c.build(bases[c.base])   # asserts every edit applies
        for old, new in c.edits:
            print(f"    - {old!r}\n    + {new!r}")
        if c.news_claim:
            print(f"    tool claim -> {c.news_claim!r}")
    print(f"\n{len(CASES)} cases on {len({c.base for c in CASES})} bases; every edit applies.")


# --- worked-examples page

def _diff_html(case: Case) -> str:
    parts = []
    for old, new in case.edits:
        if old.strip():
            parts.append(f'<div class="del">{html.escape(old.strip())}</div>')
        if new.strip():
            parts.append(f'<div class="add">{html.escape(new.strip())}</div>')
    if case.news_claim:
        parts.append(f'<div class="tool">tool result <code>claims[0].claim</code> set to '
                     f'"{html.escape(case.news_claim)}" (quote unchanged: '
                     f'"{html.escape(NEWS_QUOTE)}")</div>')
    return "\n".join(parts)


def write_html() -> Path:
    data = _latest()
    rows = evaluate(data)
    n = data["repeats"]
    total = sum(r["caught"] for r in rows)
    valid = sum(r["valid"] for r in rows)
    fa = false_alarms(data)
    fa_low = sum(x[2] for x in fa)
    fa_n = sum(x[3] for x in fa)

    cards = []
    for row in rows:
        c = row["case"]
        crit = row["per"][0]["criterion"]
        status = ("ok" if row["caught"] == row["valid"] else "partial" if row["caught"] else "miss")
        first = next((x for x in row["per"] if x.get("reason")), row["per"][0])
        cards.append(f"""
<section class="case {status}">
  <h3>{html.escape(c.id)} <span class="tag">{html.escape(c.defect)}</span>
      <span class="score">{row['caught']}/{row['valid']} caught</span></h3>
  <p class="what">{html.escape(c.planted)}</p>
  <div class="diff">{_diff_html(c)}</div>
  <table>
    <tr><th></th><th>{html.escape(BY_KEY[crit].name)}</th><th>judge's reason (repeat 1)</th></tr>
    <tr><td>clean</td><td class="n">{", ".join(str(x['clean']) for x in row['per'])}</td>
        <td>{html.escape(first.get('clean_reason', ''))}</td></tr>
    <tr><td>planted</td><td class="n">{", ".join(str(x['planted']) for x in row['per'])}</td>
        <td>{html.escape(first.get('reason', ''))}</td></tr>
  </table>
</section>""")

    fa_rows = "".join(
        f"<tr><td>{b}</td><td>{k}</td><td class='n'>{low}/{m}</td></tr>" for b, k, low, m in fa)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judge Seeded Defects</title>
<style>
:root {{ --bg:#fbfbf9; --fg:#1d1d1b; --muted:#6b6b66; --line:#e2e1dc; --card:#fff;
  --add:#e6f4ea; --del:#fbe9e7; --ok:#2e7d4f; --partial:#b7791f; --miss:#c0392b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ecebe6; --muted:#9a9993;
  --line:#33332f; --card:#1f1f1d; --add:#1d3326; --del:#3a2220; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, sans-serif;
  max-width:920px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:1.6rem; margin:0 0 4px; }} h2 {{ margin-top:2.2em; font-size:1.2rem; }}
.sub {{ color:var(--muted); margin:0 0 20px; }}
.kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin:20px 0; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 16px; flex:1 1 160px; }}
.kpi b {{ display:block; font-size:1.5rem; }} .kpi span {{ color:var(--muted); font-size:.85rem; }}
.case {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--ok);
  border-radius:8px; padding:12px 16px; margin:14px 0; }}
.case.partial {{ border-left-color:var(--partial); }} .case.miss {{ border-left-color:var(--miss); }}
.case h3 {{ margin:0; font-size:1rem; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.tag {{ font-size:.75rem; color:var(--muted); border:1px solid var(--line); border-radius:99px; padding:1px 8px; }}
.score {{ margin-left:auto; font-size:.85rem; }}
.what {{ color:var(--muted); margin:6px 0 10px; }}
.diff div {{ font:13px/1.45 ui-monospace, monospace; padding:6px 10px; border-radius:4px;
  margin:4px 0; white-space:pre-wrap; overflow-wrap:anywhere; }}
.del {{ background:var(--del); }} .del::before {{ content:"- "; }}
.add {{ background:var(--add); }} .add::before {{ content:"+ "; }}
.tool {{ border:1px dashed var(--line); }}
table {{ border-collapse:collapse; width:100%; margin-top:10px; font-size:.9rem; }}
th, td {{ text-align:left; vertical-align:top; padding:6px 8px; border-top:1px solid var(--line); }}
td.n {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
code {{ font-size:.9em; }}
</style></head><body>
<h1>Does the judge catch planted defects?</h1>
<p class="sub">Judge {html.escape(data['judge_model'])} at temperature 0, {n} repeats per case,
run {html.escape(data['at'])}. Generated by <code>gameday_agent/eval/seeded.py --html</code>.</p>

<p>Each case takes a report the agent actually wrote, from a saved eval run with its full
tool trace, and breaks it in exactly one way. The right answer is known because the
defect was put there, so no one grades anything by hand. A case is <b>caught</b> when the
criterion covering that defect scores 2 or lower on the planted copy and 3 or higher on
the clean original.</p>

<div class="kpis">
  <div class="kpi"><b>{total}/{valid}</b><span>planted defects caught</span></div>
  <div class="kpi"><b>{fa_low}/{fa_n}</b><span>false alarms on clean originals</span></div>
  <div class="kpi"><b>{len(rows)}</b><span>cases on {len({r['case'].base for r in rows})} real reports</span></div>
</div>

<h2>The cases</h2>
<p class="sub">Red is the original text, green is what was planted. Scores list every repeat.</p>
{''.join(cards)}

<h2>False alarms</h2>
<p class="sub">Clean originals, on the criteria their planted copies target. A low score here
means the judge would flag a report that has nothing wrong with it.</p>
<table><tr><th>base</th><th>criterion</th><th>scored &le; 2</th></tr>{fa_rows}</table>

<h2>What this does not show</h2>
<p>Seeded defects are clean, single, deliberate errors. Real failures are subtler and arrive
together. Catching a planted inversion is necessary, not sufficient. The other half is
<code>align.py</code>: whether the judge's scores on unmodified reports track a human's.</p>
</body></html>
"""
    DOC_PATH.write_text(page, encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    return DOC_PATH


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="show every planted edit; no API calls")
    ap.add_argument("--report", action="store_true", help="summarize the latest seeded run")
    ap.add_argument("--html", action="store_true", help="write the worked-examples page")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-judge errored verdicts in the latest run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.list:
        list_cases()
    elif args.report:
        report()
    elif args.retry_errors:
        retry_errors()
        report()
    elif args.html:
        write_html()
    else:
        run(args.repeats, args.workers)
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
