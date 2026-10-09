"""Check the judge against a human before believing a word it says.

This is the step everybody skips, and skipping it is why a lot of eval
dashboards are decoration. An LLM judge produces a confident number for any
input you give it. Nothing about the number being produced by a model makes
it correlate with what a person would say. Until you have measured that
correlation, the suite quality score in run_eval.py is a random variable
with a plausible-looking mean, and every decision made on it -- ship, don't
ship, that prompt edit helped -- inherits the error.

The fix is unglamorous and takes about twenty minutes: score a sample by
hand, have the judge score the same sample, and look at whether you agree.

    python gameday_agent/eval/align.py --sample 8     # score by hand
    python gameday_agent/eval/align.py --report       # judge vs you

WHAT TO DO WITH THE ANSWER. Three outcomes, and only one of them means the
judge was wrong:

  You agree            use the score, and re-check after any rubric or
                       judge-model change.

  You disagree, and
  on reflection the
  judge was right      your standard was vague. Sharpen the rubric anchor
                       -- this is the common case and the useful one.

  You disagree, and
  you were right       the rubric says something other than what you meant.
                       Rewrite the criterion, do not argue with the judge.

The last two are both rubric bugs. A judge that reads the rubric correctly
and scores differently from you is telling you the rubric does not say what
you think it says, which is worth finding out before it grades a hundred
runs.

Scores live in human_scores.json and accumulate. Eight is enough to catch a
judge that is badly miscalibrated. It is not enough to certify one, and the
report below says so rather than printing a confident number over a sample
of eight.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gameday_agent.eval.rubric import CRITERIA, BY_KEY  # noqa: E402
from gameday_agent.eval.scoring import RUNS_DIR  # noqa: E402

HUMAN = Path(__file__).resolve().parent / "human_scores.json"

# Below this many scored items the report refuses to draw a conclusion.
# An agreement rate over three samples is not a measurement.
MIN_USEFUL_SAMPLE = 5


def _load_human() -> dict:
    if not HUMAN.exists():
        return {}
    return json.loads(HUMAN.read_text(encoding="utf-8"))


def _save_human(data: dict) -> None:
    HUMAN.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _all_judged_results() -> list[dict]:
    """Every scenario result across every saved run that the judge scored.

    Keyed by run file plus scenario so the same scenario from two different
    runs is two separate items -- which is the point. The same report scored
    twice would inflate agreement without adding information.
    """
    items = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for result in payload.get("results", []):
            if not result.get("judge_scores"):
                continue
            items.append({
                "id": f"{path.stem}:{result['scenario']}",
                "scenario": result["scenario"],
                "desc": result.get("description", ""),
                "report": result.get("report", ""),
                "judge_scores": result["judge_scores"],
                "judge_reasons": result.get("judge_reasons", {}),
            })
    return items


# ------------------------------------------------------------------- scoring

def _prompt_score(criterion) -> int | None:
    """Ask for one 1-5. Enter skips, q aborts the whole session."""
    print(f"\n  {criterion.name}")
    print(f"    {criterion.question}")
    print(f"    1 = {criterion.anchor_1}")
    print(f"    5 = {criterion.anchor_5}")
    while True:
        raw = input("    your score 1-5 (enter to skip, q to quit): ").strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        if raw == "":
            return None
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("    1 to 5 please.")


def score_by_hand(n: int) -> None:
    human = _load_human()
    pool = [i for i in _all_judged_results() if i["id"] not in human]
    if not pool:
        print("Nothing left to score -- every saved judged run already has "
              "your scores. Run run_eval.py again to produce more.")
        return

    # Random, not the newest N. Scoring the most recent runs first biases the
    # sample toward whatever the agent was doing this afternoon, and the
    # judge's weak spots may live in the scenarios you stopped looking at.
    random.shuffle(pool)
    batch = pool[:n]

    print(f"{len(batch)} report(s) to score. The judge's scores are hidden "
          f"until you are done -- seeing them first would anchor you, and an "
          f"anchored human is not an independent measurement.\n")

    try:
        for idx, item in enumerate(batch, 1):
            print("=" * 68)
            print(f"[{idx}/{len(batch)}]  scenario: {item['scenario']}")
            print("=" * 68)
            print(item["report"] or "(empty report)")
            scores = {}
            for c in CRITERIA:
                got = _prompt_score(c)
                if got is not None:
                    scores[c.key] = got
            human[item["id"]] = {"scenario": item["scenario"], "scores": scores}
            _save_human(human)          # after each, so a quit keeps the work
            print(f"\n  saved ({len(human)} scored in total)\n")
    except (KeyboardInterrupt, EOFError):
        print("\n\nStopped. Everything scored so far is saved.")

    print(f"Now run:  python gameday_agent/eval/align.py --report")


# -------------------------------------------------------------------- report

def import_sheet() -> int:
    """Read eval/my-scores.txt, the file make_packet.py leaves for you.

    The interactive prompt is still there, but typing into a text file beats
    it for the one job that matters: you can do it after reading the PDF,
    away from the terminal, and correct a line you changed your mind about.

    Lines look like `20260921-142531:replay_2021w8_1: 4 5 3 2 5`. Anything
    still holding underscores is treated as not yet scored rather than as an
    error, so a half-finished sheet imports cleanly and can be finished later.
    """
    sheet = Path(__file__).resolve().parent / "my-scores.txt"
    if not sheet.exists():
        print(f"No score sheet at {sheet}.\n"
              f"Run: python gameday_agent/eval/make_packet.py")
        return 1

    judged = {i["id"]: i for i in _all_judged_results()}
    human = _load_human()
    keys = [c.key for c in CRITERIA]
    added, skipped, bad = 0, 0, []

    for lineno, raw in enumerate(sheet.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # The id itself contains a colon, so split from the RIGHT.
        item_id, _, rest = line.rpartition(":")
        item_id, rest = item_id.strip(), rest.strip()
        if not item_id:
            bad.append(f"line {lineno}: no report id")
            continue
        if "_" in rest or not rest:
            skipped += 1
            continue
        parts = rest.split()
        if len(parts) != len(keys):
            bad.append(f"line {lineno}: expected {len(keys)} numbers, got {len(parts)}")
            continue
        if not all(p.isdigit() and 1 <= int(p) <= 5 for p in parts):
            bad.append(f"line {lineno}: scores must be whole numbers 1-5 ({rest})")
            continue
        if item_id not in judged:
            bad.append(f"line {lineno}: no judged run called {item_id}")
            continue
        human[item_id] = {
            "scenario": judged[item_id]["scenario"],
            "scores": {k: int(p) for k, p in zip(keys, parts)},
        }
        added += 1

    if bad:
        print("Problems, nothing imported from those lines:")
        for b in bad:
            print(f"  {b}")
    if added:
        _save_human(human)
    print(f"\nimported {added}, still blank {skipped}, total scored {len(human)}")
    if added:
        print("\n" + "-" * 68)
        return report()
    return 1 if bad else 0


def report() -> int:
    human = _load_human()
    if not human:
        print("No human scores yet. Run with --sample 8 first.")
        return 1

    judged = {i["id"]: i for i in _all_judged_results()}

    # criterion -> list of (human, judge)
    pairs: dict[str, list[tuple[int, int]]] = {c.key: [] for c in CRITERIA}
    for item_id, entry in human.items():
        item = judged.get(item_id)
        if not item:
            continue
        for key, mine in entry["scores"].items():
            theirs = item["judge_scores"].get(key)
            if theirs is not None and key in pairs:
                pairs[key].append((mine, theirs))

    total = sum(len(v) for v in pairs.values())
    if not total:
        print("Your scores do not line up with any judged run on disk.")
        return 1

    print(f"\n{'criterion':<24} {'n':>3}  {'exact':>6} {'within1':>8} "
          f"{'bias':>6}   judge is...")
    print("-" * 74)

    flagged = []
    for c in CRITERIA:
        vals = pairs[c.key]
        if not vals:
            print(f"{c.key:<24} {0:>3}       --       --     --   not scored")
            continue
        n = len(vals)
        exact = sum(1 for h, j in vals if h == j) / n
        within = sum(1 for h, j in vals if abs(h - j) <= 1) / n
        # Signed, judge minus human. Positive means the judge is the softer
        # grader. A consistent bias is easy to live with and easy to fix in
        # the anchors; scatter with no bias is the harder problem, because
        # there is nothing to correct -- the criterion is just ambiguous.
        bias = statistics.mean(j - h for h, j in vals)

        verdict = "aligned"
        if within < 0.7:
            verdict = "NOT ALIGNED -- do not trust this criterion"
            flagged.append(c.key)
        elif abs(bias) >= 1.0:
            verdict = f"{'lenient' if bias > 0 else 'harsh'} by ~{abs(bias):.1f}"
            flagged.append(c.key)

        print(f"{c.key:<24} {n:>3}  {exact:>5.0%} {within:>8.0%} "
              f"{bias:>+6.1f}   {verdict}")

    print("-" * 74)
    items = len(set(human) & set(judged))
    print(f"{items} report(s) scored by both.")

    if items < MIN_USEFUL_SAMPLE:
        print(f"\nThat is too few to conclude anything. Score at least "
              f"{MIN_USEFUL_SAMPLE} before treating the judge as validated, "
              f"and remember that 'aligned' on a small sample mostly means "
              f"'not yet caught being wrong'.")
        return 0

    if flagged:
        print(f"\nRubric work needed on: {', '.join(flagged)}")
        print("Read the judge's reasons for those on the runs you scored. If "
              "its reasoning is sound and you still disagree, the anchor text "
              "is what needs rewriting -- not the judge.")
    else:
        print("\nJudge tracks you on every criterion. The quality scores in "
              "run_eval.py can carry weight. Re-run this after changing the "
              "rubric, the judge model, or the agent's instruction.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Validate the LLM judge against your own scoring")
    p.add_argument("--sample", type=int, metavar="N",
                   help="score N saved reports by hand")
    p.add_argument("--report", action="store_true",
                   help="compare your scores against the judge's")
    p.add_argument("--import", dest="do_import", action="store_true",
                   help="read eval/my-scores.txt, then show the comparison")
    a = p.parse_args()

    if a.do_import:
        sys.exit(import_sheet())
    elif a.sample:
        score_by_hand(a.sample)
    elif a.report:
        sys.exit(report())
    else:
        p.print_help()
