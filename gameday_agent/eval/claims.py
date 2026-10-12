"""Measure news.check_claims on claim/quote pairs with known labels.

seeded.py found the judge cannot catch an extractor claim that inverts its
own quote once the report repeats it: the report matches the tool result,
so it looks grounded. The fix lives upstream, in news.check_claims. This
file is how anyone knows that fix works.

Each pair is a quote plus a claim written from it, labeled by hand with the
answer the checker owes. The labels need no judgment call -- the claims are
written to be plainly faithful, plainly inverted, or plainly overreaching.
The two numbers that matter, kept apart on purpose:

  wrong claims replaced    a bad summary that reaches the agent is the
                           failure this exists to stop
  good claims kept         a checker that replaces everything is safe and
                           useless -- the agent loses every summary

Every pair runs --repeats times (default 3) as ONE batched call, the way
get_team_news sends them, and is reported as k/n.

    python gameday_agent/eval/claims.py
    python gameday_agent/eval/claims.py --repeats 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "advisor-agent-2026")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

from gameday_agent.eval.seeded import NEWS_CLAIM_INVERTED, NEWS_CLAIM_TRUE, NEWS_QUOTE  # noqa: E402
from gameday_agent.news import CLAIM_CHECK_MODEL, check_claims  # noqa: E402

# (id, quote, claim, should_keep). Fictional players, like every fixture.
PAIRS = [
    # The seeded.py case, verbatim.
    ("ashby_true", NEWS_QUOTE, NEWS_CLAIM_TRUE, True),
    ("ashby_inverted", NEWS_QUOTE, NEWS_CLAIM_INVERTED, False),

    ("practice_true",
     "Okafor was a full participant in Thursday's practice and carries no designation.",
     "Okafor practiced in full Thursday and has no injury designation.", True),
    ("practice_upgraded",
     "Okafor was limited in Thursday's practice with an ankle issue.",
     "Okafor was a full participant in Thursday's practice.", False),

    ("status_true",
     "The team ruled out tight end Dell Marsh for Sunday's game with a knee sprain.",
     "Dell Marsh has been ruled out for Sunday.", True),
    ("status_softened",
     "The team ruled out tight end Dell Marsh for Sunday's game with a knee sprain.",
     "Dell Marsh is questionable for Sunday with a knee sprain.", False),

    ("depth_true",
     "Coach Rivera said rookie Jalen Pruitt will start at running back with Hollis sidelined.",
     "Jalen Pruitt will start at running back.", True),
    ("depth_swapped",
     "Coach Rivera said rookie Jalen Pruitt will start at running back with Hollis sidelined.",
     "Hollis will start at running back over Jalen Pruitt.", False),

    # Overreach: the quote says less than the claim.
    ("overreach_number",
     "Tate is expected to see an expanded role in the passing game this week.",
     "Tate is expected to see 10 or more targets this week.", False),
    ("overreach_certainty",
     "Coach said Brennan is trending toward playing but will be a game-time decision.",
     "Brennan will play on Sunday.", False),

    ("transaction_true",
     "The Steelers placed wide receiver Kofi Mensah on injured reserve Saturday.",
     "Kofi Mensah was placed on injured reserve.", True),
    ("transaction_wrong",
     "The Steelers signed wide receiver Kofi Mensah off the practice squad Saturday.",
     "Kofi Mensah was released by the Steelers.", False),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    claims = [{"player": "x", "claim": c, "quote": q} for _, q, c, _ in PAIRS]
    correct = {pid: 0 for pid, *_ in PAIRS}
    verdicts = {pid: [] for pid, *_ in PAIRS}
    for _ in range(args.repeats):
        checked, _ = check_claims([dict(c) for c in claims])
        for (pid, q, c, keep), out in zip(PAIRS, checked):
            verdicts[pid].append(out["claim_check"])
            kept = out["claim"] == c
            correct[pid] += kept == keep

    n = args.repeats
    print(f"claim check {CLAIM_CHECK_MODEL}, {n} repeats\n")
    print(f"{'pair':<22} {'owed':<8} {'verdicts':<40} right")
    for pid, _, _, keep in PAIRS:
        print(f"{pid:<22} {'keep' if keep else 'replace':<8} "
              f"{','.join(verdicts[pid]):<40} {correct[pid]}/{n}")
    good = [p for p in PAIRS if p[3]]
    bad = [p for p in PAIRS if not p[3]]
    print(f"\nwrong claims replaced  {sum(correct[p[0]] for p in bad)}/{len(bad) * n}")
    print(f"good claims kept       {sum(correct[p[0]] for p in good)}/{len(good) * n}")


if __name__ == "__main__":
    main()
