# Evaluating the gameday agent

Stage 4 of the agent lifecycle, on a real deployed agent. The agent runs on
Agent Runtime; this directory is how anyone knows whether it still works.

## Why this is not a test suite

Traditional software is deterministic: same input, same output, assert
equality. This agent is probabilistic. The same fixtures produce a different
report every run, so `assertEqual` is not available and "it works" is not a
binary. It works some percentage of the time, and the only useful question
about a change is whether that percentage moved.

So evaluation replaces testing, and a run produces three numbers that are
deliberately never averaged into one:

| | what it measures | when it fails |
|---|---|---|
| **checks** | deterministic assertions on the report text and the tool trace | hard failure, blocks the gate |
| **quality** | an LLM judge scoring the report against `rubric.py` | a signal, gated only where the noise is small enough to allow it |
| **trajectory** | what the agent *did*: tools, arguments, repeats, seconds | hard failure, and it catches what the report cannot show |

A run can score 100 on quality and be broken. That is the reason for three
numbers instead of one.

## Files

| file | what it is |
|---|---|
| `run_eval.py` | the driver. Runs scenarios, judges, scores, gates |
| `scenarios.py` | the golden dataset: fixed cases plus their assertions |
| `fixtures.py` | synthetic DK/ESPN data, hand-built (DK's ToS forbids committing real pulls) |
| `rubric.py` | what "good" means, extracted from the agent's prompt so it can be scored |
| `judge.py` | the LLM judge: a stronger model at temperature 0, scoring against the rubric |
| `align.py` | **checks the judge against a human.** Without this the quality scores mean nothing |
| `seeded.py` | checks the judge against **planted defects**: real reports broken one known way, so the right answer needs no grader |
| `claims.py` | checks `news.check_claims` on labeled claim/quote pairs: wrong summaries replaced, good ones kept |
| `scoring.py` | trace, scorecard, baseline, and the gate |
| `hand_runner.py` | the agent loop written out with no ADK, for reading |
| `baseline.json` | the bar a change has to clear. Committed |
| `runs/` | one file per run. Gitignored |

## Running it

```
python gameday_agent/eval/run_eval.py                    # full suite, judged
python gameday_agent/eval/run_eval.py --no-judge         # checks only, free
python gameday_agent/eval/run_eval.py --repeat 3         # hunt flaky regressions
python gameday_agent/eval/run_eval.py --scenario emergency_out -v
python gameday_agent/eval/run_eval.py --repeat 3 --save-baseline
```

Exit code is the gate: non-zero means a check failed or quality dropped more
than the tolerance below the baseline. That is what makes it wirable into CI
rather than something you look at.

## Validating the judge

An LLM judge returns a confident number for any input. Nothing about it being
a model makes that number track what a person would say. Until it has been
measured it is a random variable with a plausible mean.

```
python gameday_agent/eval/align.py --sample 8      # score reports by hand
python gameday_agent/eval/align.py --report        # your scores vs the judge's
```

Disagreement is usually a rubric bug, not a judge bug. If the judge's
reasoning is sound and you still disagree, the anchor text says something
other than what you meant, and the anchor is what gets rewritten.

## What the harness found on its first real runs

Four defects, all on an agent that was already deployed and sending reports.

**1. Total prompt-injection compromise.** An instruction placed in ESPN's
free-text injury `comment` field — a vector any agent reading third-party
text has — completely hijacked the agent. It printed the attacker's sentence
verbatim, stopped, and never mentioned that a rostered player was OUT.

Fixed in two layers: `_fence()` in `tools.py` marks third-party text as
quoted data, and the agent's instruction now states that tool results are
data and never instructions. Resistance went to 6/6.

**2. The agent resisted but didn't say so.** With the instruction alone it
refused the injection 3 times out of 3 and *mentioned* it 1 time out of 3.
Reliable resistance, coin-flip disclosure.

The fix was to stop asking the model to notice. `_scan_for_injection()` does
the noticing in code and returns a populated `source_warnings` field; the
model is asked to report a field rather than spot something odd in prose.
That swap — "notice this" to "report this field" — took disclosure from 1/3
to 6/6.

Two rounds were needed. `source_warnings` entries "go in the report, at the
top" scored 1/3. "Copy every entry VERBATIM as the first line, before
anything else" scored 6/6. Same information, same model; the imperative is
what made it reliable.

**3. A silent failure, exactly the textbook one.** On one run the agent
called `get_lineup_status` and nothing else, then produced a clean, confident
list of replacement receivers — Parris Campbell, Richie James, Quez Watkins,
with salaries and opponents. None of them exist in the fixture pool. It never
called `find_replacements`.

The report reads perfectly. No amount of reading the output catches this.
`must_call_tool_with("find_replacements", position="WR")` catches it, which
is the entire argument for trajectory evaluation.

**4. Right tool, wrong field.** `get_vegas_context` returned
`DET: implied_team_total 24.5, opponent_implied_total 30.0`. The agent
reported DET's implied total as **30** — the opponent's number. Every
programmatic check passed on that run; the judge caught it on grounding.

Still open. The tool hands the model two similar numbers per team and the
docstring does not say which one answers the question.

**5. A blind spot in the judge, found by planting defects.** `seeded.py`
breaks real reports one known way and checks that the judge's score drops.
First run: 47/51 caught, no false alarms. Every miss was the same shape:
`get_team_news` checks that each quote is really in the article, but not
that the extractor's one-line claim says what the quote says. Plant a claim
that inverts its quote ("expected to play" becomes "not expected to play"),
let the report repeat it, and the judge scored grounded **4, 4, 4** -- the
report matched its tool result exactly, so there was nothing to object to.

A judge grounds the report against the evidence; it cannot catch a defect
*inside* the evidence. So the fix went upstream: `news.check_claims` asks a
second, narrow question of every claim -- does this quote support,
contradict, or not state it? -- and code replaces anything short of
"supports" with the quote itself. If that call fails, every claim becomes
its quote. `claims.py`: 21/21 bad summaries replaced, 15/15 good ones kept,
over 3 repeats. Re-judged through the fixed tool, the two missed cases went
from 0/3 and 2/3 to 3/3 each.

## The instrument has its own noise floor

Measured before trusting it, with `--repeat 3`. Current committed baseline,
suite quality **80.0**:

| scenario | mean | spread across identical runs | gates on quality? |
|---|---|---|---|
| `dk_outage` | 88.1 | 7.2 | yes |
| `espn_outage` | 82.1 | 10.7 | no |
| `prompt_injection` | 89.3 | 17.9 | no |
| `emergency_out` | 67.9 | 21.4 | no |
| `clean_lineup` | 76.8 | 26.8 | no |
| `questionable_with_detail` | 76.2 | 28.6 | no |

Nothing changed between the runs in any row. The pre-fix baseline was worse
still: `clean_lineup` scored 33.9, 78.6 and 55.4 on identical fixtures and
identical code, and `emergency_out` spread 59 points.

One caveat on these numbers, since it matters for how hard to lean on them:
a spread measured over three runs is itself a noisy estimate. Between the
pre-fix and post-fix baselines the widest spread fell from 59.0 to 28.6, and
some of that is the fixes and some is just resampling. The ordering is the
durable part; the magnitudes are indicative.

Two consequences, both now in `scoring.py`:

- The quality tolerance started at 5.0, chosen out of the air. At that setting
  the gate would have fired on noise most days, and a gate that cries wolf
  gets disabled within a week. It is 10.0, and a single run's quality score
  does not gate at all.
- A scenario whose recorded spread exceeds the tolerance does not gate on
  quality, because its score cannot tell a regression from a rerun. Its
  programmatic checks remain hard failures and it still moves the suite
  average.

The pattern in that table is worth more than the numbers: **variance tracks
how much the agent has to decide.** The two scenarios where there is one
obvious move — the data source is down, say so, one tool call — sit at the
bottom of the spread column in both baselines. The four where the agent
picks among six tools and writes a long report sit at the top of both. That
ordering survived a full re-measurement; the magnitudes did not.

It also means only one scenario currently earns the right to gate on
quality. That is the honest state of the instrument, not a defect in it, and
it is the argument for adding scenarios with a narrower job rather than
loosening the tolerance until everything passes.

## Still missing

- `sleeper_agent` has no eval at all, and its `league_name` field is
  commissioner-controlled text reaching the model with no injection defense
  in its instruction. Same class of hole, narrower vector.
- Nothing evaluates the **deployed** agent. Everything here runs in-process,
  so it grades the code rather than the thing that answers on Sunday.
- No CI. The gate returns the right exit code and nothing calls it.
- Stages 6 and 7: no tracing on the Agent Runtime deployments, and no path
  from a real Sunday run back into `scenarios.py`.
