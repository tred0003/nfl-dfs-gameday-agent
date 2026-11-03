# ask_eval: is Ask better as an agent or as a pipeline?

Run 2026-10-28. Phase 2 of the archive Search/Ask product plan.

## The question

Ask has two designs, both production code in `webapp/ask.py`:

- **agent** (`run`, the original): the model reads the question and decides
  which archive tools to call, at runtime. This is *lazy* context.
- **pipeline** (`run_pipeline`): code always fetches the player's history and
  runs one search, in parallel. Then the model is called once, with no tools.
  This is *eager* context.

Same model (Gemini 2.5 Flash, temperature 0.2), same answer rules, same
`[P#]` citations, same deterministic check (`webapp/citations.py`). The only
thing that differs is who decides what to fetch.

## Setup

- **Questions:** 150 from the retrieval test set (`embed_tune/`) plus 20 the
  archive cannot answer (`unanswerable.jsonl`: fictional players, false
  premises checked against the snapshot, after the cutoff, not NFL). Each
  was asked twice per design: 680 answers, 0 errors.
- **Frozen archive:** the archive's clock is frozen at the test set's cutoff
  (2026-10-26). Every query filters `published < now`, so both designs see
  exactly the archive the answer key was built on.

```
python ask_eval/run.py                  # answers.jsonl (~$2)
python ask_eval/grade.py contract       # free
python ask_eval/grade.py grounding      # check-grounding API, 1 call/answer
python ask_eval/grade.py judge          # Vertex AI Gen AI evaluation service
python ask_eval/grade.py report         # results.json
```

## Results (results.json)

| | agent | pipeline |
|---|---|---|
| **Correct, 0-2** (Vertex autorater vs the answer key) | 1.08 | **1.41** |
| Fully correct (score 2) | 39% | **53%** |
| Refuses the 20 unanswerable cleanly, 0-2 | **1.95** | 1.85 |
| Grounding support, mean (check-grounding API) | 0.75 | 0.76 |
| Answers with support < 0.5 | 13% | **9%** |
| Citation agreement (model's `[P#]` vs the API's own citations) | 98.8% | 98.0% |
| Same answer type on both runs | 86% | **96.5%** |
| Latency p50 / p95, all answers | 4.2 / 8.3 s | 4.5 / 10.4 s |
| Latency p50 / p95, answers that retrieved | 4.4 / 9.1 s | 4.5 / 10.4 s |
| Cost per answer (model + retrieval) | **$0.0024** | $0.0036 |
| Answerable questions answered with no retrieval | **47 of 300** | 0 |

**Paired difference (pipeline minus agent):** +0.33 on the 0-2 scale, 95%
bootstrap CI +0.23 to +0.43, over 299 pairs. The pipeline scored higher on
110 questions and the agent on 40.

### Why the agent lost

1. **It skipped the archive.** 61 of 340 agent answers made no tool call. The
   instruction says "the archive is a day or more behind live news", and the
   model read that as permission to refuse status questions without looking.
   Example: "Will Rico Dowdle play Thursday?" got "I cannot answer… the
   archive is behind." On those answers it scored 0.21.
2. **Even when it searched, it lost by +0.22** (CI +0.12 to +0.33). It picked
   one tool (search 201 times, history 79), almost never both, and made a
   second call once in 340 answers. The flexibility it was given went
   unused. The pipeline always packs both, and the history (the newest notes
   on the player) is often where the current status is.
3. **Its answers vary more.** The same question got the same answer type on
   both runs only 86% of the time, against 96.5% for the pipeline.

The agent does refuse impossible questions slightly more cleanly (1.95 vs
1.85). The pipeline's misses there are "partial" answers that add related
facts after "nothing found". None of the pipeline's refusals is invented.

### Decision

The **pipeline is production** (`ASK_DESIGN=pipeline`, the default). The
agent stays selectable for comparison. The extra cost is about $1.20 per
1,000 questions.

## How far to trust the grader

The autorater is the Gen AI evaluation service's default; this SDK can't
choose the model. A blind Claude grader scored a random 60 answers with the
same rubric (`calibration_*.json`):

- **Agreement:** exact on 60%, within one point on 93%, quadratic-weighted
  kappa 0.56 (moderate).
- **The autorater is stricter:** mean 1.22 against Claude's 1.58. In two of
  the four 0-vs-2 splits I read, it called a detail that was absent from the
  reference a "hallucination", which its rubric says not to do.
- **Both graders see the same gap:** pipeline over agent by +0.58
  (autorater) and +0.48 (Claude) on the sample.

So the **ranking of the designs is robust**. The **absolute correctness** is
a range: about 1.4-1.8 out of 2 for the pipeline, depending on the grader.

## What each grader can and cannot tell you

- **Contract** (`citations.py`) is free and deterministic, so it can gate CI.
  It catches made-up refs, uncited facts and wrong dates. It can't tell
  whether a passage *supports* the sentence that cites it.
- **Grounding** (check-grounding API) is strict entailment. It can't ground a
  claim of absence ("no news of an ankle injury" scores about 0), so it
  under-scores honest partial answers. Each passage goes in with its date as
  text, or every correctly dated claim would fail.
- **The judge** checks correctness against the key, the only grader that
  knows the newest status. Its reference is a sample of the relevant news,
  not all of it.

## Phase 2b: the pipeline plus podcasts (2026-10-29)

**The change:** the pipeline gets a third call in the same parallel fetch:
`search_podcasts`, limited to the ranker's best 3 passages, in slots of its own. News keeps all six
of its passages. One mixed search was tried first, on the search gate, and
cost news its newest-report hits (newest@6 -4.9). The prompt adds two rules:
a podcast is a host's view, to be named and dated, and never the only source
for injury status, a transaction or who is starting. If no podcast passage
fits the question, the answer leaves podcasts out.

**The test:** design `pipeline_pod`, the same 170 questions × 2 runs and the
same frozen archive, compared with the 2026-10-28 `pipeline` answers (news
only). `python ask_eval/run.py --designs pipeline_pod`, then the three
graders.

| | news only | + podcasts v1 | **+ podcasts v2** |
|---|---|---|---|
| Correct, 0-2 | 1.405 | 1.463 | **1.447** |
| Fully correct | 53.2% | 54.7% | **56.3%** |
| Refuses the 20 unanswerable, 0-2 | 1.85 | 1.75 | 1.70 |
| Grounding support, mean | 0.764 | 0.725 | 0.737 |
| Answers with support < 0.5 | 8.7% | 14.0% | **9.0%** |
| Contract: date mismatches / uncited | 24 / 14 | 40 / 17 | 26 / 18 |
| "N days ago" per answer | 0.78 | 0.17 | 0.71 |
| Same answer type on both runs | 96.5% | 98.8% | 98.2% |
| Latency p50 / p95 | 4.5 / 10.4 s | 5.3 / 12.1 s | 5.2 / 9.8 s |
| Cost per answer | $0.0036 | $0.0050 | $0.0047 |
| Answers citing a podcast (of 310 offered one) | — | 116 | 124 |
| Status facts with ONLY a podcast cited | — | 3 | 2 |

**Paired, v2 minus news only:** +0.04 on correctness (95% CI −0.03 to
+0.13, 299 pairs; better on 61, worse on 55). This is no measurable change.
That was expected: the key grades only news, so a useful host opinion can
earn nothing here, and it can be marked down as "not in the reference".
So the test checks that podcasts do no harm to news correctness. It does not
measure what the podcasts add.

**v1 to v2, one example sentence.** v1's podcast rule showed a host's take
dated "(Fri Oct 23)". The model copied that format for every fact, news included: "N days ago" fell from
0.78 to 0.17 per answer. Explicit weekdays then got attached to the wrong
event (game day vs. the article's publish day), and grounding fell: answers below 0.5
support went from 8.7% to 14%. Answers that cited a podcast grounded as well as news-only
answers did. The drop came from news sentences. v2's example says "3 days
ago", and grounding came back.

**The rule checks.** Of the 2 claims backed only by a podcast that state a
status (and 3 in v1), every one repeats a fact that the same answer had already cited
from news ("hosts also noted he is out for the season"). That is redundant,
not wrong. No answer took a status from a podcast that the news did not have.

**Refusals, 1.85 to 1.70:** this comes down to three questions out of 20. Both podcast
runs answered "Jets' first 2027 pick" from a *news* mock draft without
the "No archived news found" opener, and cited no podcast. This is wording
drift from the longer prompt. "Jefferson's fantasy points on Sunday" wobbles
in every design. On "Super Bowl winner", one answer mentioned a host
talking about last season's win after correctly saying that nothing was found:
the only podcast-caused miss.

**Decision:** ship v2 (`ASK_PODCASTS=1`, prompt `ask-2026-10-29-pod2`).
News correctness and grounding are unchanged, and the one rule that matters,
no status taken from a podcast alone, held. The cost is about $1.10 per
1,000 questions and 0.7 s at the median.

## Known limits

- The questions were written by Claude, not real users, so the feedback
  table (Phase 3) is the next source.
- Latency was measured with 6 questions in parallel on a laptop, in-process,
  with no MCP network hop (the hop is the same for both designs).
- One judge row came back with no score and is excluded.

> Not in this public copy: `answers.jsonl` and `calibration_sample.jsonl` quote licensed news text. Scores and summaries are in `results.json` and the `graded_*.jsonl` files.
