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

## Known limits

- The questions were written by Claude, not real users, so the feedback
  table (Phase 3) is the next source.
- Latency was measured with 6 questions in parallel on a laptop, in-process,
  with no MCP network hop (the hop is the same for both designs).
- One judge row came back with no score and is excluded.
