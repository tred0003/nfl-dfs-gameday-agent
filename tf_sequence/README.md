# tf_sequence — a TensorFlow sequence model, built to close one specific gap

## Why this exists

A PMLE practice question, missed:

> As a member of the AI team at an automobile company, you're enhancing a visual
> defect detection model using TensorFlow and Keras, aiming to integrate image
> augmentation techniques like translation, cropping, and contrast adjustment...
> **These functions are randomly applied to each training batch.** To optimize
> your data processing pipeline for **runtime and compute resource utilization**,
> what approach should you adopt?
>
> - **A. Dynamically embed the augmentation functions within the tf.data pipeline.** ← correct
> - B. Dynamically embed the augmentation functions as part of Keras generators.
> - C. Utilize Dataflow to generate all possible augmentations and store them as TFRecords.
> - D. Employ Dataflow to dynamically generate augmentations per training run and stage them as TFRecords. ← chosen

The answer given was D. The instinct behind it is not stupid — it is
"Dataflow is how Google does data processing at scale," which is true in
general and wrong here. The stem says the augmentations are **random per
batch**, and that single phrase decides the question before runtime is ever
mentioned.

## The rule

```
Transform once and reuse it        ->  offline.  Dataflow, TF Transform,
                                       build_sequences.py. Pay once.

Transform differently every epoch  ->  tf.data.map. Pay every epoch, on the
                                       CPU, next to the accelerator.
```

Augmentation is the second kind. Producing a different answer each time it sees
the same example *is the mechanism*. Precompute it and you freeze one draw and
train on that same draw forever — you pay for a distributed job and lose the
thing you were paying for.

B fails differently: a Keras generator runs in the Python interpreter on the
main process, so the accelerator that just finished batch N idles while Python
assembles batch N+1. `tf.data` ops are graph ops, and two arguments fix it:

```python
ds = (ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)   # across cores
        .batch(256)
        .prefetch(tf.data.AUTOTUNE))                         # overlap with the model step
```

## What got built

A GRU that reads a player's last 8 games of raw box score and predicts this
week's DraftKings points — run head to head against the XGBoost champion in
`src/train.py`, on identical folds and an identical metric.

| file | what it does |
|---|---|
| `common.py` | the column split, and the argument for why it is drawn where it is |
| `build_sequences.py` | parquet → sharded TFRecords. Deterministic, runs once |
| `pipeline.py` | the tf.data input pipeline and the three augmentations |
| `benchmark.py` | **the lab.** Measures what each tf.data option is worth |
| `model.py` | two-branch Keras model: GRU over games, dense over context |
| `train_tf.py` | walk-forward training, graded against XGBoost and the baseline |

### The decision that makes the comparison honest

`model_table_v3.parquet` already contains `dkp_r3`, `dkp_r5`, `dkp_r8`,
`dkp_ewm`, `tgt_share_r5` and three dozen more hand-built rolling features.
Someone already looked back over the last N games and reduced them to an
average — **that reduction is the temporal modeling**, done by hand, by a human
who chose 3, 5 and 8 as the windows.

The whole pitch of a sequence model is: stop choosing that by hand.

So the network is **not given any of them**. It gets the raw game log and has
to rediscover whatever summary it wants. XGBoost keeps all of them. If the
network wins, it found a better temporal summary than the hand-built one. If it
loses, the hand-built windows were good — which is a finding, not a failure.

### The augmentations, and why they are the same idea as the image question

| this project | the exam's image scenario |
|---|---|
| `random_window_crop` — forget the oldest 0–2 games | random crop |
| `random_game_dropout` — hide 0–2 games from the window | cutout |
| `random_stat_jitter` — multiply stats by ~N(1, 0.05) | contrast / brightness |

Each is random **per example, per epoch**. Each is therefore impossible to
precompute without destroying it. Game dropout in particular is not a gimmick:
in production the game log really is ragged — byes, injuries, inactives — and a
network that leans on "the game exactly 3 weeks ago" breaks the first time that
slot is empty.

## Results

### The tf.data lab

```
seasons 2016-2024  batch 256  45,765 records
model step: matmul 1024x1024 per batch

  1 naive                     0.57s/epoch     79,755 ex/s
  2 +parallel map             0.56s/epoch     81,852 ex/s
  3 +prefetch                 0.56s/epoch     81,778 ex/s
  4 +cache (recommended)      0.52s/epoch     88,322 ex/s
  5 map before batch          0.53s/epoch     85,963 ex/s
  6 precomputed (answer D)    0.48s/epoch     95,636 ex/s   <- fastest, and wrong
```

**Read that honestly: on this data the tuned pipeline is only 1.11x the naive
one, and answer D is the fastest row in the table.** That is the real result and
it is not the one the exam question implies, because this dataset is tiny —
8×19 floats per record, ~30 MB total, and the augmentation is a handful of
elementwise ops. There is almost nothing for AUTOTUNE to parallelise.

The lesson lives in the sweep instead. `--sweep` adds clearly-labelled synthetic
per-example cost, standing in for what decoding a JPEG and applying three image
transforms actually costs:

```
   extra ops     naive     tuned   speedup
           0     0.59s     0.53s     1.11x
          20     0.73s     0.56s     1.31x
          80     1.74s     0.71s     2.46x
         240     4.50s     1.01s     4.48x
```

**That relationship is the thing to carry into the exam: the value of a
parallel, prefetched map scales with how expensive the map is.** An image
pipeline lives at the bottom of that table. A pipeline over 19 floats lives at
the top. Quoting "AUTOTUNE gives you a 4x speedup" as a fact would be repeating
a number that depends entirely on a variable nobody mentioned.

And speed was never why D is wrong:

```
distinct versions of the same 4 batches, over 4 epochs
  slice: season 2024, mean history depth 7.23/8 games
    in-pipeline map (A)          4 distinct  -> fresh every epoch
    precomputed + cached (D)     1 distinct  -> FROZEN after the first draw
```

D is fastest *because* it stopped doing the work. That is the whole question.

### The model

Walk-forward, 2021–2024, never trained on. Both models on `model_table_v3`,
same folds, same metric:

| model | n | slate Spearman | RMSE |
|---|---|---|---|
| **XGBoost (mean)** — 83 engineered features | 21,099 | **0.6423** | 6.619 |
| TF sequence (GRU) — raw game logs, no rolling features | 21,099 | 0.6317 | 6.662 |
| baseline: rolling avg | 16,762 | 0.5809 | 7.312 |

Per fold: 0.6164 / 0.6119 / 0.6455 / 0.6531.

**XGBoost wins by 0.011, and it wins at every position** (QB .572 vs .558,
RB .637 vs .626, WR .589 vs .583, TE .536 vs .519). A consistent gap across
four independent buckets is a real gap, not fold noise.

Run-to-run spread on the GRU is about ±0.0003 on the headline number (tf.data
parallelism is not bit-deterministic even with a fixed seed), so the 0.011 gap
is roughly thirty times the noise.

That is the expected result and the honest one. Gradient boosting is very hard
to beat on tabular data at this sample size — 41k training rows is small for a
network, and XGBoost gets native missing-value handling for free while the
network has to learn around 0-filled columns and their flags.

What is genuinely interesting is how close it is. The GRU came within 0.011 of
a model carrying 83 hand-built features — including `dkp_blended`, which is
**26% of XGBoost's total feature importance by itself** — while being given
none of them. It rediscovered most of that signal from raw game logs. The
hand-built rolling windows turn out to be very good, and worth their 0.011.

#### The mistake this table nearly shipped

The first run of `train_tf.py` reported XGBoost at **0.5968** and declared the
GRU the winner by 0.035.

That number came from `models/metrics.json`, which was generated by
`src/train.py` at its default `--table model_table` — a *different, smaller
feature set* than the `model_table_v3` the network was trained on. The file
records no table name, so nothing about reading it looked wrong.

Re-running `python src/train.py --table model_table_v3 --no-save` put XGBoost
at 0.6423 and reversed the result.

`train_tf.py` no longer reads that file. It takes `--xgb-spearman` explicitly
or prints no comparison row at all, because a comparison whose provenance you
cannot state is worse than no comparison. **Two models are only comparable
when they were fed the same table, and "same metric, same folds" does not
imply it.**

## Running it

```bash
# one-time: build the TFRecords (deterministic, ~1 min)
.venv-tf/Scripts/python tf_sequence/build_sequences.py

# the lab
.venv-tf/Scripts/python tf_sequence/benchmark.py
.venv-tf/Scripts/python tf_sequence/benchmark.py --sweep 0 20 80 240

# XGBoost's number on the SAME table -- this is the bar (~5 min, ffml env)
python src/train.py --table model_table_v3 --no-save

# walk-forward training (~2.5 min), with that number passed in explicitly
.venv-tf/Scripts/python tf_sequence/train_tf.py \
    --xgb-spearman 0.6423 --xgb-rmse 6.619
```

TensorFlow lives in `.venv-tf`, deliberately separate from the `ffml` env that
runs the XGBoost champion. TF pins numpy and pandas hard; the champion runs on
pandas 3.x. One shared environment means one upgrade breaks the model that
actually ships.

## On TFX, and why it is not here

TFX does not install on Windows, at any version:

```
$ pip install tfx==1.21.0
ERROR: No matching distribution found for ml-metadata<1.22.0,>=1.21.0
       (from versions: 0.12.0.dev0, 0.13.0.dev0, 0.13.1.dev0)
```

Not a Python version problem — TFX 1.21 supports 3.10 through 3.13, and this
was reproduced on both 3.11 and 3.13. `ml-metadata`, the component that backs
TFX's artifact store, ships Linux and macOS wheels only.

That is fine, and arguably clarifying. What the exam actually tests about TFX is
the component vocabulary — ExampleGen → StatisticsGen → SchemaGen →
ExampleValidator → Transform → Trainer → Evaluator → Pusher — and the idea
behind TF Transform: that preprocessing belongs **inside the exported model**,
so serving cannot drift from training.

The second idea is already implemented here without TFX. `model.py` adapts its
`Normalization` layers on training data and saves them as part of the model, so
a SavedModel takes raw feature values and normalises them itself. The classic
way to ship a broken model is to normalise in a training script with one set of
means and then write separate serving code that computes them slightly
differently; the model is fed a distribution it never trained on and quietly
degrades, with no error anywhere. Keeping the constants inside the graph makes
that impossible rather than unlikely. TF Transform is the same idea at pipeline
scale.

The remaining TFX surface belongs on Vertex AI, which is the next step.

## Next

1. **Vertex AI custom training job.** `deploy/` already has the pattern from the
   XGBoost model; `train_tf.py` already reads `AIP_MODEL_DIR`-style arguments in
   `src/train.py`'s style and can follow it.
2. **One TFX pipeline run on Vertex AI Pipelines**, on Linux, where it installs.
   Enough to have actually held the components, not just read about them.
3. **Ablate the augmentation.** Everything here assumes it helps. Train with
   `aug_kwargs={}` and find out — on 41k rows it may well be regularising a
   model that was not overfitting in the first place.
