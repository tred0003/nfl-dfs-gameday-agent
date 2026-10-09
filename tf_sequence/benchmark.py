"""Measure what each tf.data option is actually worth, on your own hardware.

This file exists because of one missed exam question:

    "Augmentation functions are randomly applied to each training batch. To
     optimize your data processing pipeline for runtime and compute resource
     utilization, what approach should you adopt?"

    A. Embed the augmentation functions within the tf.data pipeline.  <- correct
    B. Embed them as part of Keras generators.
    C. Use Dataflow to generate all possible augmentations as TFRecords.
    D. Use Dataflow to generate augmentations per training run as TFRecords.

Knowing A is the answer is worth one mark. Being able to produce the numbers
underneath it is worth considerably more, and it is the difference between
reciting a rule and having used it.

WHAT IS BEING MEASURED
----------------------
Throughput -- examples per second -- through a pipeline feeding a downstream
step that costs real time. That downstream step is not decoration. prefetch's
entire mechanism is overlapping input preparation with model computation, so a
benchmark with no model step measures a pipeline with nothing to overlap
against and concludes, wrongly, that prefetch does nothing. `--step-cost`
controls the size of a matmul that stands in for the model's forward and
backward pass.

TWO HONEST CAVEATS, STATED BEFORE THE NUMBERS
---------------------------------------------
1. THIS DATASET IS SMALL. 51k records of 8x19 floats is ~30 MB; it fits in RAM
   with room to spare. Real image pipelines decode JPEGs -- hundreds of
   microseconds of CPU per example -- and that is where AUTOTUNE earns its
   reputation. Expect the effect here to be real but more modest than the exam
   scenario implies. `--aug-cost` adds clearly-labelled synthetic work so you
   can watch the curve steepen as per-example cost rises, which is the actual
   lesson: the benefit of parallel map scales with how expensive the map is.

2. NO GPU. TensorFlow dropped native Windows GPU support after 2.10, so this
   runs on CPU, which means the input pipeline and the "model" are competing
   for the same cores. On a real accelerator the gap between naive and tuned
   is wider, because there the CPU idling is a GPU idling.

Run:  .venv-tf/Scripts/python tf_sequence/benchmark.py
      .venv-tf/Scripts/python tf_sequence/benchmark.py --aug-cost 40
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from common import TFR
from pipeline import load_spec, make_augment, make_dataset

# Each config is the previous one plus one change, so every row of the output
# isolates the value of exactly one option.
CONFIGS = {
    "1 naive": dict(
        parallel=False, prefetch=False, cache=False, map_after_batch=True,
        note="map with no num_parallel_calls, no prefetch, no cache"),
    "2 +parallel map": dict(
        parallel=True, prefetch=False, cache=False, map_after_batch=True,
        note="num_parallel_calls=AUTOTUNE on parse and augment"),
    "3 +prefetch": dict(
        parallel=True, prefetch=True, cache=False, map_after_batch=True,
        note="overlap input prep with the model step"),
    "4 +cache (recommended)": dict(
        parallel=True, prefetch=True, cache=True, map_after_batch=True,
        note="cache the DETERMINISTIC parse; augment after, every epoch"),
    "5 map before batch": dict(
        parallel=True, prefetch=True, cache=True, map_after_batch=False,
        note="same options, per-example instead of vectorised"),
}


def make_step(cost: int):
    """Stand-in for a training step, so prefetch has something to overlap.

    A matmul of size `cost` compiled into a tf.function -- not the real model,
    just a consumer that takes a believable amount of time per batch.
    """
    w = tf.random.normal([cost, cost])

    @tf.function
    def step(features):
        x = tf.reshape(features["seq"], [tf.shape(features["seq"])[0], -1])
        x = tf.pad(x, [[0, 0], [0, tf.maximum(0, cost - tf.shape(x)[1])]])[:, :cost]
        return tf.reduce_sum(tf.matmul(x, w))

    return step


def time_epochs(ds, step, epochs: int) -> tuple[float, int]:
    """One warm-up epoch (graph tracing, first file reads), then timed epochs."""
    n = 0
    for features, _ in ds:
        step(features)
        n += int(features["seq"].shape[0])

    t0 = time.perf_counter()
    for _ in range(epochs):
        for features, _ in ds:
            step(features)
    elapsed = time.perf_counter() - t0
    return elapsed / epochs, n


def precomputed_dataset(spec, seasons, batch_size, tfr_dir, aug_kwargs):
    """Answer D, built so its cost can be compared honestly.

    Augment once, materialise the result, replay it every epoch. It will be the
    fastest row in the table, and it is still the wrong answer -- see
    `distinct_draws` for the reason, which is not about speed.

    NOTE ON .cache(): a tf.data cache is only populated by an iteration that
    runs to COMPLETION. Half-iterate it -- with .take(1), say -- and the
    partial cache is discarded and the upstream map runs again next epoch.
    Getting this wrong makes a frozen pipeline look fresh, which is precisely
    backwards, so callers here must drain it once before drawing conclusions.
    """
    ds = make_dataset(seasons, spec, batch_size=batch_size, training=True,
                      tfr_dir=tfr_dir, aug_kwargs=aug_kwargs, seed=0)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def distinct_draws(spec, seasons, tfr_dir, aug_kwargs, epochs: int = 4,
                   batches: int = 4) -> None:
    """Show that the two approaches differ in KIND, not just in speed.

    This is the half of the exam question a stopwatch cannot answer. Both
    pipelines are pinned to a fixed, unshuffled slice of one season, so the
    ONLY thing that can vary between epochs is the augmentation itself --
    otherwise shuffling would be mistaken for freshness.
    """
    print(f"\ndistinct versions of the same {batches} batches, over {epochs} epochs")
    print("  (fixed order, no shuffling -- any variation is the augmentation)")

    # Mid-season on purpose. Records are written ordered by (season, week), so
    # the FIRST batches of the earliest season are week 1 of 2016 -- players
    # with no prior games at all, whose sequences are pure padding. Augmenting
    # an all-zero sequence returns an all-zero sequence every time, and this
    # check would then report "frozen" for both pipelines and appear to prove
    # something it had not tested. Skip into the season and verify there is
    # real history before drawing any conclusion.
    season = [seasons[-1]]
    base = make_dataset(season, spec, batch_size=64, training=False,
                        tfr_dir=tfr_dir, augment=False).skip(30).take(batches)

    depth = np.mean([f["seq_mask"].numpy().sum(axis=1).mean() for f, _ in base])
    print(f"  slice: season {season[0]}, mean history depth "
          f"{depth:.2f}/{spec['seq_len']} games")
    if depth < 1.0:
        print("  ABORT: this slice is all padding, nothing to augment.")
        return

    augment = make_augment(spec, **aug_kwargs)

    fresh = base.map(augment)
    # Drained once so the cache is actually complete; see precomputed_dataset.
    frozen = base.map(augment).cache()
    for _ in frozen:
        pass

    for label, ds in [("in-pipeline map (A)", fresh),
                      ("precomputed + cached (D)", frozen)]:
        seen = set()
        for _ in range(epochs):
            digest = [f["seq"].numpy().tobytes() for f, _ in ds]
            seen.add(hash(b"".join(digest)))
        verdict = ("fresh every epoch -- the model sees a new view each time"
                   if len(seen) == epochs else
                   "FROZEN after the first draw -- every epoch is identical")
        print(f"    {label:28s} {len(seen)} distinct  -> {verdict}")


def run_configs(spec, seasons, batch_size, epochs, step, tfr_dir, aug_cost,
                show: bool) -> dict[str, float]:
    """Time every config at one augmentation cost. Returns {name: seconds}."""
    aug_kwargs = {"drop": 2, "jitter": 0.05, "crop": 2, "extra_cost": aug_cost}
    out = {}

    for name, cfg in CONFIGS.items():
        opts = {k: v for k, v in cfg.items() if k != "note"}
        ds = make_dataset(seasons, spec, batch_size=batch_size, training=True,
                          tfr_dir=tfr_dir, aug_kwargs=aug_kwargs, seed=0, **opts)
        secs, n = time_epochs(ds, step, epochs)
        out[name] = secs
        if show:
            print(f"  {name:24s} {secs:7.2f}s/epoch  {n / secs:9,.0f} ex/s   "
                  f"{cfg['note']}")

    ds = precomputed_dataset(spec, seasons, batch_size, tfr_dir, aug_kwargs)
    secs, n = time_epochs(ds, step, epochs)
    out["6 precomputed (answer D)"] = secs
    if show:
        print(f"  {'6 precomputed (answer D)':24s} {secs:7.2f}s/epoch  "
              f"{n / secs:9,.0f} ex/s   augment ONCE, replay -- fast and wrong")
    return out


def main(seasons, batch_size, epochs, step_cost, aug_cost, sweep, tfr_dir) -> None:
    spec = load_spec(tfr_dir)
    records = sum(spec["counts"][str(s)] for s in seasons)

    print(f"seasons {seasons[0]}-{seasons[-1]}  batch {batch_size}  "
          f"{records:,} records")
    print(f"model step: matmul {step_cost}x{step_cost} per batch")
    print("augmentation: crop<=2 games, drop<=2 games, 5% stat jitter")
    print(f"timing {epochs} epochs after one warm-up epoch")

    step = make_step(step_cost)

    print(f"\n--- at the real augmentation cost ---------------------------")
    base = run_configs(spec, seasons, batch_size, epochs, step, tfr_dir,
                       aug_cost, show=True)
    naive, tuned = base["1 naive"], base["4 +cache (recommended)"]
    print(f"\n  tuned is {naive / tuned:.2f}x naive "
          f"({naive:.2f}s -> {tuned:.2f}s per epoch)")

    if sweep:
        # THE ACTUAL LESSON. On 8x19 floats the augmentation is so cheap that
        # AUTOTUNE has almost nothing to parallelise, and the honest result at
        # cost 0 is "barely moves". Decoding a JPEG and applying three image
        # transforms costs hundreds of microseconds per example instead of a
        # handful. This sweep buys that per-example cost synthetically so the
        # TREND is visible on a laptop: as the map gets more expensive, the
        # naive pipeline degrades and the tuned one holds. That relationship
        # is the thing worth carrying into the exam -- not any single number
        # measured on a dataset this small.
        print(f"\n--- sweep: how the gap grows with per-example cost --------")
        print("    (synthetic load, standing in for real image decode+augment)")
        print(f"\n  {'extra ops':>10}  {'naive':>8}  {'tuned':>8}  {'speedup':>8}")
        for cost in sweep:
            r = run_configs(spec, seasons, batch_size, epochs, step, tfr_dir,
                            cost, show=False)
            n_s, t_s = r["1 naive"], r["4 +cache (recommended)"]
            print(f"  {cost:>10}  {n_s:7.2f}s  {t_s:7.2f}s  {n_s / t_s:7.2f}x")

    distinct_draws(spec, seasons, tfr_dir,
                   {"drop": 2, "jitter": 0.05, "crop": 2, "extra_cost": 0})

    print("""
READ THE TABLE THIS WAY
  Rows 1-4 are the exam answer, earned one option at a time.
  Row 5 shows what map-before-batch costs: identical arithmetic, more dispatch.
  Row 6 is the trap, and the two checks above disagree about it on purpose.
  It is genuinely the fastest way to move bytes AND it trains on one frozen
  draw forever. Speed was never the reason it is wrong -- which is why the
  question says "randomly applied to each training batch" before it says
  anything about runtime. Answer D also pays for a Dataflow job before every
  single training run to produce that frozen draw.""")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", type=int, nargs="+",
                   default=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5,
                   help="short epochs are mostly timer noise; keep this >= 5")
    p.add_argument("--step-cost", type=int, default=1024,
                   help="matmul size standing in for the model's step")
    p.add_argument("--aug-cost", type=int, default=0,
                   help="synthetic per-batch ops added to the headline table")
    p.add_argument("--sweep", type=int, nargs="*", default=[0, 20, 80, 240],
                   help="augmentation costs to sweep; empty to skip")
    p.add_argument("--tfr-dir", default=str(TFR))
    a = p.parse_args()
    tf.get_logger().setLevel("ERROR")
    main(a.seasons, a.batch_size, a.epochs, a.step_cost, a.aug_cost,
         a.sweep, Path(a.tfr_dir))
