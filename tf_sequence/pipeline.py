"""The tf.data input pipeline, and the augmentation that has to live inside it.

THE RULE THIS FILE EXISTS TO MAKE PHYSICAL
------------------------------------------
    Transform once and reuse it        -> do it offline (Dataflow, TF Transform,
                                          build_sequences.py). Pay once.
    Transform differently every epoch  -> do it in tf.data.map. Pay every epoch,
                                          on the CPU, next to the accelerator.

Augmentation is the second kind. It is *supposed* to produce a different answer
every time it sees the same example -- that randomness is the entire mechanism.
Precomputing it defeats it: you would freeze one draw and then train on that
same draw every epoch, which is just a noisier fixed dataset.

That is why "generate the augmentations with Dataflow and stage them as
TFRecords" is wrong no matter how much compute you throw at it. It is not too
slow, it is the wrong shape. You would pay for a distributed job AND lose the
benefit you were paying for.

WHY tf.data AND NOT A PYTHON GENERATOR
--------------------------------------
A Keras generator runs in the Python interpreter on the main process. While it
builds batch N+1, the GPU that just finished batch N sits idle. The accelerator
is the expensive part, and it is waiting on Python.

tf.data ops are graph ops. Two arguments turn that around:

    .map(aug, num_parallel_calls=tf.data.AUTOTUNE)  fan the work across cores
    .prefetch(tf.data.AUTOTUNE)                     overlap CPU batch N+1
                                                    with GPU batch N

Run tf_sequence/benchmark.py to watch the difference in steps/sec instead of
taking that on faith.

MAP BEFORE BATCH vs MAP AFTER BATCH
-----------------------------------
The augmentations below are written to operate on a BATCH -- shape
[B, SEQ_LEN, N_SEQ] -- so they can be applied after .batch(). One vectorised
call per batch instead of B separate calls does the same arithmetic with a
fraction of the per-element dispatch overhead. `benchmark.py --config per_example`
measures the other ordering so the cost is a number rather than an assertion.

The catch, and it is a real one: after batching, a single random draw applies
to the whole batch unless you are careful to draw per-example. Every function
here draws shape [B, ...] noise, never a scalar. Getting that wrong is a quiet
bug -- training still runs, the augmentation is just far weaker than intended.
"""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from common import TFR

AUTOTUNE = tf.data.AUTOTUNE


def load_spec(tfr_dir: Path = TFR) -> dict:
    return json.loads((Path(tfr_dir) / "spec.json").read_text())


def feature_description(spec: dict) -> dict:
    return {
        "seq": tf.io.FixedLenFeature([spec["seq_len"] * spec["n_seq"]], tf.float32),
        "seq_mask": tf.io.FixedLenFeature([spec["seq_len"]], tf.float32),
        "static": tf.io.FixedLenFeature([spec["static_width"]], tf.float32),
        "label": tf.io.FixedLenFeature([1], tf.float32),
        "season": tf.io.FixedLenFeature([1], tf.int64),
        "week": tf.io.FixedLenFeature([1], tf.int64),
        "pos": tf.io.FixedLenFeature([1], tf.int64),
    }


def make_parser(spec: dict, batched: bool):
    """Deserialise TFRecord bytes into tensors.

    `batched=True` uses parse_example on a whole batch of serialised strings,
    which is markedly cheaper than parse_single_example per record -- the same
    vectorisation argument as map-after-batch, applied to parsing.
    """
    desc = feature_description(spec)
    seq_len, n_seq = spec["seq_len"], spec["n_seq"]
    parse = tf.io.parse_example if batched else tf.io.parse_single_example
    shape = [-1, seq_len, n_seq] if batched else [seq_len, n_seq]

    def _parse(raw):
        ex = parse(raw, desc)
        return {
            "seq": tf.reshape(ex["seq"], shape),
            "seq_mask": ex["seq_mask"],
            "static": ex["static"],
            "season": ex["season"],
            "week": ex["week"],
            "pos": ex["pos"],
        }, ex["label"]

    return _parse


# ------------------------------------------------------------ augmentation
# Three transforms, each the sequence-data counterpart of a standard image
# augmentation. Each is random PER EXAMPLE, PER EPOCH -- which is exactly what
# makes precomputing them incoherent.

def random_game_dropout(seq, mask, max_drop: int, seed=None):
    """Hide up to `max_drop` of the historical games. Counterpart of cutout.

    Not an arbitrary trick. In production the game log really is ragged --
    byes, injuries, inactives, a player who was on another roster. A network
    that has learned to lean on "the game exactly 3 weeks ago" breaks the first
    time that slot is empty. Dropping games during training forces it to read
    the whole window.

    Drops only where mask is already 1, so it never "un-pads" a padded slot.
    """
    b = tf.shape(mask)[0]
    n = tf.shape(mask)[1]
    # one uniform draw per (example, slot); the k smallest become the dropped k
    noise = tf.random.uniform([b, n], seed=seed)
    noise = tf.where(mask > 0, noise, tf.fill(tf.shape(noise), 2.0))  # padding sorts last
    k = tf.random.uniform([b, 1], maxval=max_drop + 1, dtype=tf.int32, seed=seed)
    rank = tf.argsort(tf.argsort(noise, axis=1), axis=1)
    keep = tf.cast(rank >= k, tf.float32)
    new_mask = mask * keep
    return seq * new_mask[:, :, None], new_mask


def random_stat_jitter(seq, mask, scale: float, n_stats: int, seed=None):
    """Multiply each stat by a per-example, per-slot draw near 1.0.

    Counterpart of contrast/brightness adjustment. A player who saw 7 targets
    could plausibly have seen 6 or 8 in the same role -- the role is the
    signal, the exact count is partly noise. Jitter says so out loud.

    `weeks_ago` is the last column and is NOT jittered: it is a calendar fact,
    not a noisy measurement. Blurring it would teach the network that time
    itself is uncertain.
    """
    b = tf.shape(seq)[0]
    n = tf.shape(seq)[1]
    f = tf.shape(seq)[2]
    noise = 1.0 + tf.random.normal([b, n, n_stats], stddev=scale, seed=seed)
    pad = tf.ones([b, n, f - n_stats])
    return seq * tf.concat([noise, pad], axis=2) * mask[:, :, None], mask


def random_window_crop(seq, mask, max_crop: int, seed=None):
    """Randomly forget the OLDEST few games. Counterpart of a random crop.

    Sequences are front-padded, so the oldest real game is the first slot with
    mask 1. Cropping from the front varies how deep a history the network gets
    while always preserving the most recent game, which is the one it can least
    afford to lose.
    """
    b = tf.shape(mask)[0]
    n = tf.shape(mask)[1]
    k = tf.random.uniform([b, 1], maxval=max_crop + 1, dtype=tf.int32, seed=seed)
    # count of padding already at the front, per example
    pad = tf.cast(n, tf.int32) - tf.cast(tf.reduce_sum(mask, axis=1, keepdims=True), tf.int32)
    pos = tf.range(n)[None, :]
    new_mask = mask * tf.cast(pos >= (pad + k), tf.float32)
    return seq * new_mask[:, :, None], new_mask


def make_augment(spec: dict, drop: int = 2, jitter: float = 0.05, crop: int = 2,
                 extra_cost: int = 0, seed=None):
    """Compose the three augmentations into one batched map function.

    `extra_cost` is a DELIBERATELY WASTEFUL knob used only by benchmark.py.
    This dataset is small -- 51k rows of 8x19 floats fits in RAM -- so the
    honest truth is that a real image pipeline has far more per-example work
    than this one does, and the AUTOTUNE/prefetch effect here is smaller than
    the effect you are being asked about on the exam. Rather than overstate
    the result, extra_cost adds measurable synthetic work so the SHAPE of the
    curve is visible on hardware you own. It is labelled as synthetic
    everywhere it appears. Set it to 0 for any real training run.
    """
    n_stats = spec["n_seq"] - 1  # weeks_ago is last and is never jittered

    def augment(features, label):
        seq, mask = features["seq"], features["seq_mask"]
        if crop:
            seq, mask = random_window_crop(seq, mask, crop, seed)
        if drop:
            seq, mask = random_game_dropout(seq, mask, drop, seed)
        if jitter:
            seq, mask = random_stat_jitter(seq, mask, jitter, n_stats, seed)

        for _ in range(extra_cost):  # synthetic load; see docstring
            seq = seq + tf.sin(seq) * 1e-7

        out = dict(features)
        out["seq"], out["seq_mask"] = seq, mask
        return out, label

    return augment


# --------------------------------------------------------------- pipelines
def shard_paths(seasons, tfr_dir: Path = TFR) -> list[str]:
    return [str(Path(tfr_dir) / f"season_{int(s)}.tfrecord") for s in seasons]


def steps_for(seasons, spec, batch_size: int) -> int:
    """Exact number of batches per epoch, from the record counts in spec.json.

    A TFRecordDataset has UNKNOWN cardinality -- there is no way to count
    records without reading the file, so tf.data reports -2 and Keras has to
    guess how long an epoch is by running one and counting. That guess is
    fragile: shard interleaving can change where batch boundaries fall, and
    when a later epoch yields a different number of batches Keras prints
    "Your input ran out of data; interrupting training" and SILENTLY TRUNCATES
    the epoch. Training still completes. The numbers are just quietly wrong.

    build_sequences.py already knows the exact per-season counts and wrote them
    to spec.json, so the honest fix is to stop guessing: repeat the dataset and
    tell Keras exactly how many steps an epoch is.
    """
    rows = sum(spec["counts"][str(int(s))] for s in seasons)
    return -(-rows // batch_size)  # ceiling division


def make_dataset(seasons, spec, batch_size=256, training=True, tfr_dir=TFR,
                 shuffle_buffer=8192, augment=True, aug_kwargs=None,
                 parallel=True, prefetch=True, cache=True, map_after_batch=True,
                 repeat=False, seed=None):
    """Build the input pipeline for a set of seasons.

    Defaults are the recommended configuration. Every knob is exposed so
    benchmark.py can turn them off one at a time and measure what each is
    worth -- the flags exist to be ablated, not to be tuned.

    ORDER MATTERS, and here is why each step sits where it does:

      interleave  read the season shards concurrently rather than end to end
      cache       the parse result is identical every epoch -- deterministic,
                  so it belongs BEFORE augmentation. Cache after augmenting and
                  you would memorise one random draw and replay it forever,
                  which is precomputing by accident.
      shuffle     after cache so the cached order isn't the training order
      batch       before map, so augmentation is vectorised over the batch
      map         the ONLY nondeterministic step; runs every epoch by design
      prefetch    always last: it overlaps this pipeline with the model step
    """
    paths = shard_paths(seasons, tfr_dir)
    comp = spec.get("compression", "GZIP")
    calls = AUTOTUNE if parallel else None

    files = tf.data.Dataset.from_tensor_slices(paths)
    if training:
        files = files.shuffle(len(paths), seed=seed)

    interleave_kw = {"cycle_length": len(paths) if parallel else 1}
    if parallel:
        # `deterministic` is meaningless without num_parallel_calls, and
        # passing it anyway earns a UserWarning on every single pipeline built.
        interleave_kw.update(num_parallel_calls=calls, deterministic=not training)
    ds = files.interleave(
        lambda p: tf.data.TFRecordDataset(p, compression_type=comp),
        **interleave_kw,
    )

    if map_after_batch:
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.map(make_parser(spec, batched=True), num_parallel_calls=calls)
        if cache:
            ds = ds.cache()
        if training:
            # batches are already formed, so shuffle reorders batches; the
            # per-example shuffle is weaker than the pre-batch version. That is
            # the real trade for vectorised parsing, and it is why this is a
            # flag rather than a hard-coded choice.
            ds = ds.shuffle(max(shuffle_buffer // batch_size, 8), seed=seed)
    else:
        ds = ds.map(make_parser(spec, batched=False), num_parallel_calls=calls)
        if cache:
            ds = ds.cache()
        if training:
            ds = ds.shuffle(shuffle_buffer, seed=seed)
        ds = ds.batch(batch_size, drop_remainder=False)

    if training and augment:
        ds = ds.map(make_augment(spec, seed=seed, **(aug_kwargs or {})),
                    num_parallel_calls=calls)

    if repeat:
        # After augmentation, never before: repeat then augment would still
        # draw fresh randomness, but repeat AFTER means each pass through the
        # data is independently augmented, which is what an epoch should be.
        ds = ds.repeat()

    if prefetch:
        ds = ds.prefetch(AUTOTUNE)
    return ds
