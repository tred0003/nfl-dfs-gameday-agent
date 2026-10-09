"""The Keras model. Two branches, because the inputs are two different kinds.

WHERE THE MODELING ACTUALLY STARTS
----------------------------------
Everything up to this point was plumbing: read parquet, window, serialise,
shuffle. This file is the first place a real choice gets made, and the choice
is the architecture.

A player-week is two unlike things stapled together:

  a SEQUENCE  eight prior games, in order, where the order carries meaning.
              A player going 4 -> 9 -> 14 targets is a different bet from one
              going 14 -> 9 -> 4, and their average is identical.

  a SNAPSHOT  this week's context -- Vegas total, wind, who is inactive, price.
              No order, no history, just a row of numbers.

Flattening the sequence into the snapshot and running one dense stack over the
whole thing would throw away the ordering, which is the only reason to build a
sequence model in the first place. So: a recurrent branch reads the games, a
dense branch reads the context, and they meet at a concatenate.

WHY GRU AND NOT LSTM OR A TRANSFORMER
-------------------------------------
The sequence is EIGHT steps long. Eight. Attention's advantage is modelling
long-range dependencies across hundreds of positions, and it costs a large
parameter budget to get there -- on 41k training rows that budget buys
overfitting, not skill. LSTM has a third gate that matters for long sequences
and is mostly dead weight at length 8. GRU is the smallest thing that actually
reads order. Picking the smallest architecture that can express the structure
you believe is there is the whole discipline; reaching for a transformer
because it is the impressive answer is how you lose to a rolling average.

PREPROCESSING LIVES INSIDE THE MODEL
------------------------------------
The Normalization layers are adapted on training data and then saved AS PART
OF the model. A SavedModel exported from here takes raw feature values and
normalises them itself.

This matters more than it looks. The classic way to ship a broken model is to
normalise in a training script with one set of means, then write separate
serving code that computes means slightly differently -- different rows,
different NaN handling, a stale constant. The model is then fed a distribution
it never trained on and quietly degrades. Nobody gets an error.

Keeping the constants inside the exported graph makes that failure impossible
rather than unlikely. It is the same idea TF Transform implements at pipeline
scale, available here with no TFX installation.
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, ops


def adapt_arrays(ds, max_batches: int = 400):
    """Collect raw feature values to fit the Normalization constants on.

    Sequence stats are gathered ONLY from slots where the mask is 1. Padding is
    an artefact of short careers, not a measurement -- letting ~19% zero-filled
    slots into the mean would pull every column toward zero and shrink the
    variance, so the layer would then under-scale real games.
    """
    seqs, stats = [], []
    for i, (feat, _) in enumerate(ds):
        if i >= max_batches:
            break
        seq = feat["seq"].numpy().reshape(-1, feat["seq"].shape[-1])
        m = feat["seq_mask"].numpy().reshape(-1) > 0
        seqs.append(seq[m])
        stats.append(feat["static"].numpy())
    return np.concatenate(seqs), np.concatenate(stats)


def build_model(spec: dict, seq_adapt: np.ndarray, static_adapt: np.ndarray,
                gru_units: int = 64, dense_units: int = 64, dropout: float = 0.2,
                learning_rate: float = 1e-3) -> keras.Model:
    seq_len, n_seq = spec["seq_len"], spec["n_seq"]
    n_static = spec["static_width"]
    n_pos = len(spec["positions"])

    seq_in = keras.Input(shape=(seq_len, n_seq), name="seq")
    mask_in = keras.Input(shape=(seq_len,), name="seq_mask")
    static_in = keras.Input(shape=(n_static,), name="static")
    pos_in = keras.Input(shape=(1,), dtype="int64", name="pos")

    # ---- sequence branch
    seq_norm = layers.Normalization(axis=-1, name="seq_norm")
    seq_norm.adapt(seq_adapt)
    x = seq_norm(seq_in)
    # Re-zero padded slots: normalising turned every padded 0.0 into
    # -mean/std, a fabricated value. The GRU mask stops it being READ, but
    # leaving it in place would be a trap for anyone who later swaps the GRU
    # for a layer that ignores masks.
    #
    # keras.ops, not tf.*: in Keras 3 the functional API builds on symbolic
    # KerasTensors, and a raw TensorFlow op rejects them outright. keras.ops
    # mirrors the numpy/TF surface and is what works on both.
    x = ops.multiply(x, ops.expand_dims(mask_in, -1))

    mask_bool = ops.cast(mask_in, "bool")
    # A fully-masked sequence -- a debut, or augmentation that dropped every
    # game -- makes the GRU emit zeros. Not an error: the static branch still
    # carries Vegas, salary and depth chart, which is genuinely all you know
    # about a player making his first start.
    x = layers.GRU(gru_units, name="game_history")(x, mask=mask_bool)
    x = layers.Dense(dense_units // 2, activation="relu", name="seq_head")(x)

    # ---- static branch
    static_norm = layers.Normalization(axis=-1, name="static_norm")
    static_norm.adapt(static_adapt)
    s = static_norm(static_in)
    s = layers.Dense(dense_units, activation="relu")(s)
    s = layers.Dropout(dropout)(s)
    s = layers.Dense(dense_units // 2, activation="relu", name="static_head")(s)

    # ---- position: learned, not one-hot
    # Four positions, so one-hot would work fine and this is not a capacity
    # argument. It is a shape argument: an embedding places QB/RB/WR/TE in a
    # space where the network can learn that two of them behave alike, which
    # one-hot's mutually-orthogonal columns cannot express.
    p = layers.Embedding(n_pos, 4, name="position_embedding")(pos_in)
    p = layers.Flatten()(p)

    z = layers.Concatenate()([x, s, p])
    z = layers.Dense(dense_units, activation="relu")(z)
    z = layers.Dropout(dropout)(z)
    out = layers.Dense(1, name="dk_points")(z)

    model = keras.Model(
        inputs={"seq": seq_in, "seq_mask": mask_in, "static": static_in, "pos": pos_in},
        outputs=out,
        name="dfs_sequence",
    )
    # Mean squared error on purpose: src/train.py fits XGBoost with
    # reg:squarederror, and changing the loss at the same time as the
    # architecture would leave you unable to say which one moved the number.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model
