"""Turn the model table into sharded TFRecords of per-player game sequences.

This is data plumbing, not modeling. It runs once. Everything expensive and
DETERMINISTIC happens here -- parsing parquet, joining, windowing, serialising.
Nothing random happens here, and that distinction is the whole point of the
file (see tf_sequence/README.md).

WHAT ONE RECORD IS
------------------
One row of model_table_v3 = one player, one week = one prediction the model is
asked to make. Attached to it:

  seq        [SEQ_LEN, N_SEQ]  the player's last 8 games, oldest first, raw
                               box score. Games strictly BEFORE this week.
  seq_mask   [SEQ_LEN]         1.0 = a real game sits in this slot,
                               0.0 = padding because the player has fewer
                               than 8 prior games (rookies, debuts, returns).
  static     [STATIC_WIDTH]    this week's context + missing-value flags.
  label      scalar            dk_points actually scored.
  season/week/pos              carried along for grading, never for training.

THE LEAKAGE RULE, ENFORCED MECHANICALLY
---------------------------------------
FEATURES.md section 8 rule 1: a Week 8 row may only contain data through
Week 7. Here that is not a convention to remember, it is a slice bound. Games
are sorted into a global week ordinal and the window is `log[:idx]` where idx
is found by bisect on a STRICTLY-less-than comparison. There is no way to
write this loop and accidentally include the current game.

WHY SHARD BY SEASON
-------------------
Walk-forward validation trains on seasons < S and tests on S. If every season
is its own shard, a fold is a file-glob -- no filtering of a big shuffled file,
and tf.data can interleave reads across the training shards in parallel.
Sharding for the access pattern you actually have is the normal reason to
shard, and it lines up here with the way the model has to be graded.

Run:  .venv-tf/Scripts/python tf_sequence/build_sequences.py
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import scoring  # noqa: E402  -- the project's DK scoring, not nflverse's generic column

from common import (  # noqa: E402
    N_SEQ, POSITIONS, PROC, RAW, SEQ_COLS, SEQ_LEN, STATIC_COLS, TARGET, TFR,
)


def global_week(season: pd.Series, week: pd.Series) -> pd.Series:
    """A single increasing integer per NFL week, so 2021 wk1 follows 2020 wk17.

    season*100+week would also sort correctly, but the GAPS would be wrong --
    2020 wk17 to 2021 wk1 would look like an 84-week hole. `weeks_ago` is a
    real feature fed to the network, so the spacing has to mean something.
    """
    key = season.astype(int) * 100 + week.astype(int)
    ordinals = {k: i for i, k in enumerate(sorted(key.unique()))}
    return key.map(ordinals).astype(int)


def load_game_log() -> pd.DataFrame:
    """Per-game raw box score for every skill player, with DK points attached."""
    w = pd.read_parquet(RAW / "weekly.parquet")
    w = w[w["season_type"].eq("REG") & w["position"].isin(POSITIONS)].copy()
    w["dk_points"] = scoring.dk_points(w)
    keep = ["player_id", "season", "week"] + [c for c in SEQ_COLS if c != "weeks_ago"]
    w = w[keep].dropna(subset=["player_id"])
    return w.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def build_windows(targets: pd.DataFrame, log: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For every target row, gather the player's previous SEQ_LEN games.

    Returns (seq [N, SEQ_LEN, N_SEQ], mask [N, SEQ_LEN]), oldest game first.
    Padding is at the FRONT, so the most recent game is always the last slot --
    a fixed position the network can rely on.
    """
    stat_cols = [c for c in SEQ_COLS if c != "weeks_ago"]
    n_stats = len(stat_cols)

    # one contiguous float array + one sorted week list per player
    by_player: dict[str, tuple[np.ndarray, list[int]]] = {}
    for pid, g in log.groupby("player_id", sort=False):
        by_player[pid] = (
            g[stat_cols].to_numpy(dtype=np.float32),
            g["gw"].tolist(),  # already sorted -- log was sorted before grouping
        )

    n = len(targets)
    seq = np.zeros((n, SEQ_LEN, N_SEQ), dtype=np.float32)
    mask = np.zeros((n, SEQ_LEN), dtype=np.float32)

    pids = targets["player_id"].to_numpy()
    gws = targets["gw"].to_numpy()
    missing = 0

    for i in range(n):
        entry = by_player.get(pids[i])
        if entry is None:
            missing += 1
            continue
        arr, weeks = entry

        # bisect_left on a strictly-increasing list gives the count of games
        # STRICTLY before this week. This single line is the leakage guard.
        idx = bisect.bisect_left(weeks, int(gws[i]))
        if idx == 0:
            continue  # no prior games at all: all-padding, mask stays zero

        lo = max(0, idx - SEQ_LEN)
        window = arr[lo:idx]
        k = len(window)

        seq[i, SEQ_LEN - k:, :n_stats] = window
        seq[i, SEQ_LEN - k:, n_stats] = gws[i] - np.asarray(weeks[lo:idx], dtype=np.float32)
        mask[i, SEQ_LEN - k:] = 1.0

    if missing:
        print(f"  {missing:,} target rows had no game log at all (left as padding)")
    return seq, mask


def build_static(targets: pd.DataFrame) -> np.ndarray:
    """Static context + a missing-value flag per column. See common.py."""
    vals = targets[STATIC_COLS].to_numpy(dtype=np.float32)
    flags = np.isnan(vals).astype(np.float32)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([vals, flags], axis=1)


def serialise(seq_row, mask_row, static_row, label, season, week, pos) -> bytes:
    f = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))  # noqa: E731
    i = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))  # noqa: E731
    ex = tf.train.Example(features=tf.train.Features(feature={
        "seq": f(seq_row.reshape(-1)),
        "seq_mask": f(mask_row),
        "static": f(static_row),
        "label": f([label]),
        "season": i([season]),
        "week": i([week]),
        "pos": i([pos]),
    }))
    return ex.SerializeToString()


def main(table: str, out_dir: Path) -> None:
    print(f"reading {PROC / (table + '.parquet')}")
    df = pd.read_parquet(PROC / f"{table}.parquet")
    df = df[df["position"].isin(POSITIONS)].copy()

    log = load_game_log()

    # one shared ordinal space for targets and history, or the bisect is garbage
    both = pd.concat([
        df[["season", "week"]], log[["season", "week"]],
    ], ignore_index=True)
    ordinals = {k: n for n, k in enumerate(
        sorted(set(zip(both["season"].astype(int), both["week"].astype(int))))
    )}
    df["gw"] = [ordinals[(s, w)] for s, w in zip(df["season"].astype(int), df["week"].astype(int))]
    log["gw"] = [ordinals[(s, w)] for s, w in zip(log["season"].astype(int), log["week"].astype(int))]
    log = log.sort_values(["player_id", "gw"]).reset_index(drop=True)

    df = df.sort_values(["season", "week", "player_id"]).reset_index(drop=True)
    print(f"{len(df):,} target rows, {len(log):,} games in the log")

    print("building windows...")
    seq, mask = build_windows(df, log)
    static = build_static(df)
    labels = df[TARGET].to_numpy(dtype=np.float32)
    pos_idx = df["position"].astype(str).map({p: i for i, p in enumerate(POSITIONS)}).to_numpy()

    full = mask.sum(axis=1)
    print(f"  {(full == SEQ_LEN).mean():.1%} of rows have a full {SEQ_LEN}-game history")
    print(f"  {(full == 0).mean():.1%} have none (debuts)  |  mean depth {full.mean():.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("season_*.tfrecord"):
        old.unlink()

    seasons = sorted(df["season"].unique())
    counts = {}
    for s in seasons:
        rows = np.flatnonzero(df["season"].to_numpy() == s)
        path = out_dir / f"season_{int(s)}.tfrecord"
        opts = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(str(path), options=opts) as w:
            for r in rows:
                w.write(serialise(
                    seq[r], mask[r], static[r], labels[r],
                    int(df["season"].iat[r]), int(df["week"].iat[r]), int(pos_idx[r]),
                ))
        counts[int(s)] = len(rows)
        print(f"  season_{int(s)}.tfrecord  {len(rows):,} records  "
              f"{path.stat().st_size / 1e6:.1f} MB")

    spec = {
        "seq_len": SEQ_LEN,
        "n_seq": N_SEQ,
        "seq_cols": SEQ_COLS,
        "static_cols": STATIC_COLS,
        "static_width": static.shape[1],
        "positions": POSITIONS,
        "target": TARGET,
        "table": table,
        "counts": counts,
        "compression": "GZIP",
    }
    (out_dir / "spec.json").write_text(json.dumps(spec, indent=2))
    print(f"\nwrote spec.json -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="model_table_v3")
    p.add_argument("--out", default=str(TFR))
    a = p.parse_args()
    main(a.table, Path(a.out))
