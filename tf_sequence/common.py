"""Shared column spec for the TensorFlow sequence model.

THE ONE DECISION THAT MAKES THIS EXPERIMENT HONEST
--------------------------------------------------
`model_table_v3.parquet` already contains hand-built rolling features:
dkp_r3, dkp_r5, dkp_r8, dkp_ewm, tgt_share_r5, route_share_ewm, and so on.
Someone wrote code that looked back over a player's last N games and reduced
them to an average. That reduction IS the temporal modeling -- done by hand,
in pandas, by a human who decided that 3, 5 and 8 games were the right windows.

The entire pitch of a sequence model is: don't decide that by hand. Hand the
network the raw game-by-game history and let it learn its own weighting.

So if the network got BOTH the raw sequence AND the pre-rolled averages, the
comparison would be meaningless -- it would be scoring a model that was handed
the answer alongside the question. The split below is therefore strict:

  SEQ_COLS     raw per-game box score, straight from weekly.parquet.
               One row per game the player actually played. No aggregation.
               This is the sequence branch's only view of player history.

  STATIC_COLS  this week's context, which has no player history in it at all:
               Vegas lines, weather, injury/vacated opportunity, salary,
               draft position, depth chart. Things that are true about
               *this matchup*, not about the player's recent form.

Every player-rolling column in model_table_v3 is deliberately EXCLUDED (see
EXCLUDED_ROLLING below). XGBoost keeps them. The network has to rediscover
them from raw games. If the network wins anyway, it learned a better temporal
summary than the hand-built one. If it loses, the hand-built windows were
good and that is a real finding, not a failure.

Note on opponent features: def_allowed_r4 and def_allowed_r8 are rolling, but
they roll over the OPPONENT DEFENSE, not over this player. Nothing in the
player's own game log can reconstruct them, so they stay in STATIC_COLS. The
rule being enforced is "the network may not be handed a summary of its own
sequence," not "no column may contain the letter r followed by a digit."
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
TFR = ROOT / "data" / "tfrecords"
OUT = ROOT / "models" / "tf_sequence"

TARGET = "dk_points"
SEQ_LEN = 8  # how many prior games the network sees

# Matches src/train.py so the two models are graded on identical folds.
TEST_SEASONS = [2021, 2022, 2023, 2024]

POSITIONS = ["QB", "RB", "WR", "TE"]


# --------------------------------------------------------------- sequence
# Raw per-game production, from data/raw/weekly.parquet. All are 100% populated
# in that file, so a padded slot is the ONLY reason a sequence value is zero --
# which is what makes the mask trustworthy.
SEQ_COLS = [
    "dk_points",          # computed by src/scoring.py, not nflverse's generic column
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_air_yards",
    "receiving_tds",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "target_share",
    "air_yards_share",
    "wopr",
    "fumbles_lost_total",
    # Derived at build time, not a box-score column: how many weeks before the
    # game being predicted this game happened. Without it the network sees
    # eight games in a row and cannot tell a clean streak from a log with an
    # eight-week injury gap buried in it.
    "weeks_ago",
]
N_SEQ = len(SEQ_COLS)


# ----------------------------------------------------------------- static
# This week's context. Nothing here summarises the player's own recent games.
STATIC_COLS = [
    # game environment (Vegas)
    "implied_team_total", "implied_opp_total", "total_line", "team_spread",
    "is_home", "rest", "div_game", "spread_x_rb", "spread_x_pass",
    # opponent defense -- rolls over the DEFENSE, not over this player
    "def_allowed_r4", "def_allowed_r8",
    # weather
    "wind", "temp", "is_indoor",
    # availability / vacated opportunity
    "vacated_tgt_share", "vacated_carry_share", "new_vacated_tgt_share",
    "n_out", "vacated_same_group", "starting_qb_out", "own_questionable",
    "own_limited_practice", "healthy_tgt_rank", "depth_rank", "green_qb",
    # price (a constraint, not a points predictor -- see FEATURES.md sec. 7,
    # but salary movement reflects DK's own reaction to news, which is signal)
    "salary", "salary_implied_pts", "salary_chg_1w", "salary_chg_3w",
    "salary_pos_pct",
    # player identity that is NOT recent form
    "draft_round", "draft_pick", "years_pro", "is_rookie", "age",
    "games_so_far", "changed_team",
]
N_STATIC = len(STATIC_COLS)


# Held out from the network on purpose. Each one is a hand-built summary of the
# very game log the sequence branch is being asked to read for itself.
EXCLUDED_ROLLING = [
    "dkp_r3", "dkp_r5", "dkp_r8", "dkp_ewm", "dkp_lag1", "dkp_std_r8",
    "dkp_blended",
    "tgt_share_r3", "tgt_share_r5", "tgt_share_ewm",
    "ay_share_r5", "wopr_r5",
    "targets_r3", "targets_r5", "carries_r3", "carries_r5", "rec_r5",
    "snap_pct_r3", "snap_pct_r5", "targets_per_snap_r5",
    "route_share_r3", "route_share_r5", "route_share_ewm", "route_share_last",
    "tprr_r5", "route_share_trend",
    "rz_tgt_share_r5", "ez_tgt_share_r5", "i10_carry_share_r5",
    "i5_carry_share_r5", "rz_tgt_r5", "i10_carries_r5",
    "value_last", "value_r3", "value_streak",
    "ps_games", "ps_dkp_pg", "ps_last4_dkp_pg", "ps_targets_pg",
    "ps_carries_pg", "ps_tgt_share", "ps_ay_share",
]


# ------------------------------------------------------------- feature width
# Every static column ships alongside a 0/1 "was this missing" flag.
#
# XGBoost gets native missing-value handling for free: it learns a default
# branch direction for every split. A dense layer has no such thing -- feed it
# a NaN and the whole network becomes NaN on the first batch. So missing values
# are written as 0.0 and the flag tells the network the zero was an absence
# rather than a real zero.
#
# This is a genuine disadvantage the network carries into the comparison, and
# it is worth saying out loud rather than hiding: 0-filling shifts the column
# mean that Normalization later adapts to. The flag lets the network correct
# for it, but "lets it learn the correction" is strictly worse than "never had
# the problem." It is one of the reasons to expect gradient boosting to hold up
# well on tabular data like this.
STATIC_WIDTH = N_STATIC * 2


def static_feature_names() -> list[str]:
    return STATIC_COLS + [f"{c}__missing" for c in STATIC_COLS]
