"""Build the modeling table: one row per player-week, features + label.

LEAKAGE DISCIPLINE (see FEATURES.md section 8): every player-history feature is
computed then shifted one game forward, so a week-N row contains only data that
existed before week N kicked off. Vegas lines, weather and rest are known before
kickoff and are used as-is.

Run: python src/features.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from scoring import add_dk_points  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

SKILL = ["QB", "RB", "WR", "TE"]
ROLL_WINDOWS = (3, 5, 8)


# --------------------------------------------------------------- game context
def team_game_context() -> pd.DataFrame:
    """One row per team-game: Vegas, weather, rest, and QB experience."""
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[s["game_type"].eq("REG")].copy()

    sides = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({
            "season": s["season"],
            "week": s["week"],
            "team": s[f"{side}_team"],
            "opponent": s[f"{opp}_team"],
            "is_home": int(side == "home"),
            "qb_id": s[f"{side}_qb_id"],
            "rest": s[f"{side}_rest"],
            "total_line": s["total_line"],
            "wind": s["wind"],
            "temp": s["temp"],
            "roof": s["roof"],
            "surface": s["surface"],
            "div_game": s["div_game"],
            "gameday": s["gameday"],
        })
        # spread_line is signed from the HOME team's perspective
        d["team_spread"] = s["spread_line"] if side == "home" else -s["spread_line"]
        sides.append(d)

    g = pd.concat(sides, ignore_index=True)

    # The workhorse Vegas feature -- expected points for THIS team.
    g["implied_team_total"] = g["total_line"] / 2 + g["team_spread"] / 2
    g["implied_opp_total"] = g["total_line"] / 2 - g["team_spread"] / 2

    # Domes have no wind/temp. That is a real condition, not missing data.
    is_indoor = g["roof"].isin(["dome", "closed"])
    g["is_indoor"] = is_indoor.astype(int)
    g["wind"] = np.where(is_indoor, 0.0, g["wind"])
    g["temp"] = np.where(is_indoor, 70.0, g["temp"])
    g["wind"] = g["wind"].fillna(g["wind"].median())
    g["temp"] = g["temp"].fillna(g["temp"].median())

    # Inexperienced starting QB. FEATURES.md 9.2 found WR1s score 22% LESS
    # with one -- the user's rule, inverted.
    g = g.sort_values(["season", "week"])
    g["qb_prior_starts"] = g.groupby("qb_id").cumcount()
    g["green_qb"] = (g["qb_prior_starts"] <= 5).astype(int)

    return g


# ------------------------------------------------------------ defense allowed
def defense_allowed(w: pd.DataFrame) -> pd.DataFrame:
    """Rolling DK points a defense has allowed to each position.

    Computed on prior weeks only, then joined onto the player's opponent.
    """
    d = (w.groupby(["opponent_team", "season", "week", "position"], as_index=False)
           ["dk_points"].sum()
           .rename(columns={"opponent_team": "defteam", "dk_points": "allowed"}))
    d = d.sort_values(["defteam", "position", "season", "week"])

    g = d.groupby(["defteam", "position", "season"], sort=False)["allowed"]
    d["def_allowed_r4"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    d["def_allowed_r8"] = g.transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    return d[["defteam", "season", "week", "position", "def_allowed_r4", "def_allowed_r8"]]


# ------------------------------------------------------------------ main build
def build(weekly: pd.DataFrame | None = None) -> pd.DataFrame:
    """`weekly` lets src/predict_week.py pass in extra rows for games not yet
    played (flagged `_future`, stats blank). Every feature is shifted one game,
    so those rows get features from completed games only."""
    w = pd.read_parquet(RAW / "weekly.parquet") if weekly is None else weekly.copy()
    if "_future" not in w.columns:
        w["_future"] = False
    w["_future"] = w["_future"].fillna(False).astype(bool)
    w = w[w["season_type"].eq("REG") & w["position"].isin(SKILL)].copy()
    # The stats_player source also lists games where a player's only stat was
    # on special teams or defense (a coverage tackle). Keep games where he did
    # something on offense -- the same population the old weekly file had.
    offense = ["attempts", "carries", "targets", "receptions",
               "passing_yards", "rushing_yards", "receiving_yards", "sacks"]
    w = w[(w[offense].fillna(0).abs().sum(axis=1) > 0) | w["_future"]]
    w = add_dk_points(w)
    w.loc[w["_future"], "dk_points"] = np.nan  # not played yet: no answer

    # snap share -- joined on name+team because snaps use PFR ids, not GSIS
    try:
        sn = pd.read_parquet(RAW / "snaps.parquet")
        sn = (sn.groupby(["player", "team", "season", "week"], as_index=False)["offense_pct"]
                .max()
                .rename(columns={"player": "player_display_name", "team": "recent_team"}))
        w = w.merge(sn, on=["player_display_name", "recent_team", "season", "week"], how="left")
    except FileNotFoundError:
        w["offense_pct"] = np.nan

    w = w.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = w.groupby(["player_id", "season"], sort=False)

    # ---- opportunity + form, all shifted (section 1 and 3)
    rolling_cols = {
        "dk_points": "dkp",
        "target_share": "tgt_share",
        "air_yards_share": "ay_share",
        "wopr": "wopr",
        "targets": "targets",
        "carries": "carries",
        "offense_pct": "snap_pct",
        "receptions": "rec",
    }
    for col, name in rolling_cols.items():
        if col not in w.columns:
            continue
        shifted = g[col].shift(1)
        w[f"{name}_lag1"] = shifted
        for n in ROLL_WINDOWS:
            w[f"{name}_r{n}"] = g[col].transform(
                lambda s, n=n: s.shift(1).rolling(n, min_periods=max(2, n // 3)).mean()
            )
        w[f"{name}_ewm"] = g[col].transform(lambda s: s.shift(1).ewm(halflife=3).mean())

    w["games_so_far"] = g.cumcount()
    w["dkp_std_r8"] = g["dk_points"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=4).std()
    )
    # opportunity per snap -- full-time decoy vs part-time weapon
    w["targets_per_snap_r5"] = w["targets_r5"] / w["snap_pct_r5"].replace(0, np.nan)

    # ---- game context (section 2, 6)
    ctx = team_game_context()
    w = w.merge(
        ctx.drop(columns=["opponent"]),
        left_on=["recent_team", "season", "week"],
        right_on=["team", "season", "week"],
        how="left",
    ).drop(columns=["team"])

    # ---- matchup (section 4)
    w = w.merge(
        defense_allowed(w),
        left_on=["opponent_team", "season", "week", "position"],
        right_on=["defteam", "season", "week", "position"],
        how="left",
    ).drop(columns=["defteam"], errors="ignore")

    # ---- game-script interaction: a big favorite's RB and a big underdog's WR
    # are both good plays, for opposite reasons.
    w["spread_x_rb"] = w["team_spread"] * w["position"].eq("RB").astype(int)
    w["spread_x_pass"] = w["team_spread"] * w["position"].isin(["WR", "TE"]).astype(int)

    w["position"] = w["position"].astype("category")
    return w


FEATURE_COLS = [
    # opportunity / form
    "dkp_r3", "dkp_r5", "dkp_r8", "dkp_ewm", "dkp_lag1", "dkp_std_r8",
    "tgt_share_r3", "tgt_share_r5", "tgt_share_ewm",
    "ay_share_r5", "wopr_r5", "targets_r3", "targets_r5",
    "carries_r3", "carries_r5", "rec_r5",
    "snap_pct_r3", "snap_pct_r5", "targets_per_snap_r5",
    "games_so_far",
    # vegas / context
    "implied_team_total", "implied_opp_total", "total_line", "team_spread",
    "is_home", "rest", "div_game", "green_qb",
    # matchup
    "def_allowed_r4", "def_allowed_r8",
    # weather
    "wind", "temp", "is_indoor",
    # interactions
    "spread_x_rb", "spread_x_pass",
]


if __name__ == "__main__":
    PROC.mkdir(parents=True, exist_ok=True)
    df = build()
    keep = ["player_id", "player_display_name", "position", "recent_team",
            "opponent_team", "season", "week", "dk_points"] + FEATURE_COLS
    out = df[[c for c in keep if c in df.columns]]
    out.to_parquet(PROC / "model_table.parquet", index=False)

    print(f"model table: {out.shape[0]:,} rows x {out.shape[1]} cols")
    print(f"seasons {out.season.min()}-{out.season.max()}")
    print(f"label mean {out.dk_points.mean():.2f}, std {out.dk_points.std():.2f}")
    miss = out[FEATURE_COLS].isna().mean().sort_values(ascending=False)
    print("\nhighest-missingness features:")
    print((miss[miss > 0].head(8) * 100).round(1).to_string())
