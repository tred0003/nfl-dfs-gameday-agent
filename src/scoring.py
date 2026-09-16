"""DraftKings NFL Classic scoring -- computed from raw stats.

Why this file exists: nflverse ships `fantasy_points_ppr`, but that is generic
full-PPR and does NOT include DK's 3-point yardage bonuses. Those bonuses are
worth ~1-2 points on exactly the high-ceiling players that decide tournaments,
so training on the generic column quietly mis-ranks your best plays.

Rules encoded below are DK NFL Classic. VERIFY against DK's live rules page
before you trust a backtest -- scoring rules do get revised between seasons.
"""

from __future__ import annotations

import pandas as pd

DK_RULES = {
    "passing_yards": 0.04,          # 1 pt / 25 yds
    "passing_tds": 4.0,
    "interceptions": -1.0,
    "rushing_yards": 0.1,           # 1 pt / 10 yds
    "rushing_tds": 6.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "receptions": 1.0,              # full PPR
    "special_teams_tds": 6.0,
    "passing_2pt_conversions": 2.0,
    "rushing_2pt_conversions": 2.0,
    "receiving_2pt_conversions": 2.0,
    # DK charges -1 for ANY fumble lost. The three specific columns
    # (sack_/rushing_/receiving_fumbles_lost) miss muffed punts, kick-return
    # fumbles and fumble-recovery fumbles -- 418 player-weeks carrying 420
    # unpenalised fumbles. `fumbles_lost_total` is a clean superset: verified
    # zero rows where it is less than the sum of the three.
    "fumbles_lost_total": -1.0,
    # Offensive fumble-recovery TD, +6 on DK, previously not scored at all.
    # 252 player-weeks, 254 touchdowns -- e.g. Mike Evans 2018 wk11 was recorded
    # as 27.0 when he actually scored 33.0.
    "fumble_recovery_tds": 6.0,
}

BONUSES = [
    ("passing_yards", 300, 3.0),    # 300+ pass yds
    ("rushing_yards", 100, 3.0),    # 100+ rush yds
    ("receiving_yards", 100, 3.0),  # 100+ rec yds
]


def dk_points(df: pd.DataFrame) -> pd.Series:
    """DK Classic fantasy points for each row of an nflverse weekly frame."""
    pts = pd.Series(0.0, index=df.index)

    for col, mult in DK_RULES.items():
        if col in df.columns:
            pts += df[col].fillna(0) * mult

    for col, threshold, bonus in BONUSES:
        if col in df.columns:
            pts += (df[col].fillna(0) >= threshold) * bonus

    return pts.round(2)


def add_dk_points(df: pd.DataFrame, col: str = "dk_points") -> pd.DataFrame:
    out = df.copy()
    out[col] = dk_points(out)
    return out
