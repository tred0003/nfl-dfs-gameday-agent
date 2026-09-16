"""Feature batch 3: salary, value momentum, and carry-over from last season.

Built on top of batch 2. Reads model_table_v2, writes model_table_v3.

  1. SALARY  (RotoGuru 2016-2021 + FantasyData 2022-2025)
     This week's DK salary, how it moved, and where it ranks at the position.
     Salary is set days before kickoff, so it is fair game. It is also DK's own
     opinion of a player -- the only opinion available for someone with no
     games yet this season.
  2. VALUE MOMENTUM
     "Expected points at this salary" (a straight line of points on salary per
     position, fit only on EARLIER seasons), and how far a player has beaten or
     missed that line in recent games. research/test_value_streaks.py found DK
     raises prices ~7-8% after a big week but under-corrects.
  3. CARRY-OVER
     Rolling averages reset every season, so in Week 1 the model knew nothing
     about anyone (rank correlation 0.06 for players with no game yet this
     season, vs FantasyData's 0.41). Fixed with last season's numbers, draft
     capital, age, depth chart, and a blended average that leans on last season
     early and on this season as games accumulate.

Leakage rules as in features_extra.py: salary and depth chart for week N are
known before kickoff; everything else uses only games before week N.

Run: python src/features_carry.py     (after src/features_extra.py)
"""

from __future__ import annotations

import glob
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features_extra import KEYS, _as_of_previous_game  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SKILL = ["QB", "RB", "WR", "TE"]

BEAT_BY = 6        # "beat his salary" = 6+ points above the salary line
PRIOR_GAMES = 4    # blended average: last season counts as this many games


def _norm_name(name: str) -> str:
    """'Mahomes II, Patrick' and 'Patrick Mahomes' -> 'patrickmahomes'."""
    if ", " in name:
        last, first = name.split(", ", 1)
        name = f"{first} {last}"
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", name.lower())
    return re.sub(r"[^a-z]", "", name)


# ------------------------------------------------------------- 1. salary
def salary_table(table: pd.DataFrame) -> pd.DataFrame:
    """One DK salary per player-week, keyed to NFL player IDs, 2016-2025.

    RotoGuru has no NFL IDs, so it is matched by name + season + week +
    position (96% of 2016-2021 model rows; the points columns agree 99%, so the
    matches are real). FantasyData links through the ID crosswalk.
    """
    rg = pd.read_parquet(RAW / "dk_salaries.parquet")
    rg = rg[rg["position"].isin(SKILL)].dropna(subset=["salary"])
    rg["nk"] = rg["name"].map(_norm_name)
    names = table[["player_id", "player_display_name", "position"]].drop_duplicates("player_id")
    names["nk"] = names["player_display_name"].map(_norm_name)
    names["position"] = names["position"].astype(str)
    # a name shared by two players at the same position is ambiguous: skip it
    names = names.drop_duplicates(["nk", "position"], keep=False)
    rg = rg.merge(names[["nk", "position", "player_id"]], on=["nk", "position"])
    rg = rg[["player_id", "season", "week", "position", "salary"]]

    ids = pd.read_parquet(RAW / "ids.parquet")[["fantasy_data_id", "gsis_id"]].dropna()
    ids["fantasy_data_id"] = ids["fantasy_data_id"].astype(int)
    frames = []
    for f in glob.glob(str(RAW / "fantasydata" / "salaries" / "*.csv")):
        season, week = map(int, re.findall(r"dk_(\d+)_wk(\d+)", f)[0])
        frames.append(pd.read_csv(f).assign(season=season, week=week))
    fd = pd.concat(frames)
    fd = fd[fd["pos"].isin(SKILL)]
    fd["id"] = pd.to_numeric(fd["id"], errors="coerce")
    fd = fd.dropna(subset=["id"]).astype({"id": int}).merge(ids, left_on="id", right_on="fantasy_data_id")
    fd = fd.rename(columns={"gsis_id": "player_id", "pos": "position", "salary_draftkings": "salary"})
    fd = fd[["player_id", "season", "week", "position", "salary"]]

    sal = pd.concat([rg, fd], ignore_index=True).drop_duplicates(KEYS)
    sal[["season", "week"]] = sal[["season", "week"]].astype("int64")
    return sal


def expected_points(sal: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """Points a typical player at this salary scores, per position.

    Fit on EARLIER seasons only (season S uses S-1 and before), so the line
    never knows how this season turned out. 2016 has no earlier season and
    uses itself -- a small leak confined to a training-only season.
    """
    played = sal.merge(table[[*KEYS, "dk_points"]], on=KEYS)
    out = pd.Series(np.nan, index=sal.index)
    for season in sorted(sal["season"].unique()):
        hist = played[played["season"] < season]
        if hist.empty:
            hist = played[played["season"] == season]
        for pos, h in hist.groupby("position"):
            slope, intercept = np.polyfit(h["salary"], h["dk_points"], 1)
            rows = sal["season"].eq(season) & sal["position"].eq(pos)
            out[rows] = intercept + slope * sal.loc[rows, "salary"]
    return out


def salary_features(table: pd.DataFrame, sal: pd.DataFrame) -> pd.DataFrame:
    sal = sal.sort_values(["player_id", "season", "week"]).copy()
    sal["salary_implied_pts"] = expected_points(sal, table)
    g = sal.groupby(["player_id", "season"], sort=False)["salary"]
    sal["salary_chg_1w"] = sal["salary"] - g.shift(1)
    sal["salary_chg_3w"] = sal["salary"] - g.shift(3)
    # where DK ranks him at his position this week (1.0 = most expensive)
    sal["salary_pos_pct"] = sal.groupby(["season", "week", "position"])["salary"].rank(pct=True)

    cur = ["salary", "salary_implied_pts", "salary_chg_1w", "salary_chg_3w", "salary_pos_pct"]
    f = table[KEYS].merge(sal[[*KEYS, *cur]], on=KEYS, how="left").set_index(table.index)[cur]

    # value history: how far he beat or missed the salary line in games he played
    played = sal.merge(table[[*KEYS, "dk_points"]], on=KEYS).sort_values(["player_id", "season", "week"])
    played["value"] = played["dk_points"] - played["salary_implied_pts"]
    played["beat"] = (played["value"] >= BEAT_BY).astype(int)
    gp = played.groupby(["player_id", "season"], sort=False)
    played["value_last"] = played["value"]
    played["value_r3"] = gp["value"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    # consecutive games beating the line, counting this game (0 once a streak breaks)
    played["value_streak"] = gp["beat"].transform(
        lambda s: s.groupby((s == 0).cumsum()).cumsum())
    hist_cols = ["value_last", "value_r3", "value_streak"]
    f[hist_cols] = _as_of_previous_game(played, table, hist_cols).values
    return f


# --------------------------------------------------------- 2. carry-over
def carryover_features(table: pd.DataFrame, weekly: pd.DataFrame, f_salary: pd.DataFrame) -> pd.DataFrame:
    """Last season's production, plus an average that blends last season with this one."""
    w = weekly[weekly["season_type"].eq("REG")][[*KEYS, "recent_team", "targets", "carries",
                                                  "target_share", "air_yards_share"]]
    t = table[[*KEYS, "dk_points"]].merge(w, on=KEYS, how="left").sort_values(["player_id", "season", "week"])

    last4 = t.groupby(["player_id", "season"]).tail(4).groupby(["player_id", "season"])["dk_points"].mean()
    ps = t.groupby(["player_id", "season"]).agg(
        ps_games=("dk_points", "size"), ps_dkp_pg=("dk_points", "mean"),
        ps_targets_pg=("targets", "mean"), ps_carries_pg=("carries", "mean"),
        ps_tgt_share=("target_share", "mean"), ps_ay_share=("air_yards_share", "mean"),
        ps_team=("recent_team", "last"))
    ps["ps_last4_dkp_pg"] = last4
    ps = ps.reset_index()
    ps["season"] = ps["season"] + 1  # last season's numbers describe THIS season

    f = table[KEYS + ["recent_team"]].merge(ps, on=["player_id", "season"], how="left")
    f["changed_team"] = np.where(f["ps_team"].isna(), np.nan, (f["ps_team"] != f["recent_team"]).astype(float))

    # blended average: last season acts like PRIOR_GAMES games of evidence, then
    # each game played this season adds one more. Early on it leans on last
    # season; by midseason it's mostly this season. With no last season, DK's
    # salary line stands in as the prior.
    g = t.groupby(["player_id", "season"], sort=False)["dk_points"]
    prev_sum = g.transform(lambda s: s.shift(1).cumsum()).fillna(0).values
    prev_n = g.cumcount().values
    order = t.index  # t is sorted; bring the arrays back to table order
    sums = pd.Series(prev_sum, index=order).reindex(table.index).values
    counts = pd.Series(prev_n, index=order).reindex(table.index).values
    prior = f["ps_dkp_pg"].fillna(pd.Series(f_salary["salary_implied_pts"].values)).values
    f["dkp_blended"] = np.where(np.isnan(prior),
                                np.where(counts > 0, sums / np.maximum(counts, 1), np.nan),
                                (PRIOR_GAMES * prior + sums) / (PRIOR_GAMES + counts))

    cols = ["ps_games", "ps_dkp_pg", "ps_last4_dkp_pg", "ps_targets_pg", "ps_carries_pg",
            "ps_tgt_share", "ps_ay_share", "changed_team", "dkp_blended"]
    return f.set_index(table.index)[cols]


# ------------------------------------------------------------ 3. profile
def profile_features(table: pd.DataFrame) -> pd.DataFrame:
    """Draft capital, age, experience, and this week's depth chart slot."""
    dp = pd.read_parquet(RAW / "players" / "draft_picks.parquet").dropna(subset=["gsis_id"])
    dp = dp.sort_values("season").drop_duplicates("gsis_id")[["gsis_id", "season", "round", "pick"]]
    dp.columns = ["player_id", "draft_year", "draft_round", "draft_pick"]

    ids = pd.read_parquet(RAW / "ids.parquet")[["gsis_id", "birthdate"]].dropna().drop_duplicates("gsis_id")
    ids = ids.rename(columns={"gsis_id": "player_id"})

    f = table[KEYS].merge(dp, on="player_id", how="left").merge(ids, on="player_id", how="left")
    f["years_pro"] = f["season"] - f["draft_year"]
    f["is_rookie"] = np.where(f["draft_year"].isna(), np.nan, (f["years_pro"] == 0).astype(float))
    season_start = pd.to_datetime(f["season"].astype(str) + "-09-01")
    f["age"] = (season_start - pd.to_datetime(f["birthdate"], errors="coerce")).dt.days / 365.25

    raw = pd.read_parquet(RAW / "depth_charts.parquet")
    # 2016-2024: one chart per team per week; depth_team 1/2/3 = starter/backup/third
    dc = raw[raw["season"].notna() & raw["game_type"].eq("REG") & raw["formation"].eq("Offense")
             & raw["position"].isin([*SKILL, "FB"])]
    dc = dc.assign(season=dc["season"].astype("int64"), week=dc["week"].astype("int64"),
                   depth_rank=pd.to_numeric(dc["depth_team"], errors="coerce"))
    weekly_depth = dc.rename(columns={"gsis_id": "player_id"}).groupby(KEYS)["depth_rank"].min().reset_index()

    # 2025 on: DAILY snapshots with no season/week. pos_rank is the same idea
    # (rank within his slot). Use the last snapshot taken before game day.
    daily = raw[raw["season"].isna() & raw["pos_abb"].isin([*SKILL, "FB"])].copy()
    daily["date"] = pd.to_datetime(daily["dt"], utc=True).dt.tz_convert(None).dt.normalize()
    sched = pd.read_parquet(RAW / "schedules.parquet")
    sched = sched[sched["game_type"].eq("REG")]
    games = pd.concat([sched.assign(team=sched["home_team"]), sched.assign(team=sched["away_team"])])
    games = games.assign(gameday=pd.to_datetime(games["gameday"]))[["season", "week", "team", "gameday"]]
    shots = daily[["team", "date"]].drop_duplicates().sort_values("date")
    games = pd.merge_asof(games.sort_values("gameday"), shots, left_on="gameday", right_on="date",
                          by="team", allow_exact_matches=False).dropna(subset=["date"])
    daily_depth = (daily.drop(columns=["season", "week"])  # blank in this format
                        .merge(games[["season", "week", "team", "date"]], on=["team", "date"])
                        .rename(columns={"gsis_id": "player_id"})
                        .groupby(KEYS)["pos_rank"].min().rename("depth_rank").reset_index())
    daily_depth[["season", "week"]] = daily_depth[["season", "week"]].astype("int64")

    depth = pd.concat([weekly_depth, daily_depth]).drop_duplicates(KEYS)
    f = f.merge(depth, on=KEYS, how="left")

    cols = ["draft_round", "draft_pick", "years_pro", "is_rookie", "age", "depth_rank"]
    return f.set_index(table.index)[cols]


# ------------------------------------------------------------------ build
def build(table: pd.DataFrame | None = None, weekly: pd.DataFrame | None = None,
          extra_salaries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Frames can be passed in (src/predict_week.py). `extra_salaries` adds live
    DK prices (player_id, season, week, position, salary) for the coming week."""
    table = pd.read_parquet(PROC / "model_table_v2.parquet") if table is None else table.copy()
    weekly = pd.read_parquet(RAW / "weekly.parquet") if weekly is None else weekly.copy()
    for df in (table, weekly):
        df[["season", "week"]] = df[["season", "week"]].astype("int64")

    sal = salary_table(table)
    if extra_salaries is not None and len(extra_salaries):
        extra = extra_salaries.astype({"season": "int64", "week": "int64"})
        sal = pd.concat([sal, extra], ignore_index=True).drop_duplicates(KEYS, keep="last")
    f_sal = salary_features(table, sal)
    batches = {
        "salary": f_sal,
        "carry-over": carryover_features(table, weekly, f_sal),
        "profile": profile_features(table),
    }
    for name, f in batches.items():
        filled = f.notna().mean() * 100
        print(f"  {name:<10} {f.shape[1]:>2} features, {filled.mean():4.0f}% filled "
              f"(lowest: {filled.idxmin()} {filled.min():.0f}%)")
    return pd.concat([table, *batches.values()], axis=1)


if __name__ == "__main__":
    out = build()
    out.to_parquet(PROC / "model_table_v3.parquet", index=False)
    print(f"model_table_v3: {out.shape[0]:,} rows x {out.shape[1]} cols")
