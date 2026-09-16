"""Live weekly projections + lineup from the v3 model.

The backtest scripts only ever predicted games that had already been played.
This one predicts a week that hasn't happened:

  1. Read the DraftKings slate: DK's own "export players" CSV (DKSalaries.csv),
     or an older data/slates parquet snapshot.
  2. Match each DK player to his NFL player ID.
  3. Add a blank "future" row per player for the target week. Every feature
     is shifted one game, so these rows get features from completed games only.
  4. Run the same three feature builders the backtest used, in memory.
  5. Train v3 on every completed game BEFORE the target week, predict the
     future rows, and hand the projections to the optimizer.

Defenses still have no model: DK's FPPG scaled by the Vegas line.

Run (Week 2 -- slate = FantasyData salary CSV or DK "export players" CSV):
  python src/predict_week.py --season 2026 --week 2 --slate data/raw/fantasydata/salaries/dk_2026_wk02.csv ^
      --prev-slate data/slates/dg151307/players_20260910T190407Z.parquet --prev-week 1

Check it on a finished week (trains only on earlier games, then scores actuals):
  python src/predict_week.py --season 2026 --week 1 ^
      --slate data/slates/dg151307/players_20260910T190407Z.parquet --score
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import features  # noqa: E402
import features_carry  # noqa: E402
import features_extra  # noqa: E402
import value_board  # noqa: E402
from optimize import optimize, show  # noqa: E402
from predict_slate import project_dk_vegas  # noqa: E402
from train import PARAMS, TARGET  # noqa: E402

RAW = ROOT / "data" / "raw"
LINEUPS = ROOT / "data" / "lineups"
SKILL = ["QB", "RB", "WR", "TE"]
TEAM_FIX = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
ID_COLS = {"player_id", "player_display_name", "position", "recent_team", "opponent_team",
           "season", "week", TARGET, "_future"}
DK_STATUS_TO_REPORT = {"OUT": "Out", "O": "Out", "IR": "Out", "D": "Doubtful", "Q": "Questionable"}


def _norm(name: str) -> str:
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(name).lower())
    return re.sub(r"[^a-z]", "", name)


# ------------------------------------------------------------------ slate
def load_slate(path: str, season: int | None = None, week: int | None = None) -> pd.DataFrame:
    """Standard columns: name, position, team, opp_team, game, salary, status, dk_fppg, fd_projection.

    Three formats:
      - DraftKings "export players" CSV (DKSalaries.csv) -- one specific DK slate
      - FantasyData salary-tool CSV -- DK salaries for EVERY game that week, plus
        FantasyData's projection; trimmed here to the Sunday main slate
      - data/slates parquet snapshot from dk_client.py
    """
    p = Path(path)
    cols = pd.read_csv(p, nrows=0).columns if p.suffix == ".csv" else []
    if "salary_draftkings" in cols:  # FantasyData
        d = pd.read_csv(p).rename(columns={"player": "name", "pos": "position",
                                           "salary_draftkings": "salary", "fpts_draftkings": "fd_projection"})
        d = d[d["position"].isin([*SKILL, "DST"])]
        for c in ("team", "opp"):
            d[c] = d[c].replace(TEAM_FIX)
        sched = pd.read_parquet(RAW / "schedules.parquet")
        sched = sched[sched["season"].eq(season) & sched["week"].eq(week) & sched["game_type"].eq("REG")
                      & sched["weekday"].eq("Sunday") & sched["gametime"].lt("19:00")]  # main slate
        games = pd.concat([sched.assign(team=sched["home_team"]), sched.assign(team=sched["away_team"])])
        games["game"] = games["away_team"] + " @ " + games["home_team"]
        d = d.merge(games[["team", "game"]], on="team")
        d["status"], d["dk_fppg"] = None, np.nan
    elif p.suffix == ".csv":  # DraftKings export
        d = pd.read_csv(p).rename(columns={"Name": "name", "Position": "position", "Salary": "salary",
                                           "TeamAbbrev": "team", "AvgPointsPerGame": "dk_fppg"})
        matchup = d["Game Info"].str.extract(r"^(\w+)@(\w+)")
        d["game"] = matchup[0] + " @ " + matchup[1]
        d["status"] = None  # the export has no injury flags
    else:  # data/slates snapshot from dk_client.py
        d = pd.read_parquet(p).drop_duplicates("player_dk_id")
    if "fd_projection" not in d.columns:
        d["fd_projection"] = np.nan
    d["name"] = d["name"].str.strip()
    d["team"] = d["team"].replace(TEAM_FIX)
    teams = d["game"].str.split(" @ ", expand=True).replace(TEAM_FIX)
    d["game"] = teams[0] + " @ " + teams[1]
    d["opp_team"] = np.where(d["team"].eq(teams[1]), teams[0], teams[1])
    return d[["name", "position", "team", "opp_team", "game", "salary", "status", "dk_fppg",
              "fd_projection"]].reset_index(drop=True)


def match_player_ids(slate: pd.DataFrame, weekly: pd.DataFrame, season: int) -> pd.DataFrame:
    """DK name -> NFL player ID. Name + position first; team breaks ties."""
    ids = pd.read_parquet(RAW / "ids.parquet")[["gsis_id", "name", "position", "team"]].dropna(subset=["gsis_id"])
    ids = ids.rename(columns={"gsis_id": "player_id", "team": "id_team"})
    recent = (weekly[weekly["season"].between(season - 1, season)]
              [["player_id", "player_display_name", "position", "recent_team"]]
              .drop_duplicates("player_id")
              .rename(columns={"player_display_name": "name", "recent_team": "id_team"}))
    book = pd.concat([recent, ids], ignore_index=True).drop_duplicates("player_id")
    book["nk"] = book["name"].map(_norm)
    book["id_team"] = book["id_team"].replace(TEAM_FIX)

    s = slate[slate["position"].isin(SKILL)].copy()
    s["nk"] = s["name"].map(_norm)
    cand = s.merge(book[["nk", "position", "player_id", "id_team"]], on=["nk", "position"], how="left")
    cand["same_team"] = cand["id_team"].eq(cand["team"])
    cand = cand.sort_values("same_team", ascending=False).drop_duplicates(["name", "team", "position"])
    return cand.drop(columns=["nk", "id_team", "same_team"])


# ------------------------------------------------------------------ features
def build_rows(season: int, week: int, slate: pd.DataFrame,
               prev_slate: pd.DataFrame | None, prev_week: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly = pd.read_parquet(RAW / "weekly.parquet")
    weekly[["season", "week"]] = weekly[["season", "week"]].astype("int64")
    # nothing from the target week or later -- as if it were before kickoff
    past = weekly[(weekly["season"] < season) | ((weekly["season"] == season) & (weekly["week"] < week))]

    matched = match_player_ids(slate, past, season)
    unmatched = matched[matched["player_id"].isna()]
    matched = matched.dropna(subset=["player_id"])

    future = pd.DataFrame({
        "player_id": matched["player_id"].values, "player_display_name": matched["name"].values,
        "position": matched["position"].values, "position_group": matched["position"].values,
        "recent_team": matched["team"].values, "opponent_team": matched["opp_team"].values,
        "season": season, "week": week, "season_type": "REG", "_future": True,
    })
    weekly_aug = pd.concat([past.assign(_future=False), future], ignore_index=True)

    # live injury flags from the slate replace the report for the target week, if present
    override = None
    flagged = matched[matched["status"].isin(DK_STATUS_TO_REPORT)]
    if matched["status"].notna().any():
        override = pd.DataFrame({
            "player_id": flagged["player_id"], "season": season, "week": week,
            "team": flagged["team"], "game_type": "REG", "report_status": flagged["status"].map(DK_STATUS_TO_REPORT),
            "practice_status": None,
        })

    salaries = [matched.assign(season=season, week=week)[["player_id", "season", "week", "position", "salary"]]]
    if prev_slate is not None:
        pm = match_player_ids(prev_slate, past, season).dropna(subset=["player_id"])
        salaries.append(pm.assign(season=season, week=prev_week)[["player_id", "season", "week", "position", "salary"]])

    print("building features (base, batch 2, batch 3)...", flush=True)
    base = features.build(weekly_aug)
    keep = ["player_id", "player_display_name", "position", "recent_team", "opponent_team",
            "season", "week", TARGET, "_future"] + features.FEATURE_COLS
    base = base[[c for c in keep if c in base.columns]]
    v2 = features_extra.build(base, weekly_aug, override)
    v3 = features_carry.build(v2, weekly_aug, pd.concat(salaries, ignore_index=True))
    return v3, matched, unmatched


# ------------------------------------------------------------------ model
def project(table: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    table = pd.concat([table, pd.get_dummies(table["position"], prefix="pos")], axis=1)
    feats = [c for c in table.columns if c not in ID_COLS]
    done = (table[TARGET].notna() & ~table["_future"]
            & ((table["season"] < season) | ((table["season"] == season) & (table["week"] < week))))
    train = table[done]
    target = table[table["_future"] & table["season"].eq(season) & table["week"].eq(week)].copy()
    print(f"training v3 on {len(train):,} completed player-games ({len(feats)} features), "
          f"predicting {len(target)} players", flush=True)
    model = xgb.XGBRegressor(objective="reg:squarederror", **PARAMS)
    model.fit(train[feats], train[TARGET], verbose=False)
    target["projection"] = model.predict(target[feats])

    # Second model, different question. `projection` is the expected score --
    # what you want in cash games, where you only need to clear a line. This one
    # is the 85th percentile: how high can he go. Tournaments pay the tail, so
    # the two columns get read differently and both belong on the board.
    ceil = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.85, **PARAMS)
    ceil.fit(train[feats], train[TARGET], verbose=False)
    target["ceiling"] = ceil.predict(target[feats])
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--slate", required=True, help="DKSalaries.csv or data/slates parquet")
    ap.add_argument("--prev-slate", default=None, help="last week's slate, for salary-change features")
    ap.add_argument("--prev-week", type=int, default=None)
    ap.add_argument("--score", action="store_true", help="week already played: score the lineup")
    ap.add_argument("--board-top", type=int, default=12,
                    help="rows per position on the deal board")
    a = ap.parse_args()

    slate = load_slate(a.slate, a.season, a.week)
    prev = load_slate(a.prev_slate, a.season, a.prev_week) if a.prev_slate else None
    table, matched, unmatched = build_rows(a.season, a.week, slate, prev, a.prev_week)
    proj = project(table, a.season, a.week)

    skill = slate[slate["position"].isin(SKILL)].merge(
        matched[["name", "team", "position", "player_id"]], on=["name", "team", "position"], how="left")
    skill = skill.merge(proj[["player_id", "projection", "ceiling", "salary_implied_pts",
                              "dkp_blended", "depth_rank"]],
                        on="player_id", how="left")
    dst = slate[slate["position"].eq("DST")].copy()
    if dst["fd_projection"].notna().all():
        dst["projection"] = dst["fd_projection"]  # FantasyData file: use its DST projection
    else:
        dst = project_dk_vegas(dst, a.season, a.week)[slate.columns.tolist() + ["projection"]]
    pool = pd.concat([skill, dst], ignore_index=True)

    # ruled out: DK status if the slate has one, else the official injury report
    report = pd.read_parquet(RAW / "injuries.parquet")
    report = report[report["season"].eq(a.season) & report["week"].eq(a.week)
                    & report["report_status"].isin(["Out", "Doubtful"])]
    ruled_out = pool["status"].isin(["OUT", "O", "IR", "D"]) | pool["player_id"].isin(report["gsis_id"])
    pool.loc[ruled_out, "projection"] = np.nan
    print(f"injury report for week {a.week}: {len(report)} Out/Doubtful"
          + ("" if len(report) else " -- NOT PUBLISHED YET. Re-pull injuries and rerun before lock."))
    pool["value"] = pool["projection"] - pool["salary_implied_pts"]

    LINEUPS.mkdir(parents=True, exist_ok=True)
    stem = f"{a.season}_wk{a.week:02d}_v3"
    pool.sort_values("projection", ascending=False).to_csv(LINEUPS / f"{stem}_projections.csv", index=False)

    print(f"\n{int(pool['projection'].notna().sum())} players projected | "
          f"{len(unmatched)} DK players not matched to an NFL ID (no projection)")
    if len(unmatched):
        print("  unmatched:", ", ".join(unmatched["name"].head(12)), "..." if len(unmatched) > 12 else "")

    print("\ntop projected value (points above the salary line), by position:")
    top = pool.dropna(subset=["value"]).sort_values("value", ascending=False).groupby("position").head(3)
    print(top[["name", "position", "team", "salary", "projection", "salary_implied_pts", "value"]]
          .round(1).to_string(index=False))

    lineup = optimize(pool.dropna(subset=["projection"]), "projection")
    if not lineup:
        raise SystemExit("optimizer found no valid lineup")
    lu = lineup[0]
    lu.to_parquet(LINEUPS / f"{stem}.parquet", index=False)
    print("\n######## v3 LINEUP")
    show(lu, "projection")

    # the lineup is the optimizer's answer; the boards are what you need to
    # argue with it. Every override has a price -- these columns are the price.
    print("\n######## PER-POSITION DEAL BOARDS")
    boards = value_board.build(pool, lu, "projection")
    value_board.show(boards, top=a.board_top)
    xlsx = value_board.to_excel(boards, lu, LINEUPS / f"{stem}_board.xlsx")
    print(f"\nboard -> {xlsx}")

    # paper trade: the same optimizer on FantasyData's projection, for grading after the games
    fd_lu = None
    if pool["fd_projection"].notna().any():
        fd_pool = pool.dropna(subset=["fd_projection"]).copy()
        fd_pool.loc[fd_pool["projection"].isna() & fd_pool["position"].ne("DST"), "fd_projection"] = np.nan
        fd = optimize(fd_pool.dropna(subset=["fd_projection"]), "fd_projection")
        if fd:
            fd_lu = fd[0]
            fd_lu.to_parquet(LINEUPS / f"{a.season}_wk{a.week:02d}_fantasydata.parquet", index=False)
            print("\n######## FANTASYDATA LINEUP (same optimizer, for comparison)")
            show(fd_lu, "fd_projection")
            overlap = set(lu["name"]) & set(fd_lu["name"])
            print(f"    shared players: {len(overlap)} of 9 -- {', '.join(sorted(overlap))}")

    if a.score:
        from backtest import dst_actual_points  # noqa: E402
        import backtest
        weekly = pd.read_parquet(RAW / "weekly.parquet")
        wk = features.add_dk_points(weekly[(weekly["season"] == a.season) & (weekly["week"] == a.week)])
        backtest.SEASONS = [a.season]
        dpts = dst_actual_points()
        dpts = dpts[dpts["week"].eq(a.week)].set_index("team")["dst_points"]
        lu = lu.merge(wk[["player_id", "dk_points"]], on="player_id", how="left")
        is_dst = lu["position"].eq("DST")
        lu.loc[is_dst, "dk_points"] = lu.loc[is_dst, "team"].map(dpts).values
        lu["dk_points"] = lu["dk_points"].fillna(0)
        print(f"\nACTUAL: {lu['dk_points'].sum():.1f} points")
        print(lu[["name", "position", "team", "salary", "projection", "dk_points"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
