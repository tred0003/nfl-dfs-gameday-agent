"""Feature batch 2: injuries, routes run, red-zone usage.

Adds three families of features to the base model table and writes
data/processed/model_table_v2.parquet. The base table is left untouched, so the
old and new models can be compared on exactly the same rows.

  1. INJURIES / VACATED OPPORTUNITY  (data/raw/injuries.parquet)
     When a teammate is ruled Out, how much of the team's targets and carries
     is he leaving behind, and where does this player rank among the healthy?
  2. ROUTES RUN  (data/raw/pbp + data/raw/participation)
     Share of the team's dropbacks this player was on the field for -- a free
     stand-in for "routes run" -- and targets per route.
  3. RED ZONE  (data/raw/pbp)
     Share of the team's targets and carries close to the end zone, where
     touchdowns (the biggest chunk of DK points) come from.

LEAKAGE RULE: every feature for week N is built only from games BEFORE week N.
Injury designations are the exception -- the final report comes out before
kickoff, so the current week's report is fair game. Whether a Questionable
player actually suited up is NOT used: that was only known on Sunday.

Run: python src/features_extra.py      (after src/features.py)
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# the injury file still uses the old codes for two relocated teams
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
KEYS = ["player_id", "season", "week"]


# ------------------------------------------------------------------ helpers
def _as_of_previous_game(state: pd.DataFrame, rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Attach each player's `cols` as they stood after his last game BEFORE `week`.

    `state` has one row per player-game with running values that INCLUDE that
    game. Looking up the latest row strictly before the target week is what
    keeps next week's information out. This also handles weeks a player
    missed: the lookup simply reaches back to his last game that season.
    """
    s = (state[["player_id", "season", "week", *cols]]
         .rename(columns={"week": "_wk"})
         .sort_values("_wk"))
    r = rows[KEYS].copy()
    r["_order"] = np.arange(len(r))
    out = pd.merge_asof(
        r.sort_values("week"), s,
        left_on="week", right_on="_wk", by=["player_id", "season"],
        allow_exact_matches=False,  # strictly earlier weeks only
    )
    return out.sort_values("_order").set_index(rows.index)[cols]


def _rolling_state(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    """Running averages of `col` through each game (within player-season)."""
    g = df.groupby(["player_id", "season"], sort=False)[col]
    df[f"{name}_r3"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df[f"{name}_r5"] = g.transform(lambda s: s.rolling(5, min_periods=2).mean())
    df[f"{name}_ewm"] = g.transform(lambda s: s.ewm(halflife=3).mean())
    df[f"{name}_last"] = df[col]
    return df


def _load_pbp(columns: list[str]) -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "pbp" / "play_by_play_*.parquet")))
    p = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    p = p[p["season_type"].eq("REG")].copy()
    p["season"] = p["season"].astype("int64")
    p["week"] = p["week"].astype("int64")
    return p


# --------------------------------------------------------- 1. injuries
def injury_features(table: pd.DataFrame, weekly: pd.DataFrame,
                    override: pd.DataFrame | None = None) -> pd.DataFrame:
    """Vacated target/carry share from teammates ruled Out or Doubtful.

    Each injured player's share is his recent (recency-weighted) share through
    his last game before the injury week. Long-term IR players are not on the
    weekly report, which is fine: their absence is already baked into
    everyone's recent numbers.
    """
    w = weekly[weekly["season_type"].eq("REG")].copy()
    team_tot = w.groupby(["recent_team", "season", "week"])[["targets", "carries"]].transform("sum")
    w["tgt_sh"] = w["targets"] / team_tot["targets"].replace(0, np.nan)
    w["car_sh"] = w["carries"] / team_tot["carries"].replace(0, np.nan)
    w = w.sort_values(["player_id", "season", "week"])
    g = w.groupby(["player_id", "season"], sort=False)
    w["tgt_sh_ewm"] = g["tgt_sh"].transform(lambda s: s.ewm(halflife=3).mean())
    w["car_sh_ewm"] = g["car_sh"].transform(lambda s: s.ewm(halflife=3).mean())
    w["att_ewm"] = g["attempts"].transform(lambda s: s.ewm(halflife=3).mean())
    w["group"] = w["position"].map({"WR": "rec", "TE": "rec", "RB": "rb", "QB": "qb"})

    inj = pd.read_parquet(RAW / "injuries.parquet")
    inj = inj[inj["game_type"].eq("REG")].copy()
    inj["team"] = inj["team"].replace(TEAM_FIX)
    inj["season"] = inj["season"].astype("int64")
    inj["week"] = inj["week"].astype("int64")
    inj = inj.rename(columns={"gsis_id": "player_id"})
    if override is not None and len(override):
        # live statuses (e.g. DK's OUT/Q flags) replace the report for those weeks
        weeks = override[["season", "week"]].drop_duplicates()
        inj = inj.merge(weeks.assign(_drop=1), on=["season", "week"], how="left")
        inj = pd.concat([inj[inj["_drop"].isna()].drop(columns="_drop"), override], ignore_index=True)

    out = inj[inj["report_status"].isin(["Out", "Doubtful"])][["player_id", "season", "week", "team"]]
    out = out.drop_duplicates(KEYS)
    # "new" = not on last week's Out/Doubtful list -- first week of the absence,
    # before salaries and rolling averages have caught up
    prev = out.assign(week=out["week"] + 1)[KEYS].assign(was_out=1)
    out = out.merge(prev, on=KEYS, how="left")
    out["is_new"] = out["was_out"].isna().astype(int)

    state = w[["player_id", "season", "week", "tgt_sh_ewm", "car_sh_ewm", "att_ewm", "group"]]
    out = out.join(_as_of_previous_game(state, out, ["tgt_sh_ewm", "car_sh_ewm", "att_ewm", "group"]))
    out = out.dropna(subset=["group"])  # never played this season -> nothing vacated
    out[["tgt_sh_ewm", "car_sh_ewm"]] = out[["tgt_sh_ewm", "car_sh_ewm"]].fillna(0)
    out["new_tgt"] = out["tgt_sh_ewm"] * out["is_new"]

    team = ["team", "season", "week"]
    vac = out.groupby(team).agg(vacated_tgt_share=("tgt_sh_ewm", "sum"),
                                vacated_carry_share=("car_sh_ewm", "sum"),
                                new_vacated_tgt_share=("new_tgt", "sum"),
                                n_out=("player_id", "size")).reset_index()
    by_group = (out.groupby([*team, "group"])["tgt_sh_ewm"].sum()
                   .unstack("group").reindex(columns=["rec", "rb"]).fillna(0)
                   .rename(columns={"rec": "vac_tgt_rec", "rb": "vac_tgt_rb"}).reset_index())
    # starting QB out = the team's highest-volume passer is on the Out list
    qb = out[out["group"].eq("qb")].groupby(team)["att_ewm"].max().rename("out_qb_att").reset_index()

    t = table.rename(columns={"recent_team": "team"})[[*KEYS, "team", "position"]]
    t = (t.merge(vac, on=team, how="left")
          .merge(by_group, on=team, how="left")
          .merge(qb, on=team, how="left"))
    t[["vacated_tgt_share", "vacated_carry_share", "new_vacated_tgt_share", "n_out",
       "vac_tgt_rec", "vac_tgt_rb"]] = t[["vacated_tgt_share", "vacated_carry_share",
                                          "new_vacated_tgt_share", "n_out",
                                          "vac_tgt_rec", "vac_tgt_rb"]].fillna(0)
    t["starting_qb_out"] = (t["out_qb_att"].fillna(0) >= 15).astype(int)
    t["vacated_same_group"] = np.where(t["position"].isin(["WR", "TE"]), t["vac_tgt_rec"],
                                       np.where(t["position"].eq("RB"), t["vac_tgt_rb"], 0.0))

    # a player listed Doubtful who plays anyway must not count his own share as "vacated"
    # (matched on team too: after a trade the injury file and box score can disagree)
    me = out[KEYS + ["team", "tgt_sh_ewm", "car_sh_ewm", "new_tgt"]].rename(
        columns={"tgt_sh_ewm": "me_tgt", "car_sh_ewm": "me_car", "new_tgt": "me_new"})
    t = t.merge(me, on=[*KEYS, "team"], how="left")
    listed = t["me_tgt"].notna()
    t[["me_tgt", "me_car", "me_new"]] = t[["me_tgt", "me_car", "me_new"]].fillna(0)
    t["vacated_tgt_share"] -= t["me_tgt"]
    t["vacated_carry_share"] -= t["me_car"]
    t["new_vacated_tgt_share"] -= t["me_new"]
    t["vacated_same_group"] -= np.where(t["position"].isin(["WR", "TE", "RB"]), t["me_tgt"], 0.0)
    t["n_out"] -= listed.astype(int)

    # the player's own designation this week
    own = inj[KEYS + ["report_status", "practice_status"]].drop_duplicates(KEYS)
    t = t.merge(own, on=KEYS, how="left")
    t["own_questionable"] = t["report_status"].eq("Questionable").astype(int)
    t["own_limited_practice"] = t["practice_status"].fillna("").str.contains("Limited|Did Not").astype(int)

    # rank among healthy teammates at the same position group, by recent target share
    t["tgt_sh_ewm"] = _as_of_previous_game(state, t, ["tgt_sh_ewm"])["tgt_sh_ewm"].values
    grp = t["position"].map({"WR": "rec", "TE": "rec", "RB": "rb", "QB": "qb"})
    t["healthy_tgt_rank"] = t.groupby(["team", "season", "week", grp])["tgt_sh_ewm"].rank(
        ascending=False, method="min")

    cols = ["vacated_tgt_share", "vacated_carry_share", "new_vacated_tgt_share", "n_out",
            "vacated_same_group", "starting_qb_out", "own_questionable",
            "own_limited_practice", "healthy_tgt_rank"]
    return t.set_index(table.index)[cols]


# --------------------------------------------------------- 2. routes run
def route_features(table: pd.DataFrame) -> pd.DataFrame:
    """Share of team dropbacks a player was on the field for, and targets per route.

    A dropback is any called pass: completions, incompletions, sacks and
    scrambles. Checked against FantasyData's official 2025 route participation
    for six top WRs: within 1-4 points. Looser for TEs and RBs, who are
    sometimes on the field for a pass play but staying in to block.
    """
    pbp = _load_pbp(["game_id", "play_id", "season", "week", "season_type", "posteam",
                     "qb_dropback", "receiver_player_id", "two_point_attempt"])
    drop = pbp[pbp["qb_dropback"].eq(1) & pbp["two_point_attempt"].ne(1)].copy()
    drop["play_id"] = drop["play_id"].astype("int64")

    parts = []
    for f in sorted(glob.glob(str(RAW / "participation" / "pbp_participation_*.parquet"))):
        p = pd.read_parquet(f, columns=["nflverse_game_id", "play_id", "offense_players"])
        p["play_id"] = p["play_id"].astype("int64")
        parts.append(p)
    part = pd.concat(parts, ignore_index=True).rename(columns={"nflverse_game_id": "game_id"})

    d = drop.merge(part, on=["game_id", "play_id"], how="inner")
    team_drops = d.groupby(["posteam", "season", "week"]).size().rename("team_dropbacks")

    on_field = (d.assign(player_id=d["offense_players"].str.split(";"))
                  .explode("player_id")
                  .groupby(["player_id", "posteam", "season", "week"]).size()
                  .rename("routes").reset_index())
    targets = (d.dropna(subset=["receiver_player_id"])
                 .groupby(["receiver_player_id", "season", "week"]).size()
                 .rename("route_targets").reset_index()
                 .rename(columns={"receiver_player_id": "player_id"}))

    s = (on_field.merge(team_drops.reset_index(), on=["posteam", "season", "week"])
                 .merge(targets, on=KEYS, how="left"))
    s["route_targets"] = s["route_targets"].fillna(0)
    s["route_share"] = s["routes"] / s["team_dropbacks"]
    s = s.sort_values(["player_id", "season", "week"])
    s = _rolling_state(s, "route_share", "route_share")
    g = s.groupby(["player_id", "season"], sort=False)
    # targets per route over the last 5 games: totals, not an average of ratios
    s["tprr_r5"] = (g["route_targets"].transform(lambda x: x.rolling(5, min_periods=2).sum())
                    / g["routes"].transform(lambda x: x.rolling(5, min_periods=2).sum()))

    cols = ["route_share_r3", "route_share_r5", "route_share_ewm", "route_share_last", "tprr_r5"]
    f = _as_of_previous_game(s, table, cols)
    # role change: last game vs his recent norm (a jump shows up here first)
    f["route_share_trend"] = f["route_share_last"] - f["route_share_r5"]
    return f


# --------------------------------------------------------- 3. red zone
def redzone_features(table: pd.DataFrame) -> pd.DataFrame:
    """Share of team targets and carries near the goal line.

    yardline_100 = yards from the opponent's end zone. Inside the 20 is the
    "red zone"; inside the 10 and 5 is where most rushing TDs happen. An
    end-zone target is a pass thrown at least as far as the goal line.
    """
    p = _load_pbp(["season", "week", "season_type", "posteam", "pass_attempt", "rush_attempt",
                   "sack", "receiver_player_id", "rusher_player_id", "yardline_100",
                   "air_yards", "two_point_attempt"])
    p = p[p["two_point_attempt"].ne(1)]
    team = ["posteam", "season", "week"]

    tg = p[p["pass_attempt"].eq(1) & p["sack"].ne(1) & p["receiver_player_id"].notna()].copy()
    tg["rz"] = tg["yardline_100"].le(20)
    tg["ez"] = tg["air_yards"].ge(tg["yardline_100"])
    ru = p[p["rush_attempt"].eq(1) & p["rusher_player_id"].notna()].copy()
    ru["i10"] = ru["yardline_100"].le(10)
    ru["i5"] = ru["yardline_100"].le(5)

    tt = tg.groupby(team)[["rz", "ez"]].sum().rename(columns={"rz": "team_rz_tgt", "ez": "team_ez_tgt"})
    rt = ru.groupby(team)[["i10", "i5"]].sum().rename(columns={"i10": "team_i10", "i5": "team_i5"})
    pt = (tg.groupby(["receiver_player_id", *team])[["rz", "ez"]].sum().reset_index()
            .rename(columns={"receiver_player_id": "player_id"}))
    pr = (ru.groupby(["rusher_player_id", *team])[["i10", "i5"]].sum().reset_index()
            .rename(columns={"rusher_player_id": "player_id"}))

    s = pt.merge(pr, on=["player_id", *team], how="outer").fillna(0)
    s = s.merge(tt.reset_index(), on=team, how="left").merge(rt.reset_index(), on=team, how="left")
    s = s.groupby(KEYS, as_index=False).sum(numeric_only=True)  # traded mid-week edge case
    s = s.sort_values(["player_id", "season", "week"])
    g = s.groupby(["player_id", "season"], sort=False)
    # shares over the last 5 games: player total / team total, so one quiet
    # week with zero red-zone trips doesn't divide by zero
    roll = lambda c: g[c].transform(lambda x: x.rolling(5, min_periods=1).sum())  # noqa: E731
    s["rz_tgt_share_r5"] = roll("rz") / roll("team_rz_tgt").replace(0, np.nan)
    s["ez_tgt_share_r5"] = roll("ez") / roll("team_ez_tgt").replace(0, np.nan)
    s["i10_carry_share_r5"] = roll("i10") / roll("team_i10").replace(0, np.nan)
    s["i5_carry_share_r5"] = roll("i5") / roll("team_i5").replace(0, np.nan)
    s["rz_tgt_r5"] = g["rz"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    s["i10_carries_r5"] = g["i10"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    cols = ["rz_tgt_share_r5", "ez_tgt_share_r5", "i10_carry_share_r5", "i5_carry_share_r5",
            "rz_tgt_r5", "i10_carries_r5"]
    return _as_of_previous_game(s, table, cols)


# ------------------------------------------------------------------ build
def build(table: pd.DataFrame | None = None, weekly: pd.DataFrame | None = None,
          injury_override: pd.DataFrame | None = None) -> pd.DataFrame:
    """Frames can be passed in (src/predict_week.py) instead of read from disk."""
    table = pd.read_parquet(PROC / "model_table.parquet") if table is None else table.copy()
    weekly = pd.read_parquet(RAW / "weekly.parquet") if weekly is None else weekly.copy()
    for df in (table, weekly):
        df[["season", "week"]] = df[["season", "week"]].astype("int64")

    batches = {
        "injuries": injury_features(table, weekly, injury_override),
        "routes": route_features(table),
        "red zone": redzone_features(table),
    }
    for name, f in batches.items():
        filled = f.notna().mean().mean() * 100
        print(f"  {name:<9} {f.shape[1]:>2} features, {filled:4.0f}% filled")
    return pd.concat([table, *batches.values()], axis=1)


if __name__ == "__main__":
    out = build()
    out.to_parquet(PROC / "model_table_v2.parquet", index=False)
    print(f"model_table_v2: {out.shape[0]:,} rows x {out.shape[1]} cols")
