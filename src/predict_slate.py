"""Turn a DK slate snapshot into projections the optimizer can consume.

Two projection sources, because Week 1 and Week 8 are genuinely different problems:

  --source model    Uses the trained XGBoost model. Needs in-season form
                    features, so it is valid from roughly Week 4 onward.

  --source dk_vegas Week 1 fallback. Every rolling feature is empty before any
                    games are played, so the model has nothing to work with.
                    Instead: take DK's own FPPG (their recent-form number,
                    shipped in the salary feed) and adjust it by how many points
                    Vegas expects the player's team to score. Transparent, and
                    strictly more information than DK's number alone.

Being explicit about which source produced a projection matters more than the
projection itself -- a Week 1 lineup should not be presented as if the model
built it.

Run: python src/predict_slate.py --slate <players_*.parquet> --source dk_vegas
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# league-average implied team total; the scaler is relative to this
LEAGUE_AVG_IMPLIED = 22.5
# how hard to let Vegas move a projection (cap the multiplier)
VEGAS_CLIP = (0.80, 1.25)
# An availability tag is real information, and DK does not cut the salary enough
# to compensate. Doubtful is much closer to Out than to Questionable -- those
# players usually do not suit up.
STATUS_DISCOUNT = {"Q": 0.85, "D": 0.45}

TEAM_FIXES = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS", "JAX": "JAX"}


def normalize_name(s: pd.Series) -> pd.Series:
    """DK and nflverse disagree on punctuation, suffixes and Jr./III."""
    out = (s.fillna("").str.lower()
            .str.replace(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", regex=True)
            .str.replace(r"[^a-z ]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip())
    return out


def vegas_for_slate(season: int, week: int) -> pd.DataFrame:
    """Implied team total per team for this week."""
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[s["season"].eq(season) & s["week"].eq(week)]
    if s.empty:
        import nfl_data_py as nfl
        s = nfl.import_schedules([season])
        s = s[s["week"].eq(week)]

    rows = []
    for side, sign in (("home", 1), ("away", -1)):
        rows.append(pd.DataFrame({
            "team": s[f"{side}_team"],
            "implied_team_total": s["total_line"] / 2 + sign * s["spread_line"] / 2,
            "implied_opp_total": s["total_line"] / 2 - sign * s["spread_line"] / 2,
            "total_line": s["total_line"],
            "team_spread": sign * s["spread_line"],
            "roof": s["roof"],
            "wind": s["wind"],
        }))
    return pd.concat(rows, ignore_index=True)


def project_dk_vegas(slate: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Week-1-safe projection: DK's FPPG scaled by the Vegas game environment."""
    d = slate.copy()
    d["team"] = d["team"].replace(TEAM_FIXES)

    v = vegas_for_slate(season, week)
    v["team"] = v["team"].replace(TEAM_FIXES)
    d = d.merge(v, on="team", how="left")

    scaler = (d["implied_team_total"] / LEAGUE_AVG_IMPLIED).clip(*VEGAS_CLIP)
    d["vegas_scaler"] = scaler.fillna(1.0)

    base = pd.to_numeric(d["dk_fppg"], errors="coerce")
    # players with no FPPG history (rookies, new starters) get a position-median
    base = base.fillna(base.groupby(d["position"]).transform("median")).fillna(0.0)

    d["status_discount"] = d["status"].map(STATUS_DISCOUNT).fillna(1.0)
    d["projection"] = (base * d["vegas_scaler"] * d["status_discount"]).round(2)
    d["proj_source"] = "dk_fppg x vegas"

    # the optimizer needs each player's opponent to avoid rostering a defense
    # against its own offense
    game = d["game"].fillna("")
    away = game.str.split(" @ ").str[0].str.strip()
    home = game.str.split(" @ ").str[1].str.strip()
    d["opp_team"] = np.where(d["team"].eq(home), away, home)

    # anyone ruled out cannot be rostered
    out_mask = d["status"].isin(["OUT", "IR"]) | d.get("is_disabled", False).fillna(False)
    d.loc[out_mask, "projection"] = np.nan
    d.loc[out_mask, "proj_source"] = "excluded (OUT/IR)"
    return d


def project_model(slate: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Model-based projection. Requires in-season features to exist."""
    from features import FEATURE_COLS  # noqa: F401
    raise NotImplementedError(
        "model projections need in-season features -- valid from ~Week 4. "
        "Use --source dk_vegas for Week 1."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slate", required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--source", choices=["dk_vegas", "model"], default="dk_vegas")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    slate = pd.read_parquet(a.slate)
    fn = {"dk_vegas": project_dk_vegas, "model": project_model}[a.source]
    d = fn(slate, a.season, a.week)

    out = Path(a.out) if a.out else Path(a.slate).with_name(
        Path(a.slate).stem.replace("players", "projections") + ".parquet")
    d.to_parquet(out, index=False)

    playable = d["projection"].notna().sum()
    print(f"{len(d)} players, {playable} playable ({len(d) - playable} OUT/IR excluded)")
    print(f"source: {a.source}  ->  {out.name}\n")
    print("top 12 projections:")
    print(d.nlargest(12, "projection")[
        ["name", "position", "team", "salary", "dk_fppg", "vegas_scaler",
         "projection", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
