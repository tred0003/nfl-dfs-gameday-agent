"""Pre-flight checks: refuse to build a lineup from data that isn't there.

WHY THIS FILE EXISTS

Every failure this guards against was found on 2026-09-16 by auditing the live
serve path, and every one of them fails SILENTLY today. Nothing raises, nothing
warns, and a plausible-looking lineup comes out the other end:

  - `participation/` stops at 2025, so all six route features -- the #3 feature
    family, 9.7% of total attribution -- are 100% NaN for every 2026 row.
  - Vegas lines are only published a week or two ahead. For 2026 week 4 and
    beyond, `total_line` is 100% null, which takes `implied_team_total`,
    `implied_opp_total`, `team_spread` and both spread interactions with it.
    FEATURES.md calls that the workhorse family.
  - `injuries.parquet` currently holds 2026 week 1 only. The vacated-opportunity
    features `fillna(0)`, so "the report isn't published" is indistinguishable
    from "this team is perfectly healthy" -- and the model reads a hard 0 as
    confident good news rather than as missing information.
  - Season-reset rolling windows leave 29 of 79 features 100% NaN on every
    serve row in weeks 1-2, against 10-39% missing in training. The model's
    missing-value branches were fit on a different population than they are
    being applied to.

XGBoost never complains about any of this. A feature that is entirely NaN just
routes every row down the split's default direction, and the model returns a
confident number built on a third of the evidence it was trained with.

THE RULE HERE: a lineup you know is degraded is worth something. A lineup you
believe is fine when it isn't is worth less than nothing, because you will bet
on it. So FATAL checks stop the run before the optimizer is ever called, and
WARN checks print loudly and let it continue.

Run standalone to check a week before you build anything:
    python src/preflight.py --season 2026 --week 3
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

FATAL, WARN, OK = "FATAL", "WARN", "OK"

# A feature being NaN on a minority of rows is normal -- rookies have no prior
# season, week 1 has no rolling window. A feature that is NaN on EVERY row being
# scored is not missing data, it is an absent input, and the model was never
# trained to handle that.
ALL_NAN_FATAL = 0.999
HIGH_NAN_WARN = 0.60


class Result:
    def __init__(self) -> None:
        self.checks: list[tuple[str, str, str]] = []

    def add(self, level: str, name: str, detail: str) -> None:
        self.checks.append((level, name, detail))

    @property
    def fatal(self) -> list[tuple[str, str, str]]:
        return [c for c in self.checks if c[0] == FATAL]

    def report(self) -> bool:
        """Print every check. Returns True if it is safe to proceed."""
        width = max(len(n) for _, n, _ in self.checks) if self.checks else 10
        print("\n" + "=" * 72)
        print("PRE-FLIGHT")
        print("=" * 72)
        for level, name, detail in self.checks:
            mark = {OK: "  ok  ", WARN: " WARN ", FATAL: "FATAL "}[level]
            print(f"[{mark}] {name:<{width}}  {detail}")
        if self.fatal:
            print(f"\n{len(self.fatal)} FATAL check(s). Refusing to build a lineup.")
            print("A lineup built on absent inputs looks exactly like a good one.")
        return not self.fatal


# ------------------------------------------------------------------ checks
def check_vegas(season: int, week: int, r: Result) -> None:
    """Vegas lines are published ~1-2 weeks out; further ahead they are null."""
    s = pd.read_parquet(RAW / "schedules.parquet")
    g = s[(s.season == season) & (s.week == week)]
    if g.empty:
        r.add(FATAL, "vegas lines", f"no games found for {season} week {week}")
        return
    null = g["total_line"].isna().mean()
    if null > 0.5:
        r.add(FATAL, "vegas lines",
              f"total_line null for {null:.0%} of {len(g)} games -- the whole "
              f"Vegas family (implied totals, spread, both interactions) will be NaN. "
              f"Re-pull schedules closer to kickoff.")
    elif null > 0:
        r.add(WARN, "vegas lines", f"total_line null for {null:.0%} of {len(g)} games")
    else:
        r.add(OK, "vegas lines", f"{len(g)} games, all priced")


def check_injuries(season: int, week: int, r: Result) -> None:
    """No report is NOT the same as nobody hurt -- but fillna(0) makes it look that way."""
    p = RAW / "injuries.parquet"
    if not p.exists():
        r.add(FATAL, "injury report", "injuries.parquet missing")
        return
    inj = pd.read_parquet(p)
    wk = inj[(inj.season == season) & (inj.week == week)]
    if wk.empty:
        have = inj[inj.season == season]["week"].max() if (inj.season == season).any() else None
        r.add(FATAL, "injury report",
              f"nothing for {season} week {week}"
              + (f" (latest on disk: week {int(have)})" if pd.notna(have) else "")
              + ". vacated_* and n_out would read as 0 = 'everyone healthy'. "
                f"FIX: python src/fetch_nflverse.py --update {season} --only injuries "
                f"schedules  (seconds, merges, leaves history alone). Inactives drop "
                f"90 min before kickoff, so run it again Sunday morning before you lock.")
    else:
        out = wk["report_status"].isin(["Out", "Doubtful"]).sum()
        lvl = WARN if out == 0 else OK
        r.add(lvl, "injury report",
              f"{len(wk)} rows, {out} Out/Doubtful"
              + (" -- zero ruled out is suspicious this close to kickoff" if out == 0 else ""))


def check_participation(season: int, r: Result) -> None:
    """Source of all six route features. nflverse publishes it on a lag."""
    files = sorted(glob.glob(str(RAW / "participation" / "pbp_participation_*.parquet")))
    years = [int(re.search(r"(\d{4})", Path(f).name).group(1)) for f in files]
    if not years:
        r.add(FATAL, "participation", "no participation files at all")
    elif season not in years:
        r.add(WARN, "participation",
              f"no {season} file (have {min(years)}-{max(years)}). All six route "
              f"features -- route_share_*, tprr_r5, route_share_trend -- will be NaN. "
              f"That family is ~9.7% of model attribution and the main WR/TE signal. "
              f"TRY: python src/fetch_nflverse.py --extras-only --end {season}. If that "
              f"reports 'not published upstream' it is NOT fixable here -- which is why "
              f"this is a WARN. The snap family (snap_pct_r3/r5, targets_per_snap_r5) is "
              f"weekly, present for {season}, and covers the same players more coarsely.")
    else:
        r.add(OK, "participation", f"{season} present ({min(years)}-{max(years)})")


def check_features(table: pd.DataFrame, features: list[str], r: Result) -> None:
    """The catch-all: any feature entirely absent on the rows being scored."""
    if table.empty:
        r.add(FATAL, "feature coverage", "no rows to score")
        return
    nan = table[features].isna().mean().sort_values(ascending=False)

    dead = nan[nan >= ALL_NAN_FATAL]
    if len(dead):
        shown = ", ".join(dead.index[:6]) + (" ..." if len(dead) > 6 else "")
        r.add(FATAL, "feature coverage",
              f"{len(dead)} of {len(features)} features are 100% NaN on all "
              f"{len(table)} rows: {shown}")
    high = nan[(nan >= HIGH_NAN_WARN) & (nan < ALL_NAN_FATAL)]
    if len(high):
        r.add(WARN, "feature coverage",
              f"{len(high)} feature(s) >{HIGH_NAN_WARN:.0%} NaN: "
              + ", ".join(f"{c} {v:.0%}" for c, v in high.head(6).items()))
    if not len(dead) and not len(high):
        r.add(OK, "feature coverage",
              f"{len(features)} features, worst is {nan.iloc[0]:.0%} NaN ({nan.index[0]})")


def check_pool(pool: pd.DataFrame, r: Result) -> None:
    """A lineup needs one of each slot; unmatched players are silently unpickable."""
    if pool.empty:
        r.add(FATAL, "player pool", "pool is empty")
        return
    have = pool.dropna(subset=["projection"]) if "projection" in pool.columns else pool
    counts = have["position"].value_counts()
    need = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
    short = {p: (int(counts.get(p, 0)), n) for p, n in need.items() if counts.get(p, 0) < n}
    if short:
        r.add(FATAL, "player pool",
              "not enough projected players: "
              + ", ".join(f"{p} {h}/{n}" for p, (h, n) in short.items()))
    else:
        r.add(OK, "player pool",
              f"{len(have)} projected -- " + ", ".join(f"{p} {int(counts.get(p,0))}" for p in need))


def run(season: int, week: int, table: pd.DataFrame | None = None,
        features: list[str] | None = None, pool: pd.DataFrame | None = None,
        strict: bool = True) -> Result:
    """Run every applicable check. Raises SystemExit on FATAL when strict."""
    r = Result()
    check_vegas(season, week, r)
    check_injuries(season, week, r)
    check_participation(season, r)
    if table is not None and features:
        check_features(table, features, r)
    if pool is not None:
        check_pool(pool, r)

    ok = r.report()
    if strict and not ok:
        raise SystemExit(
            "\npre-flight failed. Fix the inputs, or re-run with --skip-preflight "
            "if you understand exactly what is missing and want a lineup anyway."
        )
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    a = p.parse_args()
    run(a.season, a.week, strict=False)
