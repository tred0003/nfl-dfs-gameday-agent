"""Record lineups and score them after games -- the season-long scoreboard.

No DraftKings account access. DK's own scoring is reproducible from public box
scores (see scoring.py), so actual points are COMPUTED rather than scraped. The
only thing that genuinely needs DK is your contest placement, which is a manual
one-click CSV export and lives in `placement` if you choose to fill it in.

The point of this file is the head-to-head: your lineup vs the model's vs a
naive chalk baseline, every week, accumulating. That comparison is the entire
evaluation story of the project -- without it there is no evidence either way.

  python src/track.py record --week 1 --source mine --from-csv my_lineup.csv
  python src/track.py record --week 1 --source model --from-projections <parquet>
  python src/track.py score  --week 1
  python src/track.py standings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from predict_slate import normalize_name  # noqa: E402
from scoring import add_dk_points  # noqa: E402

LINEUPS = ROOT / "data" / "lineups"
RESULTS = ROOT / "data" / "lineups" / "results.parquet"
SEASON = 2026


def _store(df: pd.DataFrame, season: int, week: int, source: str) -> Path:
    LINEUPS.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["season"], df["week"], df["source"] = season, week, source
    path = LINEUPS / f"{season}_wk{week:02d}_{source}.parquet"
    df.to_parquet(path, index=False)
    print(f"recorded {len(df)} players -> {path.name}")
    return path


def record(args) -> None:
    if args.from_projections:
        from optimize import optimize
        pool = pd.read_parquet(args.from_projections)
        lus = optimize(pool, n_lineups=1)
        if not lus:
            raise SystemExit("optimizer found no feasible lineup")
        df = lus[0][["name", "position", "team", "salary", "projection"]]
    elif args.from_csv:
        # a plain CSV with a `name` column is enough; salary/position optional
        df = pd.read_csv(args.from_csv)
        if "name" not in df.columns:
            raise SystemExit("CSV needs a 'name' column")
    else:
        raise SystemExit("pass --from-csv or --from-projections")

    if len(df) != 9:
        print(f"  WARNING: {len(df)} players, expected 9")
    _store(df, args.season, args.week, args.source)


def actual_points(season: int, week: int) -> pd.DataFrame:
    """Real DK points for every player in a week, computed from box scores."""
    import nfl_data_py as nfl

    try:
        w = nfl.import_weekly_data([season])
    except Exception as e:
        raise SystemExit(
            f"nflverse has no weekly data for {season} yet ({type(e).__name__}). "
            "Box scores usually land a day or two after games. Retry Monday."
        )

    w = w[w["week"].eq(week)]
    if w.empty:
        raise SystemExit(f"no {season} week {week} rows yet -- try again later")

    w = add_dk_points(w)
    w["join_name"] = normalize_name(w["player_display_name"])
    return w[["join_name", "player_display_name", "position",
              "recent_team", "dk_points"]]


def score(args) -> None:
    actual = actual_points(args.season, args.week)
    rows = []

    for path in sorted(LINEUPS.glob(f"{args.season}_wk{args.week:02d}_*.parquet")):
        lu = pd.read_parquet(path)
        lu["join_name"] = normalize_name(lu["name"])
        m = lu.merge(actual, on="join_name", how="left")

        missing = m["dk_points"].isna().sum()
        # DST has no row in weekly player stats -- expected, not an error
        total = m["dk_points"].fillna(0).sum()

        print(f"\n--- {path.stem} ---")
        cols = [c for c in ["name", "position_x", "team", "salary",
                            "projection", "dk_points"] if c in m.columns]
        print(m[cols].to_string(index=False))
        print(f"    TOTAL {total:.1f}" + (f"   ({missing} unmatched)" if missing else ""))

        rows.append({"season": args.season, "week": args.week,
                     "source": lu["source"].iloc[0], "points": round(float(total), 2),
                     "unmatched": int(missing)})

    if not rows:
        raise SystemExit(f"no lineups recorded for week {args.week}")

    new = pd.DataFrame(rows)
    if RESULTS.exists():
        old = pd.read_parquet(RESULTS)
        old = old[~(old["season"].eq(args.season) & old["week"].eq(args.week))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(RESULTS, index=False)
    print(f"\nresults -> {RESULTS.name}")


def standings(args) -> None:
    if not RESULTS.exists():
        raise SystemExit("no results yet -- run `score` after a slate finishes")
    r = pd.read_parquet(RESULTS)
    print("\nweek by week:")
    print(r.pivot_table(index=["season", "week"], columns="source",
                        values="points").round(1).to_string())
    print("\nseason totals:")
    print(r.groupby("source")["points"].agg(
        weeks="size", total="sum", mean="mean").round(2).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record")
    r.add_argument("--week", type=int, required=True)
    r.add_argument("--season", type=int, default=SEASON)
    r.add_argument("--source", required=True,
                   help="mine | model | chalk -- whose lineup this is")
    r.add_argument("--from-csv")
    r.add_argument("--from-projections")
    r.set_defaults(func=record)

    s = sub.add_parser("score")
    s.add_argument("--week", type=int, required=True)
    s.add_argument("--season", type=int, default=SEASON)
    s.set_defaults(func=score)

    t = sub.add_parser("standings")
    t.set_defaults(func=standings)

    a = p.parse_args()
    a.func(a)
