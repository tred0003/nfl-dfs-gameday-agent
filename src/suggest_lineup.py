"""Post suggested DK lineups to Discord, built from the cloud projections.

    python src/suggest_lineup.py              # 3 lineups -> Discord
    python src/suggest_lineup.py --dry-run    # print instead of posting
    python src/suggest_lineup.py --n 1

Reads the latest project-slate run from BigQuery (dfs.projections) rather than
re-projecting, so the lineups use exactly the numbers the agent's value picks
use. The model projects; the optimizer (optimize.py, an integer program) picks
the nine. No LLM anywhere in this path.

Filtered out before the solver sees them:
  - OUT / IR / Doubtful players
  - backup QBs. depth_rank is a percentile where 1.0 is LAST, so the starter
    is the team's lowest value. Without this the solver happily rosters a $4,000
    backup whose projection assumes he plays -- the 10/10 value list led with one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from optimize import optimize  # noqa: E402
from watch_slate import load_env, send_discord  # noqa: E402

PROJECT = os.environ.get("ML_PROJECT", "fantasy-football-ml-503718")
TABLE = os.environ.get("PROJECTIONS_TABLE", f"{PROJECT}.dfs.projections")
SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]


def latest_run() -> pd.DataFrame:
    """The newest run for THIS week's main slate.

    Filtered on draft group, not just newest: on a Thursday before the job has
    run, the newest row in the table is last week's slate, and a lineup of
    players who already played would post without complaint.
    """
    from google.cloud import bigquery
    from dk_client import list_slates, pick_main_slate
    dg = pick_main_slate(list_slates())
    sql = f"""
        SELECT * FROM `{TABLE}`
        WHERE run_id = (SELECT run_id FROM `{TABLE}` WHERE draft_group_id = @dg
                        ORDER BY run_ts DESC LIMIT 1)
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("dg", "INT64", dg)])
    return bigquery.Client(project=PROJECT).query(sql, job_config=cfg).to_dataframe()


def post(pool: pd.DataFrame, n: int = 3, dry_run: bool = False) -> bool:
    """Solve and post. Called by project_slate.py right after it writes a run,
    so the scheduled post never races the projections it depends on."""
    lineups = optimize(playable(pool).dropna(subset=["projection"]), "projection",
                       n_lineups=n)
    if not lineups:
        raise RuntimeError("optimizer found no valid lineup")
    msg = fmt(lineups, pool)
    if dry_run:
        print(msg)
        return True
    return send_discord(msg)


def playable(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool[~pool["status"].fillna("").isin(["O", "OUT", "IR", "D"])].copy()
    qb = pool["position"].eq("QB")
    starter = pool[qb].groupby("team")["depth_rank"].transform("min")
    backup = qb & pool["depth_rank"].notna() & pool["depth_rank"].ne(starter)
    return pool[~backup]


def slotted(lineup: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Give each player a DK roster slot; the extra RB/WR/TE is the FLEX."""
    need = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
    out, flex = [], None
    for _, r in lineup.iterrows():
        if need.get(r["position"], 0) > 0:
            need[r["position"]] -= 1
            out.append((r["position"], r))
        else:
            flex = r
    if flex is not None:
        out.insert(7, ("FLEX", flex))
    return out


def fmt(lineups: list[pd.DataFrame], pool: pd.DataFrame) -> str:
    r0 = pool.iloc[0]
    at = pd.Timestamp(r0["run_ts"]).tz_convert("America/New_York")
    when = f"{at:%a} {at.hour % 12 or 12}:{at:%M} {at:%p}"   # %-I is not portable
    lines = [f"**🧮 Suggested DK lineups — Week {int(r0['week'])}**",
             f"_Model v{r0['model_version']}, projected {when} ET. "
             "DST is rule-based. Check injuries before lock._"]
    for n, lu in enumerate(lineups, 1):
        sal, pts = int(lu["salary"].sum()), lu["projection"].sum()
        lines.append(f"\n**Lineup {n}** — {pts:.1f} pts, ${sal:,}")
        for slot, r in slotted(lu):
            q = " ⚠️Q" if str(r["status"]) == "Q" else ""
            lines.append(f"`{slot:<4}` {r['name']} ({r['team']}) ${int(r['salary']):,} — "
                         f"{r['projection']:.1f}{q}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Suggested lineups -> Discord")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    load_env()
    pool = latest_run()
    if pool.empty:
        raise SystemExit(f"no projections for this week's slate in {TABLE} yet")
    return 0 if post(pool, a.n, a.dry_run) else 1


if __name__ == "__main__":
    sys.exit(main())
