"""This week's player pool and suggested lineups, for the web app.

Both come from the same place the agent's value picks do: the latest run of
the projection job in BigQuery (fantasy-football-ml-503718.dfs.projections,
see deploy/PROJECTIONS-CLOUD.md). The web app never runs the model.

SHARED, NOT PER USER. The pool and the suggested lineups are the same for
every signed-in user, so they are computed once per projection run and
cached in the process:

    pool         re-read at most every POOL_TTL seconds (one small query)
    suggestions  solved once per run_id -- a new run is the only thing that
                 can change them

A hundred users loading the page cost one BigQuery query and one solve, not
a hundred. Only the lineup itself is per user, and that lives in Firestore.
"""

from __future__ import annotations

import os
import threading
import time

PROJECTIONS_TABLE = os.environ.get(
    "PROJECTIONS_TABLE", "fantasy-football-ml-503718.dfs.projections")
POOL_TTL = 600        # projections change twice a week; 10 minutes is plenty
N_SUGGESTED = 3

# Columns the page needs. Everything else in the table stays server-side.
COLUMNS = ["name", "position", "team", "opp_team", "game", "salary", "status",
           "projection", "ceiling", "value"]


class SlateError(Exception):
    """The pool could not be read. The page says so instead of showing nothing."""


def _clean(v):
    """BigQuery NaN / NA -> None, numpy scalars -> Python, for JSON."""
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v.item() if hasattr(v, "item") else v


class BigQuerySlate:
    """The production source. Tests use a fake with the same two methods."""

    def __init__(self, table: str = PROJECTIONS_TABLE, ttl: int = POOL_TTL):
        self.table = table
        self.ttl = ttl
        self._lock = threading.Lock()
        self._pool: dict | None = None
        self._pool_at = 0.0
        self._suggested: dict[str, list] = {}

    def _read(self) -> dict:
        from google.cloud import bigquery

        client = bigquery.Client(project=self.table.split(".")[0])
        # Latest run of the latest slate. Not filtered to a draft group: the
        # job only projects the main slate, and the DK endpoint that would
        # name it is not reachable from Cloud Run (see dk_client.py).
        q = f"""
            SELECT run_id, run_ts, week, draft_group_id, {", ".join(COLUMNS)}
            FROM `{self.table}`
            WHERE run_id = (SELECT run_id FROM `{self.table}`
                            ORDER BY run_ts DESC LIMIT 1)
        """
        df = client.query(q).to_dataframe()
        if df.empty:
            raise SlateError("no projection run found")
        first = df.iloc[0]
        players = [{c: _clean(r[c]) for c in COLUMNS} for _, r in df.iterrows()]
        for p in players:
            if p["projection"] is not None:
                p["projection"] = round(float(p["projection"]), 1)
            for k in ("ceiling", "value"):
                if p[k] is not None:
                    p[k] = round(float(p[k]), 1)
            p["salary"] = int(p["salary"] or 0)
            # The job writes a healthy player's status as the string "None".
            if str(p["status"]).strip().lower() in ("", "none", "nan"):
                p["status"] = None
        players.sort(key=lambda p: -(p["projection"] or 0))
        return {
            "run_id": str(first["run_id"]),
            "week": int(first["week"]),
            "draft_group_id": int(first["draft_group_id"]),
            "projected_at": first["run_ts"].isoformat(),
            "players": players,
        }

    def pool(self) -> dict:
        with self._lock:
            if self._pool is None or time.time() - self._pool_at > self.ttl:
                try:
                    self._pool = self._read()
                except SlateError:
                    raise
                except Exception as e:  # noqa: BLE001
                    if self._pool is None:
                        raise SlateError(f"{type(e).__name__}: {str(e)[:200]}") from e
                    # A failed refresh keeps serving the last good pool rather
                    # than blanking the page; the next request tries again.
                self._pool_at = time.time()
            return self._pool

    def suggested(self) -> dict:
        pool = self.pool()
        rid = pool["run_id"]
        with self._lock:
            if rid not in self._suggested:
                self._suggested = {rid: solve(pool["players"], N_SUGGESTED)}
            lineups = self._suggested[rid]
        return {"run_id": rid, "week": pool["week"],
                "projected_at": pool["projected_at"], "lineups": lineups}


def solve(players: list[dict], n: int) -> list[dict]:
    """Top-n DK lineups by projection, via the same ILP as src/optimize.py."""
    import pandas as pd
    from optimize import SALARY_CAP, optimize

    df = pd.DataFrame(players)
    # Ruled-out players carry no projection and drop out inside optimize().
    df = df[~df["status"].fillna("").str.upper().isin(["O", "OUT", "IR", "D", "DOUBTFUL"])]
    out = []
    for lu in optimize(df, n_lineups=n):
        rows = [{c: _clean(r[c]) for c in COLUMNS} for _, r in lu.iterrows()]
        out.append({
            "players": rows,
            "salary": int(sum(r["salary"] for r in rows)),
            "salary_cap": SALARY_CAP,
            "projection": round(sum(r["projection"] or 0 for r in rows), 1),
        })
    return out
