"""Tests for the projection sanity gate: python -m unittest discover -s tests

Offline: a synthetic 16-team slate stands in for BigQuery, and the real
optimizer (pulp) solves it, so the gate is tested against genuine lineups.
Each test breaks ONE thing in a healthy pool and asserts which check fires,
so a failing test names the check that regressed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import projection_checks as pc  # noqa: E402
import suggest_lineup as sl  # noqa: E402

SHAPE = [("QB", 1, 6500), ("RB", 3, 6500), ("WR", 4, 6500), ("TE", 2, 4500), ("DST", 1, 3000)]


def healthy_pool(seed: int = 0) -> pd.DataFrame:
    """16 teams in 8 games; projection tracks salary plus noise, like the real model."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(16):
        team, opp = f"T{t:02d}", f"T{t ^ 1:02d}"
        game = f"T{min(t, t ^ 1):02d}@T{max(t, t ^ 1):02d}"
        for pos, count, top in SHAPE:
            for k in range(count):
                salary = int(max(2500, top - k * 1100 + rng.integers(-300, 300)) // 100 * 100)
                proj = salary / 1000 * (2.2 if pos != "DST" else 1.6) + rng.normal(0, 1.5)
                rows.append(dict(name=f"{team}-{pos}{k}", position=pos, team=team,
                                 opp_team=opp, game=game, salary=salary,
                                 status="None", projection=max(proj, 0.5),
                                 depth_rank=0.0 if k == 0 else 1.0,
                                 week=6, model_version="5",
                                 run_ts=pd.Timestamp("2026-10-12 12:00", tz="UTC")))
    return pd.DataFrame(rows)


def checks_of(findings, level="block") -> set[str]:
    return {f.check for f in findings if f.level == level}


class PoolChecks(unittest.TestCase):
    def test_healthy_pool_is_clean(self):
        self.assertEqual(pc.check_pool(healthy_pool()), [])

    def test_too_few_projections_blocks(self):
        p = healthy_pool()
        p.loc[p.index[:-100], "projection"] = np.nan
        self.assertEqual(checks_of(pc.check_pool(p)), {"coverage"})

    def test_out_of_range_projection_blocks(self):
        p = healthy_pool()
        p.loc[3, "projection"] = 91.0
        p.loc[4, "projection"] = -2.0
        self.assertEqual(checks_of(pc.check_pool(p)), {"range"})

    def test_scrambled_projections_block(self):
        p = healthy_pool()
        p["projection"] = np.random.default_rng(1).permutation(p["projection"].to_numpy())
        self.assertEqual(checks_of(pc.check_pool(p)), {"salary-corr"})

    def test_constant_projections_block(self):
        p = healthy_pool()
        p["projection"] = 8.0
        self.assertEqual(checks_of(pc.check_pool(p)), {"salary-corr"})

    def test_mixed_model_versions_block(self):
        p = healthy_pool()
        p.loc[0, "model_version"] = "4"
        self.assertEqual(checks_of(pc.check_pool(p)), {"mixed-run"})

    def test_unprojected_healthy_star_warns_but_does_not_block(self):
        """The Hollywood Brown case: a name that never joined, so no projection."""
        p = healthy_pool()
        i = p.index[(p.position == "WR") & (p.salary >= 6000)][0]
        p.loc[i, "projection"] = np.nan
        f = pc.check_pool(p)
        self.assertEqual(checks_of(f, "warn"), {"unprojected"})
        self.assertEqual(checks_of(f), set())
        self.assertIn(p.loc[i, "name"], f[0].message)

    def test_unprojected_but_out_star_is_silent(self):
        """Real 10/12 run: Lamar, Barkley, Smith had no projection AND were OUT."""
        p = healthy_pool()
        i = p.index[(p.position == "WR") & (p.salary >= 6000)][0]
        p.loc[i, ["projection", "status"]] = [np.nan, "OUT"]
        self.assertEqual(pc.check_pool(p), [])


class LineupChecks(unittest.TestCase):
    def solved(self, n: int = 3) -> list[pd.DataFrame]:
        return sl.optimize(sl.playable(healthy_pool()), "projection", n_lineups=n)

    def test_real_solver_output_passes(self):
        lineups = self.solved()
        self.assertEqual(len(lineups), 3)
        self.assertEqual(pc.check_lineups(lineups), [])

    def test_over_cap_blocks(self):
        lu = self.solved(1)[0].copy()
        lu.loc[lu.index[0], "salary"] += 9000
        self.assertIn("cap", checks_of(pc.check_lineups([lu])))

    def test_duplicate_player_blocks(self):
        lu = self.solved(1)[0].copy()
        lu.iloc[1] = lu.iloc[0]
        self.assertIn("duplicate", checks_of(pc.check_lineups([lu])))

    def test_wrong_size_blocks(self):
        lu = self.solved(1)[0].iloc[:8]
        self.assertEqual(checks_of(pc.check_lineups([lu])), {"shape"})

    def test_illegal_position_mix_blocks(self):
        lu = self.solved(1)[0].copy()
        lu["position"] = "WR"
        self.assertIn("positions", checks_of(pc.check_lineups([lu])))

    def test_ruled_out_player_blocks(self):
        lu = self.solved(1)[0].copy()
        lu.loc[lu.index[2], "status"] = "OUT"
        self.assertIn("out-player", checks_of(pc.check_lineups([lu])))

    def test_implausible_total_blocks(self):
        lu = self.solved(1)[0].copy()
        lu["projection"] = 30.0
        self.assertIn("lineup-total", checks_of(pc.check_lineups([lu])))


class PostGate(unittest.TestCase):
    """post() is where the checks decide whether anything reaches Discord."""

    def test_healthy_pool_posts(self):
        with mock.patch.object(sl, "send_discord", return_value=True) as send:
            self.assertTrue(sl.post(healthy_pool()))
        self.assertEqual(send.call_count, 1)
        self.assertIn("Suggested DK lineups", send.call_args[0][0])

    def test_bad_pool_withholds_notifies_and_raises(self):
        p = healthy_pool()
        p["projection"] = np.random.default_rng(1).permutation(p["projection"].to_numpy())
        with mock.patch.object(sl, "send_discord", return_value=True) as send:
            with self.assertRaises(sl.ProjectionCheckFailed):
                sl.post(p)
        self.assertEqual(send.call_count, 1)
        self.assertIn("withheld", send.call_args[0][0])
        self.assertNotIn("Lineup 1", send.call_args[0][0])

    def test_warning_rides_along_with_the_lineups(self):
        p = healthy_pool()
        i = p.index[(p.position == "WR") & (p.salary >= 6000)][0]
        p.loc[i, "projection"] = np.nan
        with mock.patch.object(sl, "send_discord", return_value=True) as send:
            self.assertTrue(sl.post(p))
        msg = send.call_args[0][0]
        self.assertIn("Lineup 1", msg)
        self.assertIn("⚠️", msg)
        self.assertIn(p.loc[i, "name"], msg)

    def test_dry_run_withhold_does_not_hit_discord(self):
        p = healthy_pool()
        p["projection"] = 8.0
        with mock.patch.object(sl, "send_discord") as send:
            with self.assertRaises(sl.ProjectionCheckFailed):
                sl.post(p, dry_run=True)
        send.assert_not_called()


class WriteGate(unittest.TestCase):
    """gate_write() keeps a bad run out of BigQuery, which the agent's value
    picks read -- a bad table is worse than a missing one."""

    ps = sl     # gate_write lives in suggest_lineup so tests skip the ML imports

    def scrambled(self) -> pd.DataFrame:
        p = healthy_pool()
        p["projection"] = np.random.default_rng(1).permutation(p["projection"].to_numpy())
        return p

    def test_healthy_run_may_be_written(self):
        with mock.patch.object(sl, "send_discord") as send:
            self.assertTrue(self.ps.gate_write(healthy_pool(), notify=True))
        send.assert_not_called()

    def test_bad_run_is_refused_and_discord_told(self):
        with mock.patch.object(sl, "send_discord", return_value=True) as send:
            self.assertFalse(self.ps.gate_write(self.scrambled(), notify=True))
        self.assertEqual(send.call_count, 1)
        self.assertIn("withheld", send.call_args[0][0])

    def test_bad_friday_run_is_refused_silently(self):
        """Friday has no Discord post; the exit-1 alert is its only signal."""
        with mock.patch.object(sl, "send_discord") as send:
            self.assertFalse(self.ps.gate_write(self.scrambled(), notify=False))
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
