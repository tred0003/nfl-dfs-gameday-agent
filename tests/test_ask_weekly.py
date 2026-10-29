"""Tests for Ask's weekly exam (webapp/ask_weekly.py): python -m unittest discover -s tests

No model, no BigQuery, no network: the question maker gets a fake pool, the
grader a fake box score.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

import ask_weekly as aw  # noqa: E402

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def _pool(players):
    return {"run_id": "20261029T131016Z-test", "season": 2026, "week": 8, "players": players}


def _p(name, pos, proj, status=None):
    return {"name": name, "position": pos, "projection": proj, "status": status}


# A full slate's worth: plenty of close pairs at every position, a few Q players.
SLATE = ([_p(f"WR {i}", "WR", 8 + i * 0.5) for i in range(40)]
         + [_p(f"RB {i}", "RB", 8 + i * 0.5) for i in range(30)]
         + [_p(f"TE {i}", "TE", 6 + i * 0.4) for i in range(20)]
         + [_p(f"QB {i}", "QB", 12 + i * 0.5) for i in range(16)]
         + [_p(f"Q WR {i}", "WR", 9 + i, "Q") for i in range(6)]
         + [_p("Out Guy", "WR", 15.0, "O"), _p("Backup", "WR", 2.0)])


class QuestionsTest(unittest.TestCase):
    def setUp(self):
        self.qs = aw.make_questions(_pool(SLATE))

    def test_fifty_questions_each_player_once(self):
        self.assertEqual(len(self.qs), aw.N_CLOSE + aw.N_QUESTIONABLE)
        names = [n for q in self.qs for n in (q["player_a"], q["player_b"])]
        self.assertEqual(len(names), len(set(names)))

    def test_pairs_are_close_and_at_one_position(self):
        for q in self.qs:
            self.assertLessEqual(abs(q["proj_a"] - q["proj_b"]), aw.CLOSE_POINTS)
            self.assertIn(q["player_a"], q["question"])
            self.assertIn(q["player_b"], q["question"])

    def test_ruled_out_and_low_projections_never_asked(self):
        names = {n for q in self.qs for n in (q["player_a"], q["player_b"])}
        self.assertNotIn("Out Guy", names)
        self.assertNotIn("Backup", names)

    def test_every_questionable_player_gets_a_healthy_partner(self):
        qq = [q for q in self.qs if q["kind"] == "questionable"]
        self.assertEqual(len(qq), 6)
        for q in qq:
            self.assertEqual(sorted([q["status_a"] or "", q["status_b"] or ""]), ["", "Q"])

    def test_the_model_pick_is_the_higher_projection(self):
        for q in self.qs:
            hi = q["player_a"] if q["proj_a"] >= q["proj_b"] else q["player_b"]
            self.assertEqual(q["model_pick"], hi)

    def test_same_run_same_questions(self):
        self.assertEqual(self.qs, aw.make_questions(_pool(SLATE)))

    def test_unused_questionable_slots_go_to_close_pairs(self):
        self.assertEqual(sum(q["kind"] == "close" for q in self.qs), 44)


@unittest.skipIf(pd is None, "needs pandas")
class GradeTest(unittest.TestCase):
    POINTS = None

    def setUp(self):
        self.points = pd.DataFrame([
            {"key": "puka nacua", "week": 8, "dk_points": 21.0},
            {"key": "ceedee lamb", "week": 8, "dk_points": 14.5},
            {"key": "kyle pitts", "week": 8, "dk_points": 6.0},
            {"key": "lamar jackson", "week": 7, "dk_points": 25.0},   # sat week 8
            {"key": "jared goff", "week": 8, "dk_points": 6.0},
        ])

    def _a(self, a, b, pick, model_pick, kind="close"):
        return {"qid": "q", "season": 2026, "week": 8, "kind": kind, "position": "WR",
                "player_a": a, "player_b": b, "pick": pick, "model_pick": model_pick}

    def test_the_higher_scorer_is_the_right_answer(self):
        g = aw.grade_rows([self._a("Puka Nacua", "CeeDee Lamb", "Puka Nacua", "CeeDee Lamb")],
                          self.points)[0]
        self.assertEqual(g["winner"], "Puka Nacua")
        self.assertTrue(g["ask_correct"])
        self.assertFalse(g["model_correct"])

    def test_a_player_who_sat_scored_zero(self):
        g = aw.grade_rows([self._a("Lamar Jackson", "Jared Goff", "Lamar Jackson",
                                   "Lamar Jackson", "questionable")], self.points)[0]
        self.assertEqual(g["pts_a"], 0.0)
        self.assertEqual(g["winner"], "Jared Goff")
        self.assertFalse(g["ask_correct"])

    def test_suffixes_match_and_an_unknown_name_is_ungradable(self):
        rows = aw.grade_rows([
            self._a("Kyle Pitts Sr.", "CeeDee Lamb", "CeeDee Lamb", "Kyle Pitts Sr."),
            self._a("Made Up Guy", "CeeDee Lamb", "CeeDee Lamb", "CeeDee Lamb")], self.points)
        self.assertTrue(rows[0]["ask_correct"])
        self.assertIsNone(rows[1]["ask_correct"])
        self.assertEqual(rows[1]["note"], "unmatched name")

    def test_no_pick_is_a_miss_and_a_tie_is_not_graded(self):
        rows = aw.grade_rows([
            self._a("Puka Nacua", "CeeDee Lamb", aw.NO_PICK, "Puka Nacua"),
            self._a("Kyle Pitts Sr.", "Jared Goff", "Jared Goff", "Jared Goff")], self.points)
        self.assertFalse(rows[0]["ask_correct"])
        self.assertIsNone(rows[1]["ask_correct"])
        s = aw.summarise(rows)
        self.assertEqual((s["graded"], s["ask"], s["model"], s["no_pick"]), (1, 0.0, 1.0, 1))

    def test_summary_tracks_when_ask_overrode_the_model(self):
        rows = aw.grade_rows([
            self._a("Puka Nacua", "CeeDee Lamb", "Puka Nacua", "CeeDee Lamb"),
            self._a("Puka Nacua", "CeeDee Lamb", "Puka Nacua", "Puka Nacua")], self.points)
        s = aw.summarise(rows)
        self.assertEqual((s["disagree"], s["ask_when_disagree"]), (1, 1.0))


if __name__ == "__main__":
    unittest.main()
