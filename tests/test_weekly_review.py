"""Tests for the weekly review: python -m unittest discover -s tests

Offline: a MemoryStore holds the week's reports, a fake judge scores them and
a small DataFrame stands in for nflverse snap counts. What it pins down:

  - the week's window is Tuesday..Monday night, and the Sunday is the main one
  - reused reports are counted but judged once; pre-10/16 reports are skipped
  - a 429 from the judge is retried, a real failure is not
  - warnings are scored against who actually took a snap, and "did not play"
    is only concluded once every team's data is in
  - each regression rule fires on its own, and a normal week fires none
  - the copied QUERY / CORE_TOOLS still match the originals
"""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gameday_agent"))
sys.path.insert(0, str(ROOT))
import weekly_review as wr  # noqa: E402
from user_store import MemoryStore, User  # noqa: E402

# Week 6 Thu 10/15 .. Mon 10/19; week 5 ended Mon 10/12. Week 7 starts 10/22.
SCHEDULE = pd.DataFrame({
    "season": 2026,
    "week": [5, 5, 6, 6, 6, 6, 7],
    "gameday": [date(2026, 10, 11), date(2026, 10, 12), date(2026, 10, 15),
                date(2026, 10, 18), date(2026, 10, 18), date(2026, 10, 19),
                date(2026, 10, 22)],
})


@dataclass
class FakeVerdict:
    scores: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def percent(self) -> float:
        return 25.0 * (sum(self.scores.values()) / len(self.scores) - 1) if self.scores else 0.0


def lineup_call(players: list[dict]) -> dict:
    return {"name": "get_lineup_status", "args_json": "{}",
            "result_json": json.dumps({"status": "ok", "lineup": players,
                                       "not_found_on_slate": []})}


def agent_doc(players=None, problems=None, cost=0.01, tools=None, **extra) -> dict:
    players = players or [{"name": "Josh Allen", "position": "QB", "team": "BUF",
                           "injury_status": None}]
    return {"source": "agent", "text": "report", "agent_text": "report",
            "problems": problems or [], "tools": tools if tools is not None else list(wr.CORE_TOOLS),
            "total_cost_usd": cost, "tool_calls": [lineup_call(players)], **extra}


def store_with(reports: dict[tuple[str, str], dict]) -> MemoryStore:
    st = MemoryStore([User(uid=u) for u in {u for u, _ in reports}])
    for (uid, rid), doc in reports.items():
        st.save_report(uid, rid, doc)
    return st


class WeekTest(unittest.TestCase):
    def test_window_runs_from_after_last_week_to_monday_night(self):
        first, last, sunday = wr.week_days(2026, 6, SCHEDULE)
        self.assertEqual((first, last, sunday),
                         (date(2026, 10, 13), date(2026, 10, 19), date(2026, 10, 18)))

    def test_last_completed_week_on_a_wednesday(self):
        self.assertEqual(wr.last_completed_week(2026, date(2026, 10, 21), SCHEDULE), 6)
        # Monday night not over yet (the job never runs then, but be exact)
        self.assertEqual(wr.last_completed_week(2026, date(2026, 10, 19), SCHEDULE), 5)

    def test_monday_morning_grades_the_week_just_played(self):
        self.assertEqual(wr.last_sunday_week(2026, date(2026, 10, 19), SCHEDULE), 6)
        self.assertEqual(wr.last_sunday_week(2026, date(2026, 10, 21), SCHEDULE), 6)
        # Sunday itself: that week is not over, so the week before
        self.assertEqual(wr.last_sunday_week(2026, date(2026, 10, 18), SCHEDULE), 5)

    def test_collect_keeps_only_the_window(self):
        st = store_with({("u1", "2026-10-12"): agent_doc(), ("u1", "2026-10-13"): agent_doc(),
                         ("u2", "2026-10-19"): agent_doc(), ("u2", "2026-10-20"): agent_doc()})
        rows = wr.collect(st, date(2026, 10, 13), date(2026, 10, 19))
        self.assertEqual([(r["uid"], r["date"]) for r in rows],
                         [("u1", "2026-10-13"), ("u2", "2026-10-19")])


class GradeTest(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def judge(self, query, calls, results, text):
        self.calls.append((query, calls, results, text))
        return FakeVerdict(scores={"grounded": 5, "completeness": 4})

    def rows(self, docs):
        return [{"uid": f"u{i}", "date": "2026-10-18", "doc": d} for i, d in enumerate(docs)]

    def test_judges_agent_text_with_tool_results_as_evidence(self):
        g = wr.grade(self.rows([agent_doc(agent_text="what the model wrote",
                                          text="⚠️ correction\n\nwhat the model wrote")]),
                     self.judge)
        query, calls, results, text = self.calls[0]
        self.assertEqual(query, wr.QUERY)
        self.assertEqual(text, "what the model wrote")
        self.assertEqual(calls[0], ("get_lineup_status", {}))
        self.assertEqual(results[0][1]["status"], "ok")
        self.assertEqual(g[0]["quality"], 87.5)

    def test_reused_fallback_and_old_reports_are_not_judged(self):
        old = agent_doc(); old.pop("tool_calls")
        docs = [agent_doc(reused_from="u0"), {"source": "fallback", "text": "x"}, old]
        g = wr.grade(self.rows(docs), self.judge)
        self.assertEqual(self.calls, [])
        self.assertIsNone(g[0]["cost_usd"])      # the original carries the cost
        self.assertIn("predates", g[2]["judge_error"])

    def test_429_is_retried_and_a_real_error_is_not(self):
        answers = [FakeVerdict(error="ClientError: 429 RESOURCE_EXHAUSTED"),
                   FakeVerdict(scores={"grounded": 3})]
        judge = lambda *a: answers.pop(0)  # noqa: E731
        g = wr.grade(self.rows([agent_doc()]), judge, sleep=lambda s: None)
        self.assertEqual(g[0]["quality"], 50.0)

        tries = []
        def broken(*a):
            tries.append(1)
            return FakeVerdict(error="ValueError: bad json")
        g = wr.grade(self.rows([agent_doc()]), broken, sleep=lambda s: None)
        self.assertEqual(len(tries), 1)
        self.assertIn("bad json", g[0]["judge_error"])

    def test_judge_cap(self):
        g = wr.grade(self.rows([agent_doc(), agent_doc()]), self.judge, max_judged=1)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("cap", g[1]["judge_error"])

    def test_missing_core_tools_are_recorded(self):
        g = wr.grade(self.rows([agent_doc(tools=["get_lineup_status"])]), None)
        self.assertIn("get_vegas_context", g[0]["missing_core"])


class OutcomeTest(unittest.TestCase):
    PLAYERS = [
        {"name": "Joe Mixon", "position": "RB", "team": "HOU", "injury_status": "OUT"},
        {"name": "Mark Andrews", "position": "TE", "team": "BAL", "injury_status": "Q"},
        {"name": "Marvin Harrison Jr.", "position": "WR", "team": "ARI", "injury_status": None},
        {"name": "Puka Nacua", "position": "WR", "team": "LAR", "injury_status": None},
        {"name": "Ravens", "position": "DST", "team": "BAL", "injury_status": None},
    ]
    SNAPS = pd.DataFrame({
        "week": [6, 6, 6, 5],
        "player": ["Mark Andrews", "Marvin Harrison", "Some Lineman", "Puka Nacua"],
        "team": ["BAL", "ARI", "HOU", "LA"],
        "offense_snaps": [40, 55, 60, 50],
    })

    def rows(self):
        return [{"uid": "u1", "date": "2026-10-17", "doc": agent_doc([self.PLAYERS[0]])},
                {"uid": "u1", "date": "2026-10-18", "doc": agent_doc(self.PLAYERS)},
                {"uid": "u1", "date": "2026-10-19", "doc": agent_doc([])}]  # Monday: ignored

    def test_scored_against_the_sunday_report(self):
        o = wr.outcomes(self.rows(), date(2026, 10, 18), self.SNAPS, 6, expected_teams=3)
        self.assertEqual(o["status"], "ok")
        self.assertEqual(o["players"], 4)                    # DST skipped
        self.assertEqual((o["out_warned"], o["out_correct"]), (1, 1))
        self.assertEqual((o["questionable"], o["questionable_sat"]), (1, 0))
        # Harrison matched through the "Jr." suffix; Nacua has no week-6 row.
        self.assertEqual(o["surprise_sits"], ["Puka Nacua (no snap row)"])
        self.assertEqual(o["wrong_outs"], [])

    def test_no_conclusion_until_every_team_is_in(self):
        o = wr.outcomes(self.rows(), date(2026, 10, 18), self.SNAPS, 6, expected_teams=28)
        self.assertEqual(o["status"], "partial")
        self.assertEqual(o["surprise_sits"], [])
        self.assertEqual(o["unknown"], 2)                    # Mixon and Nacua

    def test_pending_when_the_week_is_not_published(self):
        o = wr.outcomes(self.rows(), date(2026, 10, 18), self.SNAPS, 7, expected_teams=3)
        self.assertEqual(o["status"], "pending")


class RegressionTest(unittest.TestCase):
    BASE = {"reports": 7, "agent_reports": 7, "fallback_rate": 0.0, "check_fail_rate": 0.0,
            "missing_core": [], "missing_core_reports": 0, "cost_per_report": 0.01,
            "judged": 7, "judge_attempted": 7, "quality": 82.0, "criteria": {}}

    def test_normal_week_and_first_week_are_clean(self):
        self.assertEqual(wr.regressions(self.BASE, [self.BASE]), [])
        self.assertEqual(wr.regressions(self.BASE, []), [])

    def test_each_rule(self):
        cases = {
            "quality": {"quality": 70.0},
            "fallback": {"fallback_rate": 0.5},
            "core": {"missing_core_reports": 2, "missing_core": ["get_vegas_context"]},
            "report_check": {"check_fail_rate": 0.4},
            "cost": {"cost_per_report": 0.02},
            "judge calls failed": {"judged": 0, "quality": None},
            "no reports": {"reports": 0},
        }
        for word, change in cases.items():
            with self.subTest(word):
                flags = wr.regressions({**self.BASE, **change}, [self.BASE])
                self.assertEqual(len(flags), 1, flags)
                self.assertIn(word, flags[0])

    def test_below_the_floor_flags_even_with_no_history(self):
        low = {**self.BASE, "quality": wr.QUALITY_FLOOR - 0.1}
        for history in ([], [low]):
            flags = wr.regressions(low, history)
            self.assertEqual(len(flags), 1, flags)
            self.assertIn("below the floor", flags[0])
        self.assertEqual(wr.regressions({**self.BASE, "quality": wr.QUALITY_FLOOR}, []), [])

    def test_small_quality_dip_is_noise(self):
        self.assertEqual(wr.regressions({**self.BASE, "quality": 75.0}, [self.BASE]), [])


class ReviewTest(unittest.TestCase):
    def test_end_to_end_with_history(self):
        st = store_with({("u1", "2026-10-18"): agent_doc(), ("u2", "2026-10-18"):
                         agent_doc(reused_from="u1")})
        st.save_weekly("2026-w05", {"summary": {**RegressionTest.BASE, "quality": 95.0}})
        judge = lambda *a: FakeVerdict(scores={"grounded": 3})  # noqa: E731 -- 50
        snaps = lambda season: pd.DataFrame(  # noqa: E731
            {"week": [6], "player": ["Josh Allen"], "team": ["BUF"], "offense_snaps": [70]})
        r = wr.review(st, 2026, 6, judge, snaps, SCHEDULE)
        self.assertEqual(r["week_id"], "2026-w06")
        self.assertEqual(r["summary"]["reports"], 2)
        self.assertEqual(r["summary"]["judged"], 1)
        # 50 is down 45 from the prior week, and under the floor too
        self.assertEqual(len(r["flags"]), 2, r["flags"])
        self.assertIn("below the floor", r["flags"][0])
        self.assertIn("down from 95.0", r["flags"][1])
        msg = wr.render(2026, 6, r["summary"], r["history"], r["outcome"], r["graded"], r["flags"])
        self.assertTrue(msg.startswith("❌"))
        self.assertIn("W05 95.0 → W06 50.0", msg)


class CopiesMatchTest(unittest.TestCase):
    def test_query_matches_run_report(self):
        import run_report
        self.assertEqual(wr.QUERY, run_report.QUERY)

    def test_core_tools_match_prefetch(self):
        try:
            from gameday_agent import prefetch
        except ImportError:
            self.skipTest("gameday_agent needs ADK")
        self.assertEqual(wr.CORE_TOOLS, prefetch.CORE_TOOLS)


if __name__ == "__main__":
    unittest.main()
