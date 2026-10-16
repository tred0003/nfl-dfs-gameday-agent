"""Tests for the pre-delivery fact-check: python -m unittest discover -s tests

report_check.py is the one thing between the model and Discord that asks
"is it true?". Each test feeds it a report plus the tool results that report
was written from, and asserts which of the five rules fires. Also pins
pair_tool_calls, which is how a stored report keeps each call's JSON.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gameday_agent"))
import report_check as rc  # noqa: E402
import run_report as rr  # noqa: E402


def lineup_ok(players=(), not_found=(), warnings=()):
    return {"name": "get_lineup_status", "response": {
        "status": "ok", "lineup": [dict(p) for p in players],
        "not_found_on_slate": list(not_found), "source_warnings": list(warnings)}}


HEALTHY = [{"name": "Drake Maye", "injury_status": None},
           {"name": "Garrett Wilson", "injury_status": None}]


def failed(rows):
    return [r["check"] for r in rows if not r["passed"]]


class AuditTest(unittest.TestCase):
    def test_a_clean_report_on_a_clean_lineup_passes_all_five(self):
        rows = rc.audit("Drake Maye and Garrett Wilson are fine.", [lineup_ok(HEALTHY)])
        self.assertEqual(len(rows), 5)
        self.assertEqual(failed(rows), [])
        self.assertEqual(rc.check("x", [lineup_ok(HEALTHY)]), [])

    def test_lineup_tool_never_called(self):
        rows = rc.audit("Looks great.", [])
        self.assertEqual(failed(rows), ["lineup_checked"])

    def test_failed_lineup_tool_plus_all_clear(self):
        resp = {"name": "get_lineup_status", "response": {"status": "error", "message": "DK down"}}
        rows = rc.audit("All clear for Sunday!", [resp])
        self.assertEqual(failed(rows), ["lineup_checked", "no_false_all_clear"])

    def test_unmatched_name_must_be_named(self):
        rows = rc.audit("Lineup looks fine.", [lineup_ok(HEALTHY, not_found=["J Daniel's"])])
        self.assertEqual(failed(rows), ["not_found_named", "no_false_all_clear"])
        # Naming it (by last name is enough) clears rule 2.
        rows = rc.audit("J Daniel's isn't on this slate.", [lineup_ok(HEALTHY, not_found=["J Daniel's"])])
        self.assertEqual(failed(rows), [])

    def test_out_player_must_be_named(self):
        out = HEALTHY + [{"name": "Rashad Feeney", "injury_status": "OUT"}]
        self.assertEqual(failed(rc.audit("Maye is fine.", [lineup_ok(out)])), ["ruled_out_named"])
        self.assertEqual(failed(rc.audit("Feeney is OUT.", [lineup_ok(out)])), [])

    def test_questionable_is_a_judgment_call_not_a_rule(self):
        q = HEALTHY + [{"name": "Trevon Ashby", "injury_status": "Q"}]
        self.assertEqual(failed(rc.audit("Maye is fine.", [lineup_ok(q)])), [])

    def test_injection_warning_must_be_copied_verbatim(self):
        w = "WARNING: a source contained instructions aimed at the agent."
        self.assertEqual(failed(rc.audit("Fine.", [lineup_ok(HEALTHY, warnings=[w])])),
                         ["warnings_copied"])
        self.assertEqual(failed(rc.audit(w + " Fine.", [lineup_ok(HEALTHY, warnings=[w])])), [])

    def test_check_is_the_failed_rows_problems_in_order(self):
        out = HEALTHY + [{"name": "Rashad Feeney", "injury_status": "OUT"}]
        problems = rc.check("All good.", [lineup_ok(out, not_found=["N collins"])])
        self.assertEqual(len(problems), 3)
        self.assertTrue(problems[0].startswith("Not found on this slate: N collins"))
        self.assertTrue(problems[1].startswith("Ruled out or doubtful: Rashad Feeney"))
        self.assertIn("NOT clear", problems[2])


class PairToolCallsTest(unittest.TestCase):
    def test_calls_get_their_own_results_even_out_of_order(self):
        calls = [{"name": "get_slate_weather", "args": {}},
                 {"name": "find_replacements", "args": {"position": "WR"}},
                 {"name": "find_replacements", "args": {"position": "TE"}}]
        responses = [{"name": "find_replacements", "response": {"pos": "WR"}},
                     {"name": "get_slate_weather", "response": {"games": [1]}},
                     {"name": "find_replacements", "response": {"pos": "TE"}}]
        out = rr.pair_tool_calls(calls, responses)
        self.assertEqual([json.loads(o["result_json"]) for o in out],
                         [{"games": [1]}, {"pos": "WR"}, {"pos": "TE"}])
        self.assertEqual(json.loads(out[2]["args_json"]), {"position": "TE"})

    def test_huge_results_are_capped_and_marked(self):
        with mock.patch.object(rr, "MAX_RESULT_CHARS", 50):
            out = rr.pair_tool_calls([{"name": "t", "args": {}}],
                                     [{"name": "t", "response": {"x": "y" * 500}}])
        self.assertTrue(out[0]["truncated"])
        self.assertEqual(len(out[0]["result_json"]), 50)
        self.assertGreater(out[0]["result_chars"], 500)


if __name__ == "__main__":
    unittest.main()
