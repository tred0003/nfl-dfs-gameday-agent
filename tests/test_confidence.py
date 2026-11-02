"""Confidence labels are code, not model output: python -m unittest discover -s tests

The news claim from 2026-11-01 (Breece Hall, a Jet, filed under the Chiefs by a
Yahoo name match) must come out low on every path; a fresh slate-placed claim
from the team's own feed must come out high.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))
import confidence as ac  # noqa: E402  (webapp/confidence.py)

# By file, not `from gameday_agent import`: the package __init__ pulls in ADK,
# which CI does not install. confidence.py itself needs only the stdlib.
_spec = importlib.util.spec_from_file_location("gameday_confidence",
                                               ROOT / "gameday_agent" / "confidence.py")
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)


def claim(**kw):
    base = {"player": "Rashee Rice", "team": "KC", "kind": "injury", "claim": "Limited Wed.",
            "quote": "Rice was limited Wednesday.", "source": "ESPN", "url": "u",
            "placed_by_slate": True, "own_feed": True, "age_h": 6.0, "claim_check": "supports"}
    return base | kw


class ClaimConfidenceTest(unittest.TestCase):
    def test_breece_hall_case_is_low(self):
        label, why = gc.claim_confidence(
            claim(player="Breece Hall", placed_by_slate=False, own_feed=False,
                  source="Yahoo Sports"), sources=1)
        self.assertEqual(label, "low")
        self.assertIn("not on the slate", why[0])

    def test_fresh_own_feed_on_slate_is_high(self):
        self.assertEqual(gc.claim_confidence(claim(), sources=1)[0], "high")

    def test_single_league_feed_is_medium_two_outlets_is_high(self):
        c = claim(own_feed=False, source="ProFootballTalk")
        self.assertEqual(gc.claim_confidence(c, sources=1)[0], "medium")
        self.assertEqual(gc.claim_confidence(c, sources=2)[0], "high")

    def test_stale_or_rewritten_or_undated_is_low(self):
        self.assertEqual(gc.claim_confidence(claim(age_h=50), 3)[0], "low")
        self.assertEqual(gc.claim_confidence(claim(claim_check="contradicts"), 3)[0], "low")
        self.assertEqual(gc.claim_confidence(claim(age_h=None), 3)[0], "low")

    def test_a_day_old_is_medium(self):
        self.assertEqual(gc.claim_confidence(claim(age_h=30), 3)[0], "medium")

    def test_label_claims_logs_and_strips_internal_fields(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = gc.label_claims([claim(), claim(source="RotoWire", own_feed=False)])
        self.assertEqual(buf.getvalue().count('"event": "claim_confidence"'), 2)
        for c in out:
            self.assertNotIn("placed_by_slate", c)
            self.assertNotIn("age_h", c)
            if not gc.VISIBLE:
                self.assertNotIn("confidence", c)


def ev(*items):
    return [{"tool": "t", "items": list(items)}]


def item(ref, source="ESPN", days=1, text="news"):
    return {"ref": ref, "source": source, "days_ago": days, "text": text}


def checked(refs, answer_type="answered", share=1.0, problems=()):
    return {"answer_type": answer_type, "cited_share": share,
            "claims": [{"text": "x", "refs": [r]} for r in refs],
            "problems": [{"kind": k} for k in problems]}


class AnswerConfidenceTest(unittest.TestCase):
    MODEL = item("P1", source="nfl-dfs model (XGBoost)", days=0,
                 text="projection 14.2 DK pts")

    def test_well_sourced_fresh_answer_is_high(self):
        label, _ = ac.answer_confidence(checked(["P1", "P2", "P3"]),
                                        ev(self.MODEL, item("P2"), item("P3", "PFT", 2)))
        self.assertEqual(label, "high")

    def test_abstained_and_unsupported_are_low(self):
        for kind in ("abstained", "unsupported", "empty"):
            self.assertEqual(ac.answer_confidence(checked([], kind), [])[0], "low")

    def test_made_up_ref_is_low(self):
        label, why = ac.answer_confidence(checked(["P2"], problems=["unknown_ref"]), ev(item("P2")))
        self.assertEqual(label, "low")

    def test_one_source_or_old_news_is_medium(self):
        self.assertEqual(ac.answer_confidence(checked(["P2"]), ev(item("P2")))[0], "medium")
        self.assertEqual(ac.answer_confidence(checked(["P2", "P3"]),
                                              ev(item("P2", days=9), item("P3", "PFT", 8)))[0],
                         "medium")

    def test_questionable_player_caps_at_medium(self):
        q = item("P1", source="nfl-dfs model (XGBoost)", days=0,
                 text="projection 14.2 DK pts, injury status Q.")
        label, why = ac.answer_confidence(checked(["P1", "P2", "P3"]),
                                          ev(q, item("P2"), item("P3", "PFT")))
        self.assertEqual(label, "medium")
        self.assertTrue(any("Q" in w for w in why))


if __name__ == "__main__":
    unittest.main()
