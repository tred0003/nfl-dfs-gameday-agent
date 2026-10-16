"""Tests for the per-report cost math: python -m unittest discover -s tests

The number this produces gets quoted ("about a cent a report"), so the two
ways it can silently go wrong are pinned down: thinking tokens left out of
the output bill, and the same usage counted from both dict and object shapes.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from token_cost import add_usage, cost_usd, new_totals  # noqa: E402


class TokenCostTest(unittest.TestCase):
    def test_sums_every_model_call_from_dicts_and_objects(self):
        t = new_totals()
        add_usage(t, {"prompt_token_count": 1000, "candidates_token_count": 50,
                      "thoughts_token_count": 200})
        add_usage(t, SimpleNamespace(prompt_token_count=3000, candidates_token_count=400,
                                     thoughts_token_count=None))
        add_usage(t, {"promptTokenCount": 10, "candidatesTokenCount": 5})
        self.assertEqual(t, {"input": 4010, "output": 455, "thinking": 200, "model_calls": 3})

    def test_events_without_usage_are_not_model_calls(self):
        t = new_totals()
        add_usage(t, None)
        add_usage(t, {})
        add_usage(t, {"total_token_count": None})
        self.assertEqual(t["model_calls"], 0)

    def test_thinking_tokens_are_billed_as_output(self):
        # 1M input on Flash = $0.30; 1M output + 1M thinking = 2M at $2.50.
        t = {"input": 1_000_000, "output": 1_000_000, "thinking": 1_000_000}
        self.assertAlmostEqual(cost_usd(t, "gemini-2.5-flash"), 0.30 + 5.00)

    def test_the_real_u3_report_is_about_a_cent(self):
        # The 10/16 production run for u3, as logged.
        t = {"input": 14549, "output": 1495, "thinking": 1214, "model_calls": 2}
        self.assertAlmostEqual(cost_usd(t, "gemini-2.5-flash"), 0.011137, places=6)

    def test_unknown_model_has_no_price_rather_than_a_wrong_one(self):
        self.assertIsNone(cost_usd({"input": 5}, "gemini-9-ultra"))


if __name__ == "__main__":
    unittest.main()
