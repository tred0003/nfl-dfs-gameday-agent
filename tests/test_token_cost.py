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
from token_cost import (  # noqa: E402
    add_usage, cost_usd, new_totals, subcall_cost, subcall_record, subcalls_from_state,
)


class TokenCostTest(unittest.TestCase):
    def test_sums_every_model_call_from_dicts_and_objects(self):
        t = new_totals()
        add_usage(t, {"prompt_token_count": 1000, "candidates_token_count": 50,
                      "thoughts_token_count": 200})
        add_usage(t, SimpleNamespace(prompt_token_count=3000, candidates_token_count=400,
                                     thoughts_token_count=None))
        add_usage(t, {"promptTokenCount": 10, "candidatesTokenCount": 5})
        self.assertEqual(t, {"input": 4010, "cached": 0, "output": 455, "thinking": 200,
                             "model_calls": 3})

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

    def test_cached_input_is_billed_at_a_tenth(self):
        # prompt_token_count INCLUDES the cached tokens: 1M prompt, 600K of
        # it cached = 400K at $0.30 + 600K at $0.03.
        t = new_totals()
        add_usage(t, {"prompt_token_count": 1_000_000, "cached_content_token_count": 600_000,
                      "candidates_token_count": 0})
        self.assertEqual(t["cached"], 600_000)
        self.assertEqual(t["model_calls"], 1)
        self.assertAlmostEqual(cost_usd(t, "gemini-2.5-flash"), 0.12 + 0.018)

    def test_a_usage_with_only_cached_tokens_is_not_a_model_call(self):
        t = new_totals()
        add_usage(t, {"cached_content_token_count": 50})
        self.assertEqual(t["model_calls"], 0)

    def test_subcalls_are_read_back_from_state_deltas(self):
        rec = subcall_record("injection_screen", "gemini-2.5-flash",
                             {"prompt_token_count": 2000, "candidates_token_count": 100})
        self.assertEqual(rec["label"], "injection_screen")
        delta = {"llm_subcalls:call-1": [rec], "llm_subcalls:call-2": [rec],
                 "run_source_warnings": ["not a subcall"]}
        calls = subcalls_from_state(delta)
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(subcall_cost(calls), 2 * rec["cost_usd"])
        self.assertEqual(subcalls_from_state(None), [])

    def test_unknown_model_has_no_price_rather_than_a_wrong_one(self):
        self.assertIsNone(cost_usd({"input": 5}, "gemini-9-ultra"))


if __name__ == "__main__":
    unittest.main()
