"""Tests for the prefetch callback: python -m unittest discover -s tests

Prefetch answers the agent's first model call with the seven core tool
calls -- but only for the standard report question. These pin down that it
fires exactly then, and that its copy of the question still matches the one
the scheduled report sends.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gameday_agent"))
sys.path.insert(0, str(ROOT))

try:
    from google.genai import types

    from gameday_agent import prefetch
except ImportError:  # ADK not installed in this interpreter
    prefetch = None


def _ctx(text):
    return SimpleNamespace(user_content=types.Content(role="user", parts=[types.Part(text=text)]))


def _request(*contents):
    return SimpleNamespace(contents=list(contents))


@unittest.skipIf(prefetch is None, "gameday_agent needs ADK")
class PrefetchTest(unittest.TestCase):
    def test_standard_question_gets_the_seven_core_calls(self):
        resp = prefetch.prefetch_core_tools(_ctx(prefetch.STANDARD_QUERY), _request())
        names = [p.function_call.name for p in resp.content.parts]
        self.assertEqual(names, prefetch.CORE_TOOLS)

    def test_any_other_question_is_left_to_the_model(self):
        q = prefetch.STANDARD_QUERY + " Use search_news to confirm Rashad Feeney's status."
        self.assertIsNone(prefetch.prefetch_core_tools(_ctx(q), _request()))

    def test_only_the_first_model_call(self):
        done = types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(name="get_lineup_status", args={}))])
        self.assertIsNone(prefetch.prefetch_core_tools(_ctx(prefetch.STANDARD_QUERY), _request(done)))

    def test_the_scheduled_report_still_asks_the_standard_question(self):
        import run_report
        self.assertEqual(run_report.QUERY, prefetch.STANDARD_QUERY)


if __name__ == "__main__":
    unittest.main()
