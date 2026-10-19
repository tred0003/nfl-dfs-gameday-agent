"""Tests for in-tool usage metering: python -m unittest discover -s tests

The mechanism rests on one assumption about ADK: a sync tool runs in an
executor with a COPY of the caller's context, so a list set before the tool
is the same list the tool appends to. These pin that down without ADK, and
check that parallel calls land under separate state keys.
"""

from __future__ import annotations

import asyncio
import contextvars
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    from gameday_agent import metering  # noqa: E402
except ImportError as e:  # ADK not installed in this interpreter
    metering = None
    _SKIP = str(e)
from token_cost import subcalls_from_state  # noqa: E402


class _FakeModels:
    def generate_content(self, **kw):
        return SimpleNamespace(usage_metadata={"prompt_token_count": 1000,
                                               "candidates_token_count": 10})


_CLIENT = SimpleNamespace(models=_FakeModels())


def _tool_body():
    metering.generate(_CLIENT, "injection_screen", model="gemini-2.5-flash", contents="x")
    metering.generate(_CLIENT, "news_extract", model="gemini-2.5-flash", contents="y")


@unittest.skipIf(metering is None, "gameday_agent needs ADK")
class MeteringTest(unittest.TestCase):
    def _run_tool(self, state: dict, call_id: str):
        tool = SimpleNamespace(name="get_team_news")
        ctx = SimpleNamespace(state=state, function_call_id=call_id)

        async def one():
            metering.start_subcalls(tool, {}, ctx)
            loop = asyncio.get_running_loop()
            # What ADK does with a sync tool: run it in an executor under a
            # copy of the current context.
            copied = contextvars.copy_context()
            await loop.run_in_executor(None, copied.run, _tool_body)
            metering.collect_subcalls(tool, {}, ctx, {})
        return one()

    def test_calls_inside_a_threaded_tool_reach_state(self):
        state: dict = {}
        asyncio.run(self._run_tool(state, "c1"))
        calls = subcalls_from_state(state)
        self.assertEqual([c["label"] for c in calls], ["injection_screen", "news_extract"])
        self.assertTrue(all(c["tool"] == "get_team_news" for c in calls))

    def test_parallel_tools_keep_separate_keys(self):
        state: dict = {}

        async def both():
            await asyncio.gather(asyncio.create_task(self._run_tool(state, "a")),
                                 asyncio.create_task(self._run_tool(state, "b")))
        asyncio.run(both())
        self.assertEqual(len(subcalls_from_state(state)), 4)

    def test_a_call_outside_any_tool_is_not_recorded(self):
        _tool_body()  # no start_subcalls: nothing to append to, and no error


if __name__ == "__main__":
    unittest.main()
