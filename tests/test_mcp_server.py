"""Tests for the news-archive MCP server: python -m unittest discover -s tests

Driven through a real MCP client connected in-process, so what is pinned down
is what Claude Code would see, not just what the Python functions return:
  - exactly two tools, both marked read-only
  - player_history returns every name asked for, in order, [] included, and
    shares archive.player_notes (the agent's dedupe, dates and fencing)
  - bad input and a failed read come back as status "error", not a crash
  - search_news_archive has no per-run limit here (no run to count against)

BigQuery and the embedder are replaced by fakes. Skipped where the MCP SDK or
ADK is not installed -- CI installs neither, same as test_archive.py.
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    from mcp import Client

    from gameday_agent import archive
    from mcp_server import server
except ImportError:
    server = None

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


def _row(key, days_ago, text):
    return {"player_key": key, "published": NOW - timedelta(days=days_ago),
            "source": "ESPN fantasy (RotoWire)", "headline": "h", "text": text}


def _call(name, args):
    async def go():
        async with Client(server.mcp) as c:
            r = await c.call_tool(name, args)
            return json.loads(r.content[0].text)
    return asyncio.run(go())


@unittest.skipIf(server is None, "needs the mcp SDK and ADK")
class McpServerTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(archive, "_now", lambda: NOW)
        p.start()
        self.addCleanup(p.stop)

    def test_two_read_only_tools(self):
        async def go():
            async with Client(server.mcp) as c:
                return (await c.list_tools()).tools
        tools = asyncio.run(go())
        self.assertEqual(sorted(t.name for t in tools), ["player_history", "search_news_archive"])
        self.assertTrue(all(t.annotations.read_only_hint for t in tools))

    def test_history_keeps_every_name_in_order(self):
        key = archive.name_keys(["Kenneth Walker III"])[0]
        rows = [_row(key, 1, "Limited Wednesday."), _row(key, 1, "Limited Wednesday."),
                _row(key, 8, "Limited last week too.")]
        with mock.patch.object(archive, "_history_rows", return_value=(rows, NOW)) as q:
            out = _call("player_history", {"players": ["Nobody Here", "Kenneth Walker III",
                                                       "Kenneth Walker III"]})
        self.assertEqual(out["status"], "ok")
        self.assertEqual([p["player"] for p in out["players"]], ["Nobody Here", "Kenneth Walker III"])
        self.assertEqual(out["players"][0]["notes"], [])
        notes = out["players"][1]["notes"]
        self.assertEqual([n["days_ago"] for n in notes], [1, 8])     # deduped, newest first
        self.assertIn("Archived note", notes[0]["text"])             # fenced as third-party
        self.assertEqual(q.call_count, 1)                            # one query for all names

    def test_bad_input_and_read_failure_are_errors(self):
        self.assertEqual(_call("player_history", {"players": [" "]})["status"], "error")
        many = [f"Player {i}" for i in range(server.MAX_PLAYERS + 1)]
        self.assertEqual(_call("player_history", {"players": many})["status"], "error")
        with mock.patch.object(archive, "_history_rows", side_effect=RuntimeError("bq down")):
            out = _call("player_history", {"players": ["Kenneth Walker III"]})
        self.assertEqual(out["status"], "error")
        self.assertIn("bq down", out["message"])

    def test_search_has_no_run_limit(self):
        row = {"published": NOW - timedelta(days=2), "kind": "story", "source": "ESPN",
               "headline": "h", "text": "Split backfield since Week 4.", "url": None}
        with mock.patch.object(archive, "_search_rows", return_value=[row]):
            outs = [_call("search_news_archive", {"question": "backfield split?"})
                    for _ in range(archive.SEARCH_LIMIT + 1)]
        self.assertEqual({o["status"] for o in outs}, {"ok"})


if __name__ == "__main__":
    unittest.main()
