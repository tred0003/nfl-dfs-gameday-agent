"""Tests for the news-archive tools and refresh: python -m unittest discover -s tests

What these pin down, with BigQuery replaced by fakes:
  - a DK name and an ESPN name meet on one player_keys spelling
  - get_player_history keeps every rostered player (an empty list is an
    answer), drops the DST, de-duplicates a note tagged twice, dates every
    note and fences its text
  - a stale archive says so, and a failed read is an error, not "no news"
  - search_news_archive stops at its limit and says "no_results" honestly
  - the refresh never re-appends a chunk the table already has
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd

    from gameday_agent import archive, tools
    from gameday_agent.slate import LINEUP_STATE_KEY
    from news_rag import refresh
except ImportError:  # ADK not installed in this interpreter
    archive = None

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


def _slate():
    players = pd.DataFrame([
        {"name": "Kenneth Walker III", "position": "RB", "team": "SEA", "game": "SEA @ LAR", "salary": 6800},
        {"name": "Deion Marsh", "position": "WR", "team": "MIN", "game": "GB @ MIN", "salary": 7300},
        {"name": "Vikings", "position": "DST", "team": "MIN", "game": "GB @ MIN", "salary": 3000},
    ])
    return players, pd.DataFrame()


def _ctx(names):
    return SimpleNamespace(state={LINEUP_STATE_KEY: names})


def _row(key, days_ago, text, hours=0):
    return {"player_key": key, "published": NOW - timedelta(days=days_ago, hours=hours),
            "source": "ESPN fantasy (RotoWire)", "headline": "h", "text": text}


@unittest.skipIf(archive is None, "gameday_agent needs ADK")
class HistoryTest(unittest.TestCase):
    def setUp(self):
        patches = [mock.patch.object(tools, "slate", _slate),
                   mock.patch.object(archive, "_now", lambda: NOW)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _run(self, rows, newest=NOW - timedelta(hours=5)):
        seen = {}

        def fake(keys, cutoff, days, n):
            seen.update(keys=keys, cutoff=cutoff, days=days, n=n)
            return rows, newest

        with mock.patch.object(archive, "_history_rows", fake):
            out = archive.get_player_history(_ctx(["Kenneth Walker III", "Deion Marsh", "MIN"]))
        return out, seen

    def test_dk_and_espn_spellings_share_a_key(self):
        self.assertEqual(archive.name_keys(["Kenneth Walker III"]), archive.name_keys(["Kenneth Walker"]))
        self.assertEqual(archive.name_keys(["A.J. Brown"]), ["aj brown"])

    def test_every_rostered_player_listed_and_dst_left_out(self):
        out, seen = self._run([_row("deion marsh", 1, "Marsh (hamstring) limited Thursday.")])
        self.assertEqual(out["status"], "ok")
        self.assertEqual([p["player"] for p in out["players"]], ["Kenneth Walker III", "Deion Marsh"])
        self.assertEqual(out["players"][0]["notes"], [])
        self.assertEqual(sorted(seen["keys"]), ["deion marsh", "kenneth walker"])
        self.assertEqual(seen["cutoff"], NOW)

    def test_notes_are_dated_fenced_newest_first_and_deduplicated(self):
        out, _ = self._run([
            _row("deion marsh", 15, "Marsh tweaked his hamstring."),
            _row("deion marsh", 1, 'Marsh "limited" again.'),
            _row("deion marsh", 1, 'Marsh "limited" again.', hours=1),
        ])
        notes = out["players"][1]["notes"]
        self.assertEqual([n["days_ago"] for n in notes], [1, 15])
        self.assertIn("third-party text", notes[0]["text"])
        self.assertNotIn('"limited"', notes[0]["text"])   # quotes cannot close the fence
        self.assertNotIn("freshness_warning", out)

    def test_stale_archive_warns(self):
        out, _ = self._run([], newest=NOW - timedelta(hours=60))
        self.assertEqual(out["archive_hours_old"], 60.0)
        self.assertIn("60 hours old", out["freshness_warning"])

    def test_empty_archive_warns(self):
        out, _ = self._run([], newest=None)
        self.assertIsNone(out["archive_hours_old"])
        self.assertIn("stopped", out["freshness_warning"])

    def test_read_failure_is_an_error_not_an_empty_history(self):
        def boom(*_a):
            raise RuntimeError("403 Access Denied")
        with mock.patch.object(archive, "_history_rows", boom):
            out = archive.get_player_history(_ctx(["Deion Marsh"]))
        self.assertEqual(out["status"], "error")
        self.assertIn("could not be checked", out["message"])

    def test_no_lineup_is_an_error(self):
        out = archive.get_player_history(_ctx([]))
        self.assertEqual(out["status"], "error")


@unittest.skipIf(archive is None, "gameday_agent needs ADK")
class SearchTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(archive, "_now", lambda: NOW)
        p.start()
        self.addCleanup(p.stop)

    def test_limit_is_per_run(self):
        ctx = SimpleNamespace(state={})
        with mock.patch.object(archive, "_search_rows", lambda *a: []):
            for _ in range(archive.SEARCH_LIMIT):
                self.assertEqual(archive.search_news_archive("q", tool_context=ctx)["status"], "no_results")
            self.assertEqual(archive.search_news_archive("q", tool_context=ctx)["status"], "error")

    def test_passages_deduplicated_and_newest_first(self):
        rows = [_row(None, 3, "older") | {"url": "u1"}, _row(None, 1, "newer") | {"url": "u2"},
                _row(None, 3, "older") | {"url": "u1"}]
        with mock.patch.object(archive, "_search_rows", lambda *a: rows):
            out = archive.search_news_archive("snap share?", player="Deion Marsh",
                                              tool_context=SimpleNamespace(state={}))
        self.assertEqual(out["status"], "ok")
        self.assertEqual([p["days_ago"] for p in out["passages"]], [1, 3])
        self.assertEqual(out["passages"][0]["url"], "u2")


@unittest.skipIf(archive is None, "gameday_agent needs ADK")
class RefreshTest(unittest.TestCase):
    def test_known_chunks_are_not_appended_again(self):
        art = {"id": "blurb-1", "kind": "player_blurb", "source": "s", "published": "2026-10-21T10:00:00Z",
               "headline": "h", "text": "t", "players": [{"espn_id": "1", "name": "Deion Marsh"}],
               "teams": ["MIN"], "url": None}
        self.assertEqual(len(refresh.new_chunks([art], set())), 1)
        self.assertTrue(refresh.new_chunks([art], {"blurb-1"}).empty)


if __name__ == "__main__":
    unittest.main()
