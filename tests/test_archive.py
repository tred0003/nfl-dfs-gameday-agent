"""Tests for the news-archive tools and refresh: python -m unittest discover -s tests

What these pin down, with BigQuery replaced by fakes:
  - a DK name and an ESPN name meet on one player_keys spelling
  - get_player_history keeps every rostered player (an empty list is an
    answer), drops the DST, de-duplicates a note tagged twice, dates every
    note and fences its text
  - a stale archive says so, and a failed read is an error, not "no news"
  - search_news_archive stops at its limit and says "no_results" honestly
  - the re-ranker's order is used, and a failure or the daily cap falls back
    to the embedding's order instead of breaking search
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

    def test_passages_carry_ids_and_the_search_its_version_and_cost(self):
        rows = [_row(None, 1, "a") | {"chunk_id": "x#0", "url": None}]

        def fake(q, player, cutoff, days, k, recent, stats, *_):
            stats.update(ranked=True, fallback=None, query_chars=1000, bq_bytes=0,
                         ms={"embed": 5})
            return rows

        with mock.patch.object(archive, "_search_rows", fake):
            out = archive.search_news_archive("q", tool_context=SimpleNamespace(state={}))
        self.assertEqual(out["passages"][0]["id"], "x#0")
        self.assertEqual(out["meta"]["retriever"], archive.RETRIEVER_VERSION)
        self.assertIn("rank=semantic-ranker-default-005", archive.RETRIEVER_VERSION)
        self.assertNotIn("@latest", archive.RERANK_MODEL)        # pinned, never floating
        self.assertTrue(out["meta"]["ranked"])
        self.assertAlmostEqual(out["meta"]["cost_usd"],
                               archive.RERANK_USD + archive.EMBED_USD_PER_KCHAR, places=6)

    def test_podcast_search_reads_its_own_table_with_its_own_limit(self):
        tables = []

        def fake(q, player, cutoff, days, k, recent, stats, table):
            tables.append(table)
            return [_row(None, 2, "x") | {"headline": "The Ringer NFL Show: Week 7",
                                          "text": "The Ringer NFL Show: Week 7. He looks done.",
                                          "chunk_id": "pod:ringer-nfl:abc@600#1", "url": "a.mp3#t=640"}]

        ctx = SimpleNamespace(state={"archive_search_calls": archive.SEARCH_LIMIT})
        with mock.patch.object(archive, "_search_rows", fake):
            out = archive.search_podcasts("is he done?", tool_context=ctx)
            self.assertEqual(archive.search_news_archive("q", tool_context=ctx)["status"], "error")
        self.assertEqual(out["status"], "ok")             # news calls used up, podcasts not
        self.assertEqual(tables, [archive.PODCAST_TABLE])
        self.assertNotEqual(archive.PODCAST_TABLE, archive.TABLE)
        self.assertNotIn("Week 7.", out["passages"][0]["text"])    # prefix is in `headline`
        self.assertIn("He looks done.", out["passages"][0]["text"])
        self.assertEqual(ctx.state["podcast_search_calls"], 1)

    def test_the_search_log_line_goes_to_stderr_never_stdout(self):
        # stdout is the MCP protocol channel under `server.py --stdio`.
        import contextlib
        import io
        import json as _json
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(archive, "_search_rows", lambda *a: []), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            archive.search_news_archive("q", tool_context=SimpleNamespace(state={}))
        self.assertEqual(out.getvalue(), "")
        line = _json.loads(err.getvalue().strip().splitlines()[-1])
        self.assertEqual((line["event"], line["retriever"]), ("archive_search", archive.RETRIEVER_VERSION))


class _FakeRanker:
    def __init__(self, order=None, error=None):
        self.order_, self.error, self.calls = order, error, 0

    def order(self, question, texts, titles):
        self.calls += 1
        self.seen = (question, titles)
        if self.error:
            raise self.error
        return self.order_


@unittest.skipIf(archive is None, "gameday_agent needs ADK")
class RerankTest(unittest.TestCase):
    """The re-ranker may only make search better: any failure keeps the
    embedding's order, and the daily cap stops calls without stopping search."""

    def setUp(self):
        self.rows = [{"text": t, "published": NOW - timedelta(days=d)} for t, d in (("a", 1), ("b", 2), ("c", 30))]
        patches = [mock.patch.object(archive, "_now", lambda: NOW),
                   mock.patch.object(archive, "_rerank_day", None),
                   mock.patch.object(archive, "_rerank_calls", 0)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _with(self, ranker):
        p = mock.patch.object(archive, "_ranker", ranker)
        p.start()
        self.addCleanup(p.stop)

    def test_ranker_order_is_used(self):
        self._with(_FakeRanker(order=[2, 0, 1]))
        self.assertEqual([r["text"] for r in archive._rerank("q", self.rows)], ["c", "a", "b"])

    def test_the_ranker_is_told_the_dates(self):
        ranker = _FakeRanker(order=[0, 1, 2])
        self._with(ranker)
        archive._rerank("When is he back?", self.rows)
        question, titles = ranker.seen
        self.assertEqual(question, "Today is Thu Oct 22, 2026. When is he back?")
        self.assertEqual(titles, ["Published Wed Oct 21, 2026", "Published Tue Oct 20, 2026",
                                  "Published Tue Sep 22, 2026"])

    def test_rows_the_ranker_leaves_out_are_kept_at_the_end(self):
        self._with(_FakeRanker(order=[1]))
        self.assertEqual([r["text"] for r in archive._rerank("q", self.rows)], ["b", "a", "c"])

    def test_the_outcome_is_recorded_for_the_trace(self):
        self._with(_FakeRanker(order=[0, 1, 2]))
        stats = {}
        archive._rerank("q", self.rows, stats)
        self.assertEqual(stats, {"ranked": True, "fallback": None})
        self._with(_FakeRanker(error=TimeoutError("slow")))
        with self.assertLogs(archive.log, "WARNING"):
            archive._rerank("q", self.rows, stats)
        self.assertEqual(stats, {"ranked": False, "fallback": "TimeoutError"})

    def test_a_failure_keeps_the_embedding_order(self):
        self._with(_FakeRanker(error=TimeoutError("slow")))
        with self.assertLogs(archive.log, "WARNING"):
            self.assertEqual(archive._rerank("q", self.rows), self.rows)

    def test_past_the_daily_cap_no_call_is_made(self):
        ranker = _FakeRanker(order=[2, 1, 0])
        self._with(ranker)
        with mock.patch.object(archive, "RERANK_DAILY_CAP", 2), self.assertLogs(archive.log, "WARNING"):
            for _ in range(3):
                out = archive._rerank("q", self.rows)
        self.assertEqual(ranker.calls, 2)
        self.assertEqual(out, self.rows)

    def test_newest_of_the_relevant_goes_first(self):
        # c (score .9, 30 days old) and a (.85, 1 day) are both relevant: a goes
        # first. b (.2) is not relevant and keeps its ranker place after them.
        rows = [{"text": t, "published": NOW - timedelta(days=d), "_score": s}
                for t, d, s in (("c", 30, .9), ("a", 1, .85), ("b", 2, .2))]
        with mock.patch.object(archive, "RECENCY_FRAC", 0.8):
            self.assertEqual([r["text"] for r in archive._recency_first(rows)], ["a", "c", "b"])
        with mock.patch.object(archive, "RECENCY_FRAC", 0.0):          # off
            self.assertEqual([r["text"] for r in archive._recency_first(rows)], ["c", "a", "b"])

    def test_no_scores_means_no_reordering(self):
        rows = [{"text": t, "published": NOW - timedelta(days=d)} for t, d in (("c", 30), ("a", 1))]
        self.assertEqual(archive._recency_first(rows), rows)          # ranker fell back

    def test_ranker_scores_are_kept_on_the_rows(self):
        self._with(_FakeRanker(order=[(2, .9), (0, .5), (1, .1)]))
        out = archive._rerank("q", self.rows)
        self.assertEqual([(r["text"], r["_score"]) for r in out], [("c", .9), ("a", .5), ("b", .1)])

    def test_the_cap_resets_each_day(self):
        ranker = _FakeRanker(order=[0, 1, 2])
        self._with(ranker)
        with mock.patch.object(archive, "RERANK_DAILY_CAP", 1):
            archive._rerank("q", self.rows)
            with mock.patch.object(archive, "_now", lambda: NOW + timedelta(days=1)):
                archive._rerank("q", self.rows)
        self.assertEqual(ranker.calls, 2)


@unittest.skipIf(archive is None, "gameday_agent needs ADK")
class RefreshTest(unittest.TestCase):
    def test_known_chunks_are_not_appended_again(self):
        art = {"id": "blurb-1", "kind": "player_blurb", "source": "s", "published": "2026-10-21T10:00:00Z",
               "headline": "h", "text": "t", "players": [{"espn_id": "1", "name": "Deion Marsh"}],
               "teams": ["MIN"], "url": None}
        self.assertEqual(len(refresh.new_chunks([art], set())), 1)
        self.assertTrue(refresh.new_chunks([art], {"blurb-1"}).empty)

    def test_a_flag_stands_only_when_the_confirm_model_agrees(self):
        from gameday_agent import injection
        from news_rag import index

        def fake(texts, model=None, **_kw):
            first = {"a": "clean", "b": "instruction", "c": "instruction", "d": "instruction"}
            confirm = {"b": "clean", "c": "instruction", "d": "unchecked"}
            table = confirm if model == index.CONFIRM_MODEL else first
            return [{"verdict": table[t], "evidence": ""} for t in texts]

        with mock.patch.object(injection, "screen", fake):
            self.assertEqual(index.screen_texts(["a", "b", "c", "d"]),
                             ["clean", "clean", "instruction", "unchecked"])


if __name__ == "__main__":
    unittest.main()
