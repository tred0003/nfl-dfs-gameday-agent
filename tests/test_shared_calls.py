"""Tests for the work shared between users: python -m unittest discover -s tests

The screen verdict cache, per-team news extraction and the claim-check cache
all exist to avoid paying twice for the same text. What they must never do:
reuse a FAILED answer, hand back another team's claims, or point a claim at
the wrong article once per-team indices are mapped back.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    from gameday_agent import injection, news  # noqa: E402
except ImportError:  # ADK not installed in this interpreter
    injection = news = None


class _Client:
    """Answers generate_content from a function of the prompt, counting calls."""

    def __init__(self, answer):
        self.calls, self._answer = [], answer
        self.models = self

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(contents)
        parsed = self._answer(contents)
        if isinstance(parsed, Exception):
            raise parsed
        return SimpleNamespace(parsed=parsed, usage_metadata=None)


@unittest.skipIf(injection is None, "gameday_agent needs ADK")
class ScreenCacheTest(unittest.TestCase):
    def setUp(self):
        injection._VERDICTS.clear()

    def test_a_text_is_screened_once(self):
        client = _Client(lambda _: [{"item": 0, "verdict": "clean"}])
        injection.screen(["Feeney limited Wednesday."], client=client)
        again = injection.screen(["Feeney limited Wednesday."], client=client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(again[0]["verdict"], "clean")

    def test_only_new_texts_are_sent(self):
        client = _Client(lambda p: [{"item": i, "verdict": "clean"} for i in range(p.count("<item "))])
        injection.screen(["a" * 30], client=client)
        injection.screen(["a" * 30, "b" * 30], client=client)
        self.assertEqual(len(client.calls), 2)
        self.assertNotIn("a" * 30, client.calls[1])

    def test_a_failed_screen_is_not_remembered(self):
        broken = _Client(lambda _: RuntimeError("429"))
        self.assertEqual(injection.screen(["x" * 30], client=broken)[0]["verdict"], "unchecked")
        ok = _Client(lambda _: [{"item": 0, "verdict": "instruction", "evidence": "x"}])
        self.assertEqual(injection.screen(["x" * 30], client=ok)[0]["verdict"], "instruction")
        self.assertEqual(len(ok.calls), 1)

    def test_cache_false_always_asks(self):
        client = _Client(lambda _: [{"item": 0, "verdict": "clean"}])
        injection.screen(["y" * 30], client=client, cache=False)
        injection.screen(["y" * 30], client=client, cache=False)
        self.assertEqual(len(client.calls), 2)


@unittest.skipIf(news is None, "gameday_agent needs ADK")
class SharedExtractionTest(unittest.TestCase):
    ARTICLES = [
        {"team": "DET", "source": "ESPN", "title": "Lions note", "text": "Petty was limited Wednesday."},
        {"team": "MIN", "source": "ESPN", "title": "Vikings note", "text": "Marsh took first-team reps."},
        {"team": "DET", "source": "ESPN", "title": "Lions two", "text": "Goff practiced in full."},
    ]

    def setUp(self):
        news._SHARED.clear()

    def _answer(self, prompt):
        # Every team's call returns one claim about its LAST article.
        n = prompt.count("<article ")
        team = prompt.split("The team of interest: ")[1].split(".")[0]
        return [{"article": n - 1, "player": "X", "team": team, "kind": "other",
                 "claim": "c", "quote": "q"}]

    def test_one_call_per_team_with_indices_mapped_back(self):
        client = _Client(self._answer)
        with mock.patch.object(news, "_client", lambda: client):
            claims, failed = news._extract([dict(a) for a in self.ARTICLES])
        self.assertEqual(failed, [])
        self.assertEqual(len(client.calls), 2)
        by_team = {c["team"]: c["article"] for c in claims}
        # DET's last article is index 2 overall; MIN's only one is index 1.
        self.assertEqual(by_team, {"DET": 2, "MIN": 1})
        # The roster is no longer in the prompt: the call is the same for anyone.
        self.assertTrue(all("rosters" not in p for p in client.calls))

    def test_the_same_team_and_articles_are_not_extracted_twice(self):
        client = _Client(self._answer)
        with mock.patch.object(news, "_client", lambda: client):
            news._extract([dict(a) for a in self.ARTICLES])
            news._extract([dict(self.ARTICLES[1])])  # MIN again, same article
        self.assertEqual(len(client.calls), 2)

    def test_one_failing_team_does_not_sink_the_rest(self):
        def answer(prompt):
            return RuntimeError("boom") if "MIN" in prompt.split("<article")[0] else self._answer(prompt)
        client = _Client(answer)
        with mock.patch.object(news, "_client", lambda: client):
            claims, failed = news._extract([dict(a) for a in self.ARTICLES])
        self.assertEqual([c["team"] for c in claims], ["DET"])
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0].startswith("MIN"))

    def test_claim_check_verdicts_are_shared_and_failures_are_not(self):
        claims = [{"claim": "Petty limited", "quote": "Petty was limited Wednesday."}]
        ok = _Client(lambda _: [{"item": 0, "verdict": "supports"}])
        with mock.patch.object(news, "_client", lambda: ok):
            out1, _ = news.check_claims([dict(c) for c in claims])
            out2, _ = news.check_claims([dict(c) for c in claims])
        self.assertEqual(len(ok.calls), 1)
        self.assertEqual(out2[0]["claim_check"], "supports")
        news._SHARED.clear()
        broken = _Client(lambda _: RuntimeError("429"))
        with mock.patch.object(news, "_client", lambda: broken):
            out, replaced = news.check_claims([dict(c) for c in claims])
        self.assertEqual(replaced, 1)              # fail closed: claim becomes its quote
        self.assertEqual(news._SHARED, {})         # and nothing was remembered


if __name__ == "__main__":
    unittest.main()
