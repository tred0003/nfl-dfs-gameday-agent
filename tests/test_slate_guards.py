"""Guards from the 2026-11-01 wrong-slate report: python -m unittest discover -s tests

At 12:31 ET the main slate's contests had left DK's lobby, the picker took a
3-game "Afternoon Only" slate, and a Jets injury came back as a Chiefs one.
These pin the three fixes: the picker skips side slates and prefers the slate
the projection job saved, and news drops players it cannot place when the
article's team was only a name-match guess.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import dk_client  # noqa: E402

try:
    from gameday_agent import news  # noqa: E402
except ImportError:  # ADK not installed in this interpreter
    news = None

NOW = datetime(2026, 11, 1, 17, 31, tzinfo=timezone.utc)


def lobby(*rows):
    return pd.DataFrame([{"draft_group_id": dg, "game_type": "Classic", "slate_label": label,
                          "start_time_utc": NOW, "example_contest": name, "top_prize": prize}
                         for dg, label, prize, name in rows])


def draftables(n_games, kickoff=NOW):
    comps = [{"startTime": (kickoff + timedelta(hours=3 * (i % 2))).isoformat()}
             for i in range(n_games)]
    return {"draftables": [{}] * (40 * n_games), "competitions": comps}


class PickMainSlateTest(unittest.TestCase):
    def pick(self, slates, by_dg, saved=None):
        with mock.patch.object(dk_client, "saved_main_slate", return_value=saved), \
             mock.patch.object(dk_client, "_get_json",
                               side_effect=lambda url: by_dg[int(url.split("/")[-2])]), \
             mock.patch.dict("os.environ", {"PIN_DRAFT_GROUP": ""}):
            return dk_client.pick_main_slate(slates)

    def test_a_rich_side_slate_with_three_games_is_skipped(self):
        slates = lobby((155593, "Sun 4:05PM", 500_000, "Afternoon Only Rush"),
                       (153728, "Sun 1:00PM", 20_000, "Main slate small contest"))
        dg = self.pick(slates, {155593: draftables(3), 153728: draftables(11)})
        self.assertEqual(dg, 153728)

    def test_no_main_slate_left_raises_instead_of_guessing(self):
        slates = lobby((155593, "Sun 4:05PM", 500_000, "Afternoon Only Rush"))
        with self.assertRaises(RuntimeError):
            self.pick(slates, {155593: draftables(3)})

    def test_the_saved_slate_wins_over_the_lobby(self):
        slates = lobby((155593, "Sun 4:05PM", 500_000, "Afternoon Only Rush"))
        self.assertEqual(self.pick(slates, {}, saved=153728), 153728)


class SavedMainSlateTest(unittest.TestCase):
    def saved(self, now, comps, rows=({"draft_group_id": 153728},)):
        client = mock.MagicMock()
        client.query.return_value.result.return_value = list(rows)
        with mock.patch("google.cloud.bigquery.Client", return_value=client), \
             mock.patch.object(dk_client, "_get_json", return_value=comps):
            return dk_client.saved_main_slate(now=now)

    def test_current_during_sunday(self):
        self.assertEqual(self.saved(NOW, draftables(11, kickoff=NOW - timedelta(minutes=30))),
                         153728)

    def test_stale_by_monday_morning(self):
        monday = NOW + timedelta(hours=20)
        self.assertIsNone(self.saved(monday, draftables(11, kickoff=NOW - timedelta(minutes=30))))

    def test_no_recent_run(self):
        self.assertIsNone(self.saved(NOW, draftables(11), rows=()))

    def test_any_failure_falls_back_quietly(self):
        with mock.patch("google.cloud.bigquery.Client", side_effect=RuntimeError("no creds")):
            self.assertIsNone(dk_client.saved_main_slate(now=NOW))


@unittest.skipIf(news is None, "gameday_agent needs ADK")
class VerifyTeamTest(unittest.TestCase):
    QUOTE = "Breece Hall will not be suiting up for the New York Jets today."

    def article(self, source):
        return {"title": "Jets vs. Raiders", "text": self.QUOTE + " Next week: the Kansas City Chiefs.",
                "team": "KC", "source": source, "url": "u",
                "published": NOW}

    def claim(self, player="Breece Hall"):
        return {"article": 0, "player": player, "team": "KC", "kind": "injury",
                "claim": "Hall is out.", "quote": self.QUOTE}

    def test_unplaceable_player_from_a_league_feed_is_dropped(self):
        kept, _ = news.verify([self.claim()], [self.article("Yahoo Sports")], ["KC"], {})
        self.assertEqual(kept, [])

    def test_unplaceable_player_from_the_teams_own_feed_is_kept(self):
        kept, _ = news.verify([self.claim()], [self.article("ESPN")], ["KC"], {})
        self.assertEqual([c["team"] for c in kept], ["KC"])

    def test_slate_team_still_decides_for_known_players(self):
        team_of = {"breece hall": "NYJ"}
        kept, _ = news.verify([self.claim()], [self.article("ESPN")], ["KC"], team_of)
        self.assertEqual(kept, [])
        kept, _ = news.verify([self.claim()], [self.article("Yahoo Sports")], ["KC", "NYJ"], team_of)
        self.assertEqual([c["team"] for c in kept], ["NYJ"])


if __name__ == "__main__":
    unittest.main()
