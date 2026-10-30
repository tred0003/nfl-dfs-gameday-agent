"""Tests for the screenshot lineup import: python -m unittest tests.test_lineup_import

Offline. A fake reader stands in for Document AI and returns the OCR lines a
DK lineup screen gives (the line list below is the shape the real OCR
processor returned for a rendered DK-style screenshot on 2026-10-30).

The rules under test: exact slate names only (never a first-initial guess),
suffixes and punctuation don't matter, two defenses is a guess so neither is
used, the upload is checked before it is sent anywhere, and nothing is saved.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "webapp"),
                str(Path(__file__).resolve().parents[1] / "src")]

from lineup_import import LineupImportError, check_upload, match_lineup  # noqa: E402
from main import create_app  # noqa: E402
from user_store import MemoryStore, User  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64

POOL = [{"name": n, "position": pos, "team": t, "salary": s} for n, pos, t, s in [
    ("Michael Penix Jr.", "QB", "ATL", 5300), ("Aaron Jones Sr.", "RB", "MIN", 6200),
    ("J.K. Dobbins", "RB", "DEN", 5300), ("Amon-Ra St. Brown", "WR", "DET", 8000),
    ("Ja'Marr Chase", "WR", "CIN", 7700), ("A.J. Brown", "WR", "NE", 5900),
    ("Kyle Pitts Sr.", "TE", "ATL", 4500), ("Jahmyr Gibbs", "RB", "DET", 9500),
    ("Bengals", "DST", "CIN", 3800), ("Titans", "DST", "TEN", 2900),
    ("Jalon Daniels", "QB", "KC", 4000), ("Justin Jefferson", "WR", "MIN", 8200),
]]

SCREEN = [(t, 0.98) for t in [
    "Lineup", "NFL Classic $50,000 Salary Cap", "Rem. Salary $1,200 Avg/Player $5,422",
    "Michael Penix Jr.", "$5,300", "QB", "ATL @TB Sun 1:00 PM", "12.6 FPPG",
    "Aaron Jones Sr.", "$6,200", "RB", "MIN @ DET Sun 1:00 PM",
    "J.K. Dobbins", "RB", "DEN @KC Sun 1:00 PM",
    "Amon-Ra St. Brown", "WR", "Ja'Marr Chase", "WR", "A.J. Brown", "NE @ MIA Sun 1:00 PM",
    "TE", "Kyle Pitts Sr.", "FLEX", "Jahmyr Gibbs", "DST", "Bengals", "CIN @TEN Sun 1:00 PM",
]]


def names(out: dict) -> list[str]:
    return [p["name"] for p in out["players"]]


class MatchTest(unittest.TestCase):
    def test_a_full_dk_screen_gives_nine_players_in_screen_order(self):
        out = match_lineup(SCREEN, POOL)
        self.assertEqual(names(out), ["Michael Penix Jr.", "Aaron Jones Sr.", "J.K. Dobbins",
                                      "Amon-Ra St. Brown", "Ja'Marr Chase", "A.J. Brown",
                                      "Kyle Pitts Sr.", "Jahmyr Gibbs", "Bengals"])
        self.assertEqual((out["warnings"], out["not_recognised"]), ([], []))

    def test_suffixes_and_punctuation_do_not_matter(self):
        out = match_lineup([("Aaron Jones", 0.9), ("AJ Brown", 0.9), ("Jamarr Chase", 0.9),
                            ("Amon Ra St Brown", 0.9)], POOL)
        self.assertEqual(names(out), ["Aaron Jones Sr.", "A.J. Brown", "Ja'Marr Chase",
                                      "Amon-Ra St. Brown"])

    def test_a_first_initial_is_never_guessed(self):
        # The app.js lesson: "J Daniels" must not become Jalon Daniels.
        out = match_lineup([("J. Daniels", 0.9), ("Jayden Daniels", 0.9)], POOL)
        self.assertEqual(names(out), [])
        # Both are shown to the user as not recognised, to fix by hand.
        self.assertEqual(out["not_recognised"], ["J. Daniels", "Jayden Daniels"])

    def test_a_longer_name_is_not_also_read_as_a_shorter_one(self):
        pool = POOL + [{"name": "St. Brown", "position": "WR", "team": "X", "salary": 1}]
        self.assertEqual(names(match_lineup([("Amon-Ra St. Brown", 0.9)], pool)),
                         ["Amon-Ra St. Brown"])

    def test_a_name_split_over_two_lines_is_joined(self):
        self.assertEqual(names(match_lineup([("Justin", 0.9), ("Jefferson", 0.8)], POOL)),
                         ["Justin Jefferson"])

    def test_two_defenses_is_a_guess_so_neither_is_used(self):
        out = match_lineup([("Jahmyr Gibbs", 0.9), ("Bengals", 0.9), ("Titans", 0.9)], POOL)
        self.assertEqual(names(out), ["Jahmyr Gibbs"])
        self.assertIn("More than one defense", out["warnings"][0])

    def test_a_player_seen_twice_counts_once_and_ten_are_cut_to_nine(self):
        lines = [(p["name"], 0.9) for p in POOL if p["position"] != "DST"] + [("Bengals", 0.9)]
        out = match_lineup(lines + lines[:2], POOL)
        self.assertEqual(len(out["players"]), 9)
        self.assertIn("kept the first 9", out["warnings"][0])

    def test_nothing_found_says_so(self):
        out = match_lineup([("My Contests", 0.9), ("Upcoming", 0.9)], POOL)
        self.assertEqual(out["players"], [])
        self.assertIn("No player", out["warnings"][0])
        self.assertEqual(out["not_recognised"], [])


class UploadTest(unittest.TestCase):
    def test_types_sizes_and_magic_bytes(self):
        self.assertEqual(check_upload(PNG, "image/png"), "image/png")
        self.assertEqual(check_upload(b"\xff\xd8\xff" + b"0" * 10, "image/jpg"), "image/jpeg")
        for data, mime in [(PNG, "text/html"), (b"", "image/png"), (b"<svg>", "image/png"),
                           (b"\x89PNG\r\n\x1a\n" + b"0" * (5 * 1024 * 1024), "image/png")]:
            with self.assertRaises(LineupImportError):
                check_upload(data, mime)


class FakeSlate:
    def pool(self):
        return {"run_id": "r1", "week": 8, "players": POOL}


class FakeReader:
    def __init__(self, lines=SCREEN, fail=False):
        self.lines_out, self.fail, self.calls = lines, fail, 0

    def lines(self, data, mime):
        self.calls += 1
        if self.fail:
            raise LineupImportError("DeadlineExceeded: 30s")
        return self.lines_out


def fake_verify(token: str) -> str:
    if not token.startswith("token-"):
        raise ValueError("signature check failed")
    return token.removeprefix("token-")


def bearer(uid: str) -> dict:
    return {"Authorization": f"Bearer token-{uid}"}


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore([User(uid="u1", name="one", lineup=["Old Guy"])])
        self.reader = FakeReader()
        self.client = self.app(self.reader)

    def app(self, reader):
        return create_app(store=self.store, verify=fake_verify, slate=FakeSlate(), news=object(),
                          agent=object(), reader=reader).test_client()

    def post(self, data=PNG, mime="image/png", uid="u1"):
        return self.client.post("/api/lineup/import", data=data, content_type=mime,
                                headers=bearer(uid) if uid else {})

    def test_returns_the_players_and_saves_nothing(self):
        r = self.post()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json["players"]), 9)
        self.assertEqual(r.json["week"], 8)
        self.assertEqual(self.store.get_user("u1").lineup, ["Old Guy"])

    def test_needs_a_signed_in_user(self):
        self.assertEqual(self.post(uid=None).status_code, 401)
        self.assertEqual(self.reader.calls, 0)

    def test_a_bad_file_is_400_and_never_sent_to_document_ai(self):
        for data, mime in [(b"<html>", "image/png"), (PNG, "text/plain")]:
            self.assertEqual(self.post(data, mime).status_code, 400)
        self.assertEqual(self.reader.calls, 0)

    def test_an_unreadable_image_is_503_not_an_empty_lineup(self):
        r = self.app(FakeReader(fail=True)).post("/api/lineup/import", data=PNG,
                                                 content_type="image/png", headers=bearer("u1"))
        self.assertEqual(r.status_code, 503)
        self.assertIn("could not be read", r.json["error"])

    def test_daily_limit(self):
        for _ in range(20):
            self.assertEqual(self.post().status_code, 200)
        self.assertEqual(self.post().status_code, 429)


if __name__ == "__main__":
    unittest.main()
