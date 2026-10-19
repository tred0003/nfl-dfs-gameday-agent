"""Tests for the web app's auth and isolation: python -m unittest discover -s tests

Offline: a MemoryStore stands in for Firestore and a fake verifier stands in
for Firebase. The fake maps a token string to a uid exactly the way the real
one does after checking the signature -- so "token-u2" is "a valid token for
u2", and anything else is a forged or expired one.

The point of these tests is the isolation rule: who you are comes from the
verified token, and nothing a client sends can change it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))
from main import create_app  # noqa: E402
from slate import SlateError, solve  # noqa: E402
from user_store import MemoryStore, User  # noqa: E402


def fake_verify(token: str) -> str:
    if not token.startswith("token-"):
        raise ValueError("signature check failed")
    return token.removeprefix("token-")


def bearer(uid: str) -> dict:
    return {"Authorization": f"Bearer token-{uid}"}


class FakeSlate:
    """Stands in for BigQuery. Counts calls so the caching contract is visible."""

    def __init__(self, fail: bool = False):
        self.fail = fail

    def pool(self):
        if self.fail:
            raise SlateError("no projection run found")
        return {"run_id": "r1", "week": 7, "players": [{"name": "Drake Maye"}]}

    def suggested(self):
        if self.fail:
            raise SlateError("no projection run found")
        return {"run_id": "r1", "week": 7, "lineups": []}


class WebAppTest(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore([User(uid="u1", name="one", lineup=["A1", "A2"]),
                                  User(uid="u2", name="two", lineup=["B1", "B2"])])
        self.store.save_report("u1", "2026-10-16", {"text": "u1 secret report",
                                                    "source": "agent"})
        self.client = create_app(store=self.store, verify=fake_verify,
                                 slate=FakeSlate()).test_client()

    # ---- who you are -------------------------------------------------------
    def test_no_token_is_401(self):
        self.assertEqual(self.client.get("/api/me").status_code, 401)

    def test_forged_token_is_401(self):
        r = self.client.get("/api/me", headers={"Authorization": "Bearer i-am-u1"})
        self.assertEqual(r.status_code, 401)

    def test_signed_in_but_not_provisioned_is_403(self):
        self.assertEqual(self.client.get("/api/me", headers=bearer("u9")).status_code, 403)

    def test_me_returns_only_your_own_lineup_and_report(self):
        me = self.client.get("/api/me", headers=bearer("u1")).get_json()
        self.assertEqual((me["uid"], me["lineup"]), ("u1", ["A1", "A2"]))
        self.assertEqual(me["report"]["text"], "u1 secret report")

    # ---- user2 cannot reach user1 -----------------------------------------
    def test_u2_never_sees_u1s_report(self):
        me = self.client.get("/api/me?uid=u1", headers=bearer("u2")).get_json()
        self.assertEqual(me["uid"], "u2")
        self.assertIsNone(me["report"])

    def test_uid_in_the_body_is_ignored(self):
        r = self.client.put("/api/lineup", headers=bearer("u2"),
                            json={"uid": "u1", "lineup": ["X1"]})
        self.assertEqual(r.get_json()["uid"], "u2")
        self.assertEqual(self.store.get_user("u1").lineup, ["A1", "A2"])
        self.assertEqual(self.store.get_user("u2").lineup, ["X1"])

    # ---- lineup edits -------------------------------------------------------
    def test_lineup_is_trimmed_and_blanks_dropped(self):
        r = self.client.put("/api/lineup", headers=bearer("u1"),
                            json={"lineup": [" Drake Maye ", "", "Raiders"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.store.get_user("u1").lineup, ["Drake Maye", "Raiders"])

    def test_empty_lineup_is_rejected(self):
        r = self.client.put("/api/lineup", headers=bearer("u1"), json={"lineup": ["", " "]})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.store.get_user("u1").lineup, ["A1", "A2"])

    def test_bad_shapes_are_rejected(self):
        for body in ({"lineup": "A1,A2"}, {"lineup": [1, 2]}, {},
                     {"lineup": ["x" * 61]}, {"lineup": ["p"] * 13}):
            r = self.client.put("/api/lineup", headers=bearer("u1"), json=body)
            self.assertEqual(r.status_code, 400, body)


    # ---- slate and suggestions: shared, but still behind sign-in -------------
    def test_slate_needs_a_signed_in_provisioned_user(self):
        for path in ("/api/slate", "/api/suggested"):
            self.assertEqual(self.client.get(path).status_code, 401, path)
            self.assertEqual(self.client.get(path, headers=bearer("u9")).status_code, 403, path)

    def test_slate_and_suggested_return_the_shared_pool(self):
        r = self.client.get("/api/slate", headers=bearer("u1")).get_json()
        self.assertEqual(r["players"][0]["name"], "Drake Maye")
        self.assertEqual(self.client.get("/api/suggested", headers=bearer("u2")).get_json()["week"], 7)

    def test_unreadable_slate_is_503_not_an_empty_pool(self):
        client = create_app(store=self.store, verify=fake_verify,
                            slate=FakeSlate(fail=True)).test_client()
        r = client.get("/api/slate", headers=bearer("u1"))
        self.assertEqual(r.status_code, 503)
        self.assertIn("could not be loaded", r.get_json()["error"])


def _player(name, pos, team, opp, salary, proj, status=None):
    return {"name": name, "position": pos, "team": team, "opp_team": opp,
            "game": " @ ".join(sorted([team, opp])), "salary": salary,
            "status": status, "projection": proj, "ceiling": None, "value": None}


class SolveTest(unittest.TestCase):
    """The suggestion solver: legal DK lineups, and never a ruled-out player."""

    def pool(self):
        # Salaries high enough to reach optimize()'s $49,000 floor.
        teams =[("AAA", "BBB"), ("BBB", "AAA"), ("CCC", "DDD"), ("DDD", "CCC")]
        out = []
        for t, o in teams:
            out.append(_player(f"{t} QB", "QB", t, o, 7000, 18.0))
            out += [_player(f"{t} RB{i}", "RB", t, o, 6500, 12.0 - i) for i in range(2)]
            out += [_player(f"{t} WR{i}", "WR", t, o, 6500, 11.0 - i) for i in range(3)]
            out.append(_player(f"{t} TE", "TE", t, o, 3500, 7.0))
            out.append(_player(f"{t} DST", "DST", t, o, 2500, 6.0))
        # The best projection on the slate, ruled out: must never be suggested.
        out.append(_player("Hurt Star", "WR", "AAA", "BBB", 3000, 40.0, status="O"))
        return out

    def test_lineups_are_legal_and_skip_ruled_out_players(self):
        lineups = solve(self.pool(), 2)
        self.assertEqual(len(lineups), 2)
        for lu in lineups:
            names = [p["name"] for p in lu["players"]]
            self.assertEqual(len(names), 9)
            self.assertNotIn("Hurt Star", names)
            self.assertLessEqual(lu["salary"], 50_000)
            pos = [p["position"] for p in lu["players"]]
            self.assertEqual((pos.count("QB"), pos.count("DST")), (1, 1))


if __name__ == "__main__":
    unittest.main()
