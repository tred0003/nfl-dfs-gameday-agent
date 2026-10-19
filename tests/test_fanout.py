"""Tests for the per-user report loop: python -m unittest discover -s tests

Offline: a MemoryStore stands in for Firestore and a fake `produce` stands in
for the agent, so this runs in CI with no credentials, no Gemini and no
Discord. What it pins down is the multi-user contract of step 1:

  - each user's report is built from THEIR lineup and stored under THEIR uid
  - one user failing does not stop the others, but does fail the run
  - a user with no lineup gets no report, never an empty "all clear"
  - only users marked discord=True are posted
  - the agent session is opened under the user's own uid
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gameday_agent"))
import run_report as rr  # noqa: E402
from user_store import LineupError, MemoryStore, User  # noqa: E402

DAY = "2026-10-18"


def users() -> list[User]:
    return [User(uid="u1", lineup=["A1", "A2"], discord=True),
            User(uid="u2", lineup=["B1", "B2"]),
            User(uid="u3", lineup=["C1", "C2"])]


def echo(user: User) -> dict:
    """A fake agent: the report is just the lineup it was handed."""
    return {"text": f"{user.uid}: " + ",".join(user.checked_lineup()),
            "source": "agent", "problems": [], "tools": []}


class RunUsersTest(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(users())
        self.sent: list[str] = []
        # The real loop logs every step; keep test output readable.
        patcher = mock.patch.object(rr, "log")
        patcher.start()
        self.addCleanup(patcher.stop)

    def deliver(self, text: str) -> bool:
        self.sent.append(text)
        return True

    def run_all(self, produce=echo, deliver="default", dry_run=False) -> int:
        return rr.run_users(self.store.active_users(), self.store, produce,
                            self.deliver if deliver == "default" else deliver,
                            DAY, dry_run=dry_run)

    def test_each_report_is_built_from_and_stored_under_its_own_user(self):
        self.assertEqual(self.run_all(), 0)
        self.assertEqual(self.store.get_report("u1", DAY)["text"], "u1: A1,A2")
        self.assertEqual(self.store.get_report("u2", DAY)["text"], "u2: B1,B2")
        self.assertEqual(self.store.get_report("u3", DAY)["text"], "u3: C1,C2")

    def test_only_discord_users_are_posted(self):
        self.run_all()
        self.assertEqual(self.sent, ["u1: A1,A2"])

    def test_no_discord_stores_but_posts_nothing(self):
        self.assertEqual(self.run_all(deliver=None), 0)
        self.assertEqual(self.sent, [])
        self.assertIsNotNone(self.store.get_report("u1", DAY))

    def test_empty_lineup_gets_no_report_and_fails_the_run(self):
        self.store.put_user(User(uid="u2", lineup=["  ", ""]))
        self.assertEqual(self.run_all(), 1)
        self.assertIsNone(self.store.get_report("u2", DAY))
        # ...and the users either side of it still got theirs.
        self.assertIsNotNone(self.store.get_report("u1", DAY))
        self.assertIsNotNone(self.store.get_report("u3", DAY))

    def test_one_user_crashing_does_not_stop_the_rest(self):
        def flaky(user):
            if user.uid == "u1":
                raise RuntimeError("agent and fallback both down")
            return echo(user)
        self.assertEqual(self.run_all(produce=flaky), 1)
        self.assertIsNone(self.store.get_report("u1", DAY))
        self.assertIsNotNone(self.store.get_report("u2", DAY))
        self.assertIsNotNone(self.store.get_report("u3", DAY))

    def test_failed_discord_delivery_fails_the_run(self):
        self.assertEqual(self.run_all(deliver=lambda text: False), 1)
        # Stored before posting: the record exists even though Discord failed.
        self.assertIsNotNone(self.store.get_report("u1", DAY))

    def test_dry_run_stores_and_sends_nothing(self):
        with mock.patch("builtins.print"):
            self.assertEqual(self.run_all(dry_run=True), 0)
        self.assertEqual(self.store.reports, {})
        self.assertEqual(self.sent, [])

    def test_identical_lineups_reuse_one_agent_report(self):
        self.store.put_user(User(uid="u3", lineup=["A2", "a1"]))  # u1's players
        calls = []

        def counting(user):
            calls.append(user.uid)
            return echo(user) | {"cost_usd": 0.01, "total_cost_usd": 0.05}
        self.assertEqual(self.run_all(produce=counting), 0)
        self.assertEqual(calls, ["u1", "u2"])
        reused = self.store.get_report("u3", DAY)
        self.assertEqual(reused["text"], "u1: A1,A2")
        self.assertEqual(reused["reused_from"], "u1")
        self.assertEqual(reused["total_cost_usd"], 0.0)
        self.assertEqual(self.store.get_report("u1", DAY)["total_cost_usd"], 0.05)

    def test_a_fallback_report_is_not_reused(self):
        self.store.put_user(User(uid="u3", lineup=["A1", "A2"]))
        calls = []

        def fallback(user):
            calls.append(user.uid)
            return echo(user) | {"source": "fallback"}
        self.run_all(produce=fallback)
        self.assertEqual(calls, ["u1", "u2", "u3"])

    def test_inactive_users_are_skipped(self):
        self.store.put_user(User(uid="u3", lineup=["C1"], active=False))
        self.run_all()
        self.assertIsNone(self.store.get_report("u3", DAY))


class ProduceReportTest(unittest.TestCase):
    """produce_report hands the agent this user's lineup, under this user's uid."""

    def setUp(self):
        for name in ("log", "audit_report"):
            patcher = mock.patch.object(rr, name, return_value=[])
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_remote_session_uses_the_users_uid_and_lineup(self):
        with mock.patch.object(rr, "run_agent_remote", return_value="all clear") as remote:
            fields = rr.produce_report(User(uid="u2", lineup=["B1", "B2"]),
                                       "projects/p/reasoningEngines/1",
                                       local=False, no_fallback=True)
        args, kwargs = remote.call_args
        self.assertEqual(args[1], ["B1", "B2"])
        self.assertEqual(kwargs["user_id"], "u2")
        self.assertEqual(fields["source"], "agent")

    def test_no_lineup_raises_before_the_agent_is_called(self):
        with mock.patch.object(rr, "run_agent_remote") as remote:
            with self.assertRaises(LineupError):
                rr.produce_report(User(uid="u2", lineup=[]), "res",
                                  local=False, no_fallback=True)
        remote.assert_not_called()


class StoreTest(unittest.TestCase):
    def test_a_uid_cannot_reach_another_users_path(self):
        store = MemoryStore()
        for bad in ("", "u1/reports", "../u1"):
            with self.assertRaises(ValueError):
                store.save_report(bad, DAY, {"text": "x"})

    def test_reports_are_keyed_by_user(self):
        store = MemoryStore(users())
        store.save_report("u1", DAY, {"text": "mine"})
        self.assertIsNone(store.get_report("u2", DAY))


if __name__ == "__main__":
    unittest.main()
