"""Per-user lineups and reports in Firestore -- what replaced the single Sheet.

Why: the gameday report served one person from one Google Sheet. Step 1 of
the multi-user plan moves the lineup and the finished report into Firestore,
keyed by user, so the 12:30 job can loop over users and (step 2) a small web
app can let each one edit their own lineup and read their own report.

Layout:

    users/{uid}                        one document per user
        email      str                 who they are (fake users: tred0003+uN@gmail.com)
        name       str                 display name, for the web page and logs
        lineup     list[str]           player names, same shape the Sheet gave
        active     bool                the scheduled job skips inactive users
        discord    bool                also post this user's report to Discord
        updated_at timestamp
    users/{uid}/reports/{report_id}    one document per report
        text, source ("agent" | "fallback"), problems, tools, created_at

Reports are a SUBCOLLECTION of the user, not a top-level collection with a
uid field. The path itself says whose data it is, which is what makes the
step-2 security rule one line (request.auth.uid == uid) and the isolation
test easy to state: user2's code has no path into users/user1.

ISOLATION, AT THIS LAYER. Every read and write below takes a uid and builds
the path from it. The only call that crosses users is active_users(), and
only the scheduled job calls it -- the web app (step 2) never will. Where the
uid comes FROM is the web app's problem: a verified sign-in token, never a
value the browser sends.

Two stores with one interface: FirestoreStore for real, MemoryStore for the
unit tests, so CI tests the fan-out loop with no credentials and no network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

USERS = "users"
REPORTS = "reports"
# users/{uid}/asks/{ask_id}: every News desk answer -- question, answer,
# the passage ids it cited, versions, cost -- and the user's feedback on it.
# The audit trail ("who asked what, and what did they see") and the seed of
# a live eval set. A subcollection, like reports: one user cannot name
# another's.
ASKS = "asks"
# users/{uid}/limits/{YYYY-MM-DD}: {route: count}. Per-user daily limits that
# survive a restart and hold across instances (an in-memory counter did
# neither: each instance had its own, and a deploy reset it).
LIMITS = "limits"
# One document per NFL week, written by gameday_agent/weekly_review.py: the
# week's judge scores, check pass rate, fallback rate, cost and the warning
# outcomes. Top level, not under a user -- it spans every user -- and only the
# weekly job's service account reads or writes it.
WEEKLY = "eval_weekly"


class LineupError(RuntimeError):
    """This user has no usable lineup. Same meaning the Sheet's error had."""


@dataclass
class User:
    uid: str
    email: str = ""
    name: str = ""
    lineup: list[str] = field(default_factory=list)
    active: bool = True
    discord: bool = False

    def checked_lineup(self) -> list[str]:
        """The lineup, or LineupError. Never an empty list that reads as all clear.

        Same rule the Sheet had: no players means no report for this user, not
        a report about nobody. An empty lineup checked by the agent comes back
        "nothing to worry about", which is the confidently-wrong failure.
        """
        names = [str(n).strip() for n in (self.lineup or []) if str(n).strip()]
        if not names:
            raise LineupError(f"user {self.uid} has no players in their lineup")
        return names

    def to_doc(self) -> dict:
        return {"email": self.email, "name": self.name, "lineup": list(self.lineup),
                "active": self.active, "discord": self.discord}

    @classmethod
    def from_doc(cls, uid: str, doc: dict) -> "User":
        return cls(uid=uid, email=doc.get("email", ""), name=doc.get("name", ""),
                   lineup=list(doc.get("lineup") or []),
                   active=bool(doc.get("active", True)),
                   discord=bool(doc.get("discord", False)))


def _require_uid(uid: str) -> str:
    # An empty uid would address the collection itself; a slash would let one
    # uid name another user's subcollection. Neither is a user.
    if not uid or "/" in uid:
        raise ValueError(f"bad uid {uid!r}")
    return uid


class FirestoreStore:
    """The real store. Auth is Application Default Credentials, like the Sheet was:
    the job's service account in Cloud Run, your own login on the laptop."""

    def __init__(self, project: str | None = None, database: str = "(default)"):
        from google.cloud import firestore  # lazy: tests and --help need no client

        self._fs = firestore
        self._db = firestore.Client(
            project=project or os.environ.get("GOOGLE_CLOUD_PROJECT", "advisor-agent-2026"),
            database=database)

    def _user(self, uid: str):
        return self._db.collection(USERS).document(_require_uid(uid))

    def active_users(self) -> list[User]:
        # Scheduled job only. Ordered by uid so the run order -- and the log --
        # is the same every day.
        snaps = self._db.collection(USERS).where("active", "==", True).stream()
        return sorted((User.from_doc(s.id, s.to_dict() or {}) for s in snaps),
                      key=lambda u: u.uid)

    def get_user(self, uid: str) -> User | None:
        snap = self._user(uid).get()
        return User.from_doc(snap.id, snap.to_dict() or {}) if snap.exists else None

    def put_user(self, user: User) -> None:
        self._user(user.uid).set({**user.to_doc(),
                                  "updated_at": self._fs.SERVER_TIMESTAMP})

    def save_report(self, uid: str, report_id: str, fields: dict) -> None:
        self._user(uid).collection(REPORTS).document(report_id).set(
            {**fields, "created_at": self._fs.SERVER_TIMESTAMP})

    def get_report(self, uid: str, report_id: str) -> dict | None:
        snap = self._user(uid).collection(REPORTS).document(report_id).get()
        return snap.to_dict() if snap.exists else None

    def save_ask(self, uid: str, ask_id: str, fields: dict) -> None:
        self._user(uid).collection(ASKS).document(_require_uid(ask_id)).set(
            {**fields, "created_at": self._fs.SERVER_TIMESTAMP})

    def get_ask(self, uid: str, ask_id: str) -> dict | None:
        snap = self._user(uid).collection(ASKS).document(_require_uid(ask_id)).get()
        return snap.to_dict() if snap.exists else None

    def set_ask_feedback(self, uid: str, ask_id: str, feedback: dict) -> None:
        self._user(uid).collection(ASKS).document(_require_uid(ask_id)).update(
            {"feedback": {**feedback, "at": self._fs.SERVER_TIMESTAMP}})

    def spend(self, uid: str, route: str, day: str, limit: int) -> bool:
        """Count one call against today's limit; False (and no count) once spent.
        A transaction, so two instances cannot both take the last call."""
        ref = self._user(uid).collection(LIMITS).document(day)

        @self._fs.transactional
        def take(tx) -> bool:
            snap = ref.get(transaction=tx)
            n = int((snap.to_dict() or {}).get(route, 0)) if snap.exists else 0
            if n >= limit:
                return False
            tx.set(ref, {route: n + 1}, merge=True)
            return True

        return take(self._db.transaction())

    def latest_report(self, uid: str) -> tuple[str, dict] | None:
        # Report ids are YYYY-MM-DD, so the highest id is the newest report.
        # Not order_by("__name__", DESCENDING): a descending key query needs a
        # composite index Firestore does not create on its own, and without
        # one every /api/me was a 500 (2026-10-19). Listing ids needs no index
        # and is one report per day per user -- small for a long time.
        ids = [d.id for d in self._user(uid).collection(REPORTS).list_documents()]
        if not ids:
            return None
        snap = self._user(uid).collection(REPORTS).document(max(ids)).get()
        return (snap.id, snap.to_dict() or {}) if snap.exists else None

    def set_lineup(self, uid: str, lineup: list[str]) -> None:
        # update(), not set(): the web page may change the lineup and nothing
        # else. A set() here would also wipe name, active and discord.
        self._user(uid).update({"lineup": list(lineup),
                                "updated_at": self._fs.SERVER_TIMESTAMP})

    # ---- weekly review only. Like active_users(), these cross users or sit
    # outside them; the web app never calls them.
    def all_users(self) -> list[User]:
        # Inactive users too: a user switched off mid-week still had reports
        # earlier in it, and those count.
        return sorted((User.from_doc(s.id, s.to_dict() or {})
                       for s in self._db.collection(USERS).stream()),
                      key=lambda u: u.uid)

    def report_ids(self, uid: str) -> list[str]:
        # Ids only, no index needed -- same reason as latest_report().
        return sorted(d.id for d in self._user(uid).collection(REPORTS).list_documents())

    def save_weekly(self, week_id: str, fields: dict) -> None:
        self._db.collection(WEEKLY).document(week_id).set(
            {**fields, "created_at": self._fs.SERVER_TIMESTAMP})

    def weekly_history(self, before: str, n: int) -> list[tuple[str, dict]]:
        """Up to n weekly docs with ids sorting before `before`, oldest first."""
        ids = sorted(d.id for d in self._db.collection(WEEKLY).list_documents()
                     if d.id < before)[-n:] if n > 0 else []
        out = []
        for i in ids:
            snap = self._db.collection(WEEKLY).document(i).get()
            if snap.exists:
                out.append((i, snap.to_dict() or {}))
        return out


class MemoryStore:
    """Same interface, in a dict. For the unit tests."""

    def __init__(self, users: list[User] | None = None):
        self.users: dict[str, User] = {}
        self.reports: dict[tuple[str, str], dict] = {}
        self.asks: dict[tuple[str, str], dict] = {}
        self.limits: dict[tuple[str, str, str], int] = {}
        self.weekly: dict[str, dict] = {}
        for u in users or []:
            self.put_user(u)

    def active_users(self) -> list[User]:
        return sorted((u for u in self.users.values() if u.active), key=lambda u: u.uid)

    def get_user(self, uid: str) -> User | None:
        return self.users.get(_require_uid(uid))

    def put_user(self, user: User) -> None:
        self.users[_require_uid(user.uid)] = user

    def save_report(self, uid: str, report_id: str, fields: dict) -> None:
        self.reports[(_require_uid(uid), report_id)] = {
            **fields, "created_at": datetime.now(timezone.utc)}

    def get_report(self, uid: str, report_id: str) -> dict | None:
        return self.reports.get((_require_uid(uid), report_id))

    def spend(self, uid: str, route: str, day: str, limit: int) -> bool:
        key = (_require_uid(uid), day, route)
        if self.limits.get(key, 0) >= limit:
            return False
        self.limits[key] = self.limits.get(key, 0) + 1
        return True

    def save_ask(self, uid: str, ask_id: str, fields: dict) -> None:
        self.asks[(_require_uid(uid), _require_uid(ask_id))] = {
            **fields, "created_at": datetime.now(timezone.utc)}

    def get_ask(self, uid: str, ask_id: str) -> dict | None:
        return self.asks.get((_require_uid(uid), _require_uid(ask_id)))

    def set_ask_feedback(self, uid: str, ask_id: str, feedback: dict) -> None:
        self.asks[(_require_uid(uid), _require_uid(ask_id))]["feedback"] = {
            **feedback, "at": datetime.now(timezone.utc)}

    def latest_report(self, uid: str) -> tuple[str, dict] | None:
        ids = sorted(rid for (u, rid) in self.reports if u == _require_uid(uid))
        return (ids[-1], self.reports[(uid, ids[-1])]) if ids else None

    def set_lineup(self, uid: str, lineup: list[str]) -> None:
        self.users[_require_uid(uid)].lineup = list(lineup)

    def all_users(self) -> list[User]:
        return sorted(self.users.values(), key=lambda u: u.uid)

    def report_ids(self, uid: str) -> list[str]:
        return sorted(rid for (u, rid) in self.reports if u == _require_uid(uid))

    def save_weekly(self, week_id: str, fields: dict) -> None:
        self.weekly[week_id] = {**fields, "created_at": datetime.now(timezone.utc)}

    def weekly_history(self, before: str, n: int) -> list[tuple[str, dict]]:
        ids = sorted(i for i in self.weekly if i < before)[-n:] if n > 0 else []
        return [(i, self.weekly[i]) for i in ids]
