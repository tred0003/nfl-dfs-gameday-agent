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


class MemoryStore:
    """Same interface, in a dict. For the unit tests."""

    def __init__(self, users: list[User] | None = None):
        self.users: dict[str, User] = {}
        self.reports: dict[tuple[str, str], dict] = {}
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

    def latest_report(self, uid: str) -> tuple[str, dict] | None:
        ids = sorted(rid for (u, rid) in self.reports if u == _require_uid(uid))
        return (ids[-1], self.reports[(uid, ids[-1])]) if ids else None

    def set_lineup(self, uid: str, lineup: list[str]) -> None:
        self.users[_require_uid(uid)].lineup = list(lineup)
