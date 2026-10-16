"""Create the Firebase Auth accounts for the Firestore users, with matching uids.

Multi-user step 2. The web app takes the uid from a verified Firebase token
and reads users/{uid}, so the Auth account's uid MUST equal the Firestore
document id. Firebase would normally pick a random uid; this script creates
each account with the uid chosen in step 1 (u1..u4) instead.

    python deploy/create_auth_users.py --dry-run
    python deploy/create_auth_users.py

No passwords are set here, on purpose -- nobody but the account owner should
ever know one. Each user clicks "Email me a link to set my password" on the
sign-in page; the mail goes to tred0003+uN@gmail.com, which is the same inbox.

Plain REST against the Identity Toolkit API, as your own gcloud login (ADC),
rather than the Admin SDK: the SDK wants a service-account key or a quota
project to manage users, and one POST per user does not justify either.
Existing accounts are left alone.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from user_store import FirestoreStore  # noqa: E402

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "advisor-agent-2026")
API = f"https://identitytoolkit.googleapis.com/v1/projects/{PROJECT}"


def session():
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    # User credentials bill API calls to their ADC quota project -- on this
    # laptop that is fantasy-football-ml, where Identity Toolkit is off (403).
    # A plain x-goog-user-project header loses to the credential's own, so
    # set it on the credential.
    return AuthorizedSession(creds.with_quota_project(PROJECT))


def main() -> int:
    p = argparse.ArgumentParser(description="Create Firebase Auth users for users/{uid}")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    users = FirestoreStore().active_users()
    http = None if a.dry_run else session()
    for u in users:
        print(f"{u.uid}  {u.email}", end="  ")
        if http is None:
            print("(dry run)")
            continue
        r = http.post(f"{API}/accounts:lookup", json={"localId": [u.uid]}, timeout=20)
        r.raise_for_status()
        if r.json().get("users"):
            print("exists -- skipped")
            continue
        r = http.post(f"{API}/accounts", json={"localId": u.uid, "email": u.email,
                                               "displayName": u.name}, timeout=20)
        if r.status_code != 200:
            print(f"FAILED HTTP {r.status_code}: {r.text[:200]}")
            return 1
        print("created")
    return 0


if __name__ == "__main__":
    sys.exit(main())
