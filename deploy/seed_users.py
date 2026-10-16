"""Create the users in Firestore: u1 is the real lineup, u2-u4 are fakes.

Step 1 of the multi-user plan. Four users, all the same person through Gmail
plus-addressing (tred0003+u1@gmail.com ...), so the fan-out, the per-user
sessions and (step 4) the isolation test have more than one user to work on.

    python deploy/seed_users.py --dry-run          # show what would be written
    python deploy/seed_users.py                    # write users that don't exist yet
    python deploy/seed_users.py --force            # overwrite them (loses lineup edits)

Where the lineups come from:
  u1  the real one, copied once from the old lineup Sheet (LINEUP_SHEET_ID).
      The laptop's own login has no Sheets scope, so it reads the sheet as the
      job's service account (--impersonate, needs Token Creator on it), or
      takes the names directly with --u1-lineup "A,B,C,...".
  u2-u4  the optimizer's top three lineups from the latest projections run in
      BigQuery -- the same numbers suggest_lineup.py posts. Three different
      rosters on purpose: if every fake user had the same nine players, a bug
      that sent user A's report to user B would be invisible.

Only u1 posts to Discord. The fakes' reports live in Firestore, for the web
page in step 2.

Existing users are skipped unless --force: after this first run the lineups
are the users' to edit, and a re-seed should not quietly put them back.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from user_store import FirestoreStore, User  # noqa: E402
from watch_slate import load_env  # noqa: E402

EMAIL = "tred0003+{uid}@gmail.com"
REPORTS_SA = "agent-reports@advisor-agent-2026.iam.gserviceaccount.com"


def sheet_lineup(impersonate: str) -> list[str]:
    """u1's lineup from the old Sheet, read as the service account that had access."""
    from google.auth import default, impersonated_credentials
    from lineup_sheet import SCOPES, read_lineup

    source, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds = impersonated_credentials.Credentials(
        source_credentials=source, target_principal=impersonate, target_scopes=SCOPES)
    return read_lineup(os.environ.get("LINEUP_SHEET_ID", "").strip(), credentials=creds)


def optimizer_lineups(n: int) -> list[list[str]]:
    """The top n lineups from the latest projections run, as name lists."""
    from optimize import optimize
    from suggest_lineup import latest_run, playable

    pool = latest_run()
    lineups = optimize(playable(pool).dropna(subset=["projection"]), "projection",
                       n_lineups=n)
    if len(lineups) < n:
        raise RuntimeError(f"optimizer found {len(lineups)} lineups, wanted {n}")
    return [lu["name"].tolist() for lu in lineups]


def main() -> int:
    p = argparse.ArgumentParser(description="Seed the multi-user Firestore users")
    p.add_argument("--dry-run", action="store_true", help="print, write nothing")
    p.add_argument("--force", action="store_true", help="overwrite existing users")
    p.add_argument("--u1-lineup", help="comma-separated names instead of the Sheet")
    p.add_argument("--impersonate", default=REPORTS_SA,
                   help="service account to read the Sheet as")
    a = p.parse_args()
    load_env()  # LINEUP_SHEET_ID from the repo-root .env

    u1 = ([n.strip() for n in a.u1_lineup.split(",") if n.strip()]
          if a.u1_lineup else sheet_lineup(a.impersonate))
    fakes = optimizer_lineups(3)

    users = [User(uid="u1", email=EMAIL.format(uid="u1"), name="Real lineup",
                  lineup=u1, discord=True)]
    for i, lineup in enumerate(fakes, start=2):
        users.append(User(uid=f"u{i}", email=EMAIL.format(uid=f"u{i}"),
                          name=f"Fake user {i} (optimizer lineup {i - 1})",
                          lineup=lineup))

    store = None if a.dry_run else FirestoreStore()
    for u in users:
        print(f"{u.uid}  {u.email}  discord={u.discord}\n    " + ", ".join(u.lineup))
        if store is None:
            continue
        if store.get_user(u.uid) is not None and not a.force:
            print("    exists -- skipped (use --force to overwrite)")
            continue
        store.put_user(u)
        print("    written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
