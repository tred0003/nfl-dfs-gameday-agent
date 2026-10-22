"""Thursday canary: prove every tool works through the real production path.

The gameday-canary Cloud Run job runs this every Thursday at 10:00 ET, three
days before the Sunday report matters. It goes through the same path the
daily report does -- Cloud Run -> Firestore lineup -> deployed agent on Agent
Engine -> Gemini -> every tool -> Discord -- so a pass means the Sunday run
has a working path, not just that the code is fine on a laptop.

    python gameday_agent/canary.py              # run it, post the verdict
    python gameday_agent/canary.py --dry-run    # print the verdict, don't post

WHY NOT JUST THE DAILY REPORT. The daily report cannot be the health check:
  - It FALLS BACK to the rule-based report when the agent fails, which is the
    right call for delivery and exactly wrong for detection -- a broken agent
    still produces a message and exits 0.
  - Tool errors come back as DATA ({"status": "error"}) so the model can say
    "weather could not be checked" and carry on. Also right for the report,
    also invisible to anything that only checks the exit code.
  - It only calls find_replacements when a player is OUT, so that tool can be
    broken for weeks without a single report noticing.

WHAT PASSES. Every tool in EXPECTED was called, every one returned
status "ok", and the ones that should always carry rows on a live slate are
not empty. Empty-but-ok is the quiet failure: a source that answers 200 with
nothing in it reads as "no issues" downstream.

WHAT IT DOES NOT CHECK. Whether the data is RIGHT. A tool returning stale or
wrong rows with status "ok" passes. That is the eval harness's and the LLM
judge's job, not a canary's -- the canary answers "is the plumbing alive".

A failure exits 1. The project's "Scheduled agent report FAILED" alert fires
on any failed Cloud Run job execution, so a red canary emails without any
alert of its own.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Importing run_report loads gameday_agent/.env and puts src/ on the path, in
# the order its docstring explains. It does not import the agent package.
from run_report import log, run_agent_remote  # noqa: E402
from user_store import FirestoreStore, LineupError  # noqa: E402
from watch_slate import load_env, send_discord  # noqa: E402

# Every tool in gameday_agent/tools.py TOOLS, and the response key that must be
# non-empty for the call to count as working. None = an empty result is
# legitimate (no notable scratches is a normal Thursday).
#
# Kept here, not imported from tools.py: that would pull the agent package and
# its data dependencies into an image that deliberately ships without them.
# Add a tool to TOOLS without adding it here and the canary will not test it.
EXPECTED = {
    "get_lineup_status": "lineup",
    "get_cb_matchups": "matchups",
    "get_slate_weather": "games",
    "get_vegas_context": "teams",
    "find_replacements": "options",
    "get_notable_scratches": None,
    "get_value_picks": "picks",
    "get_team_news": None,      # zero claims on a quiet week is legitimate
    "search_news": "sources",   # grounded search must come back with sources
    "get_player_history": "players",
    "search_news_archive": "passages",
}

# The archive is appended by a daily job (news_rag/refresh.py). If that job
# stops, the history tools keep answering "ok" from a history that ends days
# ago -- the quiet failure this canary exists for. Same limit the tool uses
# for its own freshness_warning (archive.STALE_HOURS; copied, not imported,
# for the reason EXPECTED is).
ARCHIVE_STALE_HOURS = 36

# Statuses that count as working, beyond "ok". The projection job runs Friday
# evening and Sunday 11:00, so on a Thursday this slate has no projections yet
# and the tool says "not_ready" -- correct, and still proof the BigQuery read
# and the cross-project permission work. "error" still fails.
ALSO_OK = {"get_value_picks": {"not_ready"}}

# Explicit, because the normal question lets the model choose its tools and the
# canary's whole point is that it must not. find_replacements gets fixed
# arguments a live slate always satisfies (plenty of WRs under $5,000).
CANARY_QUERY = (
    "This is a scheduled health check, not a real report. Call each of these "
    "tools exactly once: get_lineup_status, get_cb_matchups, get_slate_weather, "
    "get_vegas_context, get_notable_scratches, get_value_picks, get_team_news, "
    "get_player_history, "
    "search_news with query 'NFL injury news this week', "
    "search_news_archive with question 'injury update' and no player, and "
    "find_replacements with "
    "position WR and max_salary 5000. Then reply with one short line per tool "
    "saying whether it returned data."
)


def check(responses: list[dict]) -> list[tuple[str, bool, str]]:
    """One (tool, passed, detail) row per expected tool."""
    by_name: dict[str, dict] = {}
    for r in responses:
        # Keep the first response per tool; the prompt asks for one call each.
        by_name.setdefault(r.get("name"), r.get("response") or {})

    rows = []
    for tool, key in EXPECTED.items():
        if tool not in by_name:
            rows.append((tool, False, "never called"))
            continue
        resp = by_name[tool]
        status = resp.get("status")
        if status in ALSO_OK.get(tool, set()):
            rows.append((tool, True, f"{status} (expected before the Friday run)"))
            continue
        if status != "ok":
            msg = str(resp.get("message") or resp)[:150]
            rows.append((tool, False, f"status={status!r}: {msg}"))
            continue
        if tool == "get_player_history" and resp.get("players"):
            hours = resp.get("archive_hours_old")
            if hours is None or hours > ARCHIVE_STALE_HOURS:
                rows.append((tool, False, f"news archive is stale (newest item {hours} h old) "
                                          "-- check the news-archive job"))
                continue
        if key is not None:
            n = len(resp.get(key) or [])
            if n == 0:
                rows.append((tool, False, f"status ok but '{key}' is empty"))
                continue
            rows.append((tool, True, f"{n} {key}"))
        else:
            rows.append((tool, True, "ok"))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Thursday tool canary")
    p.add_argument("--dry-run", action="store_true",
                   help="print the verdict instead of posting it")
    a = p.parse_args()

    load_env()
    resource = os.environ.get("AGENT_ENGINE_ID", "").strip()
    if not resource:
        log("CANARY FAILED: AGENT_ENGINE_ID is not set")
        return 1

    t0 = time.monotonic()
    failure = None
    rows: list[tuple[str, bool, str]] = []
    try:
        # One real user's lineup, through the same store the report reads.
        # CANARY_UID picks whose; u1 is the real lineup, u2-u4 are fakes.
        uid = os.environ.get("CANARY_UID", "u1").strip()
        user = FirestoreStore().get_user(uid)
        if user is None:
            raise LineupError(f"user {uid} not found in Firestore")
        lineup = user.checked_lineup()
        log(f"lineup from Firestore users/{uid}: {len(lineup)} players")
        responses: list[dict] = []
        run_agent_remote(resource, lineup, query=CANARY_QUERY, responses=responses,
                         user_id=f"canary-{uid}")
        rows = check(responses)
    except LineupError as e:
        failure = f"lineup unavailable: {e}"
    except Exception as e:  # noqa: BLE001 -- any agent failure is a red canary
        failure = f"agent run failed: {type(e).__name__}: {str(e)[:200]}"
    secs = time.monotonic() - t0

    passed = failure is None and bool(rows) and all(ok for _, ok, _ in rows)
    n_ok = sum(ok for _, ok, _ in rows)
    head = (f"✅ **Gameday canary passed**: {n_ok}/{len(EXPECTED)} tools OK"
            if passed else
            f"❌ **Gameday canary FAILED**: {n_ok}/{len(EXPECTED)} tools OK")
    lines = [f"{head} ({secs:.0f}s, deployed agent {resource.rsplit('/', 1)[-1]})"]
    if failure:
        lines.append(f"• {failure}")
    lines += [f"{'✓' if ok else '✗'} `{tool}`: {detail}" for tool, ok, detail in rows]
    verdict = "\n".join(lines)
    log(verdict)

    if not a.dry_run and not send_discord(verdict):
        log("NOT DELIVERED: no DISCORD_WEBHOOK_URL")
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
