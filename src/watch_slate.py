"""Gameday watcher: poll the slate, detect changes that matter, push to Discord.

Runs headless. No DraftKings account -- player status (OUT / Q / IR) is in the
same public draftables feed the salaries come from, and it updates right up to
lock. Your lineup is nine names recorded once via track.py.

Designed to run either from a local scheduler or as a Cloud Run job on a Cloud
Scheduler cron. It holds no local state beyond the snapshot files, so the cloud
version needs only a bucket.

The hard part is NOT fetching -- it is deciding what deserves an interruption.
Someone relaxing with family on a Sunday should hear about a rostered player
being ruled out. They should not hear that a $3,000 WR they do not own moved
from Questionable to Out. Every rule in should_alert() exists to protect that.

  python src/watch_slate.py --week 1 --dry-run
  python src/watch_slate.py --week 1            # sends if DISCORD_WEBHOOK_URL set
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# Windows consoles default to cp1252 and blow up on the status emoji. Discord
# itself takes UTF-8 fine -- this only protects the local --dry-run path.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dk_client import fetch_draftables, list_slates, pick_main_slate  # noqa: E402
from predict_slate import normalize_name  # noqa: E402

LINEUPS = ROOT / "data" / "lineups"
SLATES = ROOT / "data" / "slates"

# Status changes worth waking someone up for.
# "D" (Doubtful) belongs here: it means unlikely to play, so it is effectively a
# scratch, not a maybe. Leaving it out silently swallowed a $6,900 TE going
# Doubtful during the first live test.
BAD_STATUSES = {"OUT", "IR", "D"}
WATCH_STATUSES = {"Q", "D", "OUT", "IR"}


def load_env() -> None:
    """Minimal .env reader -- the webhook URL must never be committed."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def my_lineup(season: int, week: int, source: str = "mine") -> pd.DataFrame | None:
    path = LINEUPS / f"{season}_wk{week:02d}_{source}.parquet"
    if not path.exists():
        return None
    lu = pd.read_parquet(path)
    lu["join_name"] = normalize_name(lu["name"])
    return lu


def previous_snapshot(dg: int) -> pd.DataFrame | None:
    files = sorted((SLATES / f"dg{dg}").glob("players_*.parquet"))
    return pd.read_parquet(files[-1]) if files else None


def diff_status(prev: pd.DataFrame, now: pd.DataFrame) -> pd.DataFrame:
    """Players whose availability changed since the last poll."""
    p = prev[["player_dk_id", "name", "position", "team", "status", "salary"]]
    n = now[["player_dk_id", "name", "position", "team", "status", "salary"]]
    m = p.merge(n, on="player_dk_id", suffixes=("_before", "_after"))
    changed = m[m["status_before"].ne(m["status_after"])].copy()
    changed["name"] = changed["name_after"]
    return changed


def should_alert(changed: pd.DataFrame, lineup: pd.DataFrame | None) -> pd.DataFrame:
    """Filter to changes that actually warrant interrupting someone.

    Rule 1: anything happening to a player you rostered.
    Rule 2: anyone becoming OUT/IR who was expensive enough to have been a
            realistic play -- that reshapes the slate even if you did not own him.
    Everything else is noise.
    """
    if changed.empty:
        return changed

    keep = pd.Series(False, index=changed.index)

    if lineup is not None:
        mine = set(lineup["join_name"])
        is_mine = normalize_name(changed["name"]).isin(mine)
        keep |= is_mine
        changed["mine"] = is_mine
    else:
        changed["mine"] = False

    became_bad = changed["status_after"].isin(BAD_STATUSES)
    keep |= became_bad & changed["salary_after"].ge(5500)
    return changed[keep]


def format_message(alerts: pd.DataFrame, week: int, minutes_to_lock: int | None) -> str:
    if alerts.empty:
        return ""
    mine = alerts[alerts["mine"]]
    other = alerts[~alerts["mine"]]

    lines = [f"**NFL DFS — Week {week} slate update**"]
    if minutes_to_lock is not None:
        lines.append(f"_{minutes_to_lock} min to lock_")

    if not mine.empty:
        lines.append("\n**IN YOUR LINEUP:**")
        for _, r in mine.iterrows():
            flag = "🔴" if r["status_after"] in BAD_STATUSES else "🟡"
            lines.append(
                f"{flag} **{r['name']}** ({r['position_after']}, {r['team_after']}) "
                f"— {r['status_before'] or 'None'} → **{r['status_after']}**"
            )
        if (mine["status_after"].isin(BAD_STATUSES)).any():
            lines.append("\n⚠️ You need a swap. Reply and I'll re-optimize.")

    if not other.empty:
        lines.append("\n_Elsewhere on the slate:_")
        for _, r in other.head(6).iterrows():
            lines.append(
                f"• {r['name']} ({r['position_after']}, {r['team_after']}) "
                f"${r['salary_after']:,} — {r['status_after']}"
            )
    return "\n".join(lines)


# Discord rejects a message body over 2000 characters. 1900 leaves room for the
# part marker appended below.
DISCORD_LIMIT = 1900


def _chunks(message: str) -> list[str]:
    """Split a long report on line boundaries, never mid-sentence.

    This used to be `message[:1900]` -- a silent truncation. That was harmless
    for the rule-based report, which is short and bounded, but the LLM report
    runs long exactly when it matters most: a scratch plus replacement options
    plus weather. The replacements live at the END of that report, so the one
    message worth reading was the one getting its tail cut off, with nothing
    printed to say so.
    """
    out, buf = [], ""
    for line in message.splitlines(keepends=True):
        # A single line longer than the limit still has to be hard-split.
        while len(line) > DISCORD_LIMIT:
            if buf:
                out.append(buf)
                buf = ""
            out.append(line[:DISCORD_LIMIT])
            line = line[DISCORD_LIMIT:]
        if len(buf) + len(line) > DISCORD_LIMIT:
            out.append(buf)
            buf = ""
        buf += line
    if buf.strip():
        out.append(buf)
    return out or [message]


def send_discord(message: str) -> bool:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        print("no DISCORD_WEBHOOK_URL set -- printing instead:\n")
        print(message)
        return False

    parts = _chunks(message)
    for i, part in enumerate(parts, 1):
        # Only marked when there IS more than one, so the common single-message
        # case reads clean.
        suffix = f"\n\n_({i}/{len(parts)})_" if len(parts) > 1 else ""
        r = requests.post(url, json={"content": part + suffix}, timeout=20)
        r.raise_for_status()
    print(f"sent to Discord ({len(parts)} message{'s' if len(parts) > 1 else ''})")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--dg", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="never send, just print")
    a = p.parse_args()

    load_env()
    dg = a.dg or pick_main_slate(list_slates())

    prev = previous_snapshot(dg)
    now, games = fetch_draftables(dg)

    # persist this poll so the next one has something to diff against
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = SLATES / f"dg{dg}"
    out.mkdir(parents=True, exist_ok=True)
    now.to_parquet(out / f"players_{ts}.parquet", index=False)

    if prev is None:
        print(f"first poll for dg{dg} -- baseline saved, nothing to compare")
        return

    changed = diff_status(prev, now)
    lineup = my_lineup(a.season, a.week)
    if lineup is None:
        print("no lineup recorded -- alerting on slate-wide changes only")

    alerts = should_alert(changed, lineup)
    print(f"{len(changed)} status changes, {len(alerts)} worth alerting on")

    if alerts.empty:
        return

    lock = pd.to_datetime(now["start_time_utc"], errors="coerce", utc=True).min()
    mins = int((lock - pd.Timestamp.now(tz="UTC")).total_seconds() // 60) if pd.notna(lock) else None

    msg = format_message(alerts, a.week, mins)
    if a.dry_run:
        print("\n--- DRY RUN ---\n" + msg)
    else:
        send_discord(msg)


if __name__ == "__main__":
    main()
