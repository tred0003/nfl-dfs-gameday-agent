"""Headless entry point: run the dynasty strategy agent once, post to Discord.

    python sleeper_agent/run_report.py --dry-run
    python sleeper_agent/run_report.py

Unlike gameday_agent, this is NOT a weekly job. A dynasty buy/sell read does not
change meaningfully between Tuesday and Thursday, and a strategy memo that
arrives every week becomes noise that gets ignored -- which is worse than not
sending it. Every two or three weeks, and around the trade deadline, is the
cadence this is written for.

Delivery lives here rather than in an ADK `after_agent_callback` for the same
reason as gameday_agent: a callback lives on the agent, so it would also fire
under `adk web` and post every dev-UI experiment to the real channel.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _load_agent_env() -> None:
    """Read sleeper_agent/.env before anything imports the agent package.

    Ordering matters and is why this is inline rather than shared: agent.py
    binds MODEL = os.environ.get("GEMINI_MODEL", ...) at import time, and
    importing a helper from inside the package would trigger __init__, which
    imports agent, before the loader ever ran.
    """
    env = Path(__file__).resolve().parent / ".env"
    if not env.exists():
        return
    import os
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_agent_env()

from watch_slate import load_env, send_discord  # noqa: E402

from sleeper_agent.agent import root_agent  # noqa: E402

QUERY = (
    "Give me my dynasty strategy read for right now. Am I a buyer or a seller, "
    "and who specifically should I be talking to?"
)

# Longer than gameday_agent's 300s: this agent pulls a 14.7 MB player dump on a
# cold cache and may call five tools across several turns.
TIMEOUT_SECONDS = 600


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


async def run_agent(query: str) -> str:
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=root_agent, app_name="sleeper_report")
    session = await runner.session_service.create_session(
        app_name="sleeper_report", user_id="scheduled"
    )

    final, calls = "", []
    async for event in runner.run_async(
        user_id="scheduled",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if getattr(part, "function_call", None):
                calls.append(part.function_call.name)
            if getattr(part, "text", None):
                final = part.text  # last text wins -- ADK streams drafts

    log(f"tools called: {', '.join(calls) or 'none'}")
    return final.strip()


def main() -> int:
    p = argparse.ArgumentParser(description="Dynasty strategy read -> Discord")
    p.add_argument("--dry-run", action="store_true",
                   help="print the memo instead of sending it")
    p.add_argument("--ask", help="ask something other than the default read")
    a = p.parse_args()

    load_env()  # DISCORD_WEBHOOK_URL from the repo-root .env
    log("starting sleeper strategy agent")

    try:
        memo = asyncio.run(asyncio.wait_for(run_agent(a.ask or QUERY), TIMEOUT_SECONDS))
        if not memo:
            raise RuntimeError("agent returned an empty memo")
        log(f"agent produced {len(memo)} characters")
    except Exception as e:
        # No rule-based fallback here, unlike gameday_agent. That one falls back
        # because a missed OUT player costs real points on a Sunday. A strategy
        # memo has no deadline -- sending a degraded one would be worse than
        # sending nothing and retrying.
        log(f"agent failed: {type(e).__name__}: {e}")
        raise

    if a.dry_run:
        log("dry run -- not sending")
        print("\n" + "-" * 60)
        print(memo)
        print("-" * 60)
        return 0

    send_discord(memo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
