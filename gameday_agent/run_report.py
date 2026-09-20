"""Headless entry point: run the ADK agent once, send the report to Discord.

This is what Windows Task Scheduler calls on Sunday morning. Nobody is watching
it, nobody types a question, and there is no second turn -- so delivery is NOT
the model's decision. The model writes the report; this script sends it.

    python gameday_agent/run_report.py --dry-run    # print, never send
    python gameday_agent/run_report.py              # send to Discord
    python gameday_agent/run_report.py --local      # run in-process, not deployed

TWO MODES. With AGENT_ENGINE_ID set it calls the agent deployed on Agent
Platform and this file is a thin client -- it asks, receives, delivers. With
--local, or with that variable unset, it runs the agent in this process
instead, which is how an unpushed change gets tested before a four-minute
redeploy.

The split matters beyond convenience. Delivery and the fallback live in the
client so that a deployed agent which is down, out of quota or mid-redeploy
still produces a report: the thing that notices the failure cannot be the
thing that failed.

Why not an ADK `after_agent_callback`, which is the framework-native way to hook
"agent finished"? Because a callback lives on the agent, so it would also fire
in `adk web` -- every experiment in the dev UI would post to the real Discord
channel. Keeping the send out here means the UI stays a safe place to poke at
the agent and only the scheduled job actually talks to Discord.

Two env files are read, on purpose:
    nfl-dfs/.env                  DISCORD_WEBHOOK_URL, AGENT_ENGINE_ID -- the
                                  CLIENT's config, never staged into the image
    nfl-dfs/gameday_agent/.env    MY_LINEUP, WEEK, model + project config, all
                                  of which `adk deploy` bakes into the
                                  deployment (so changing a lineup means a
                                  redeploy until it moves to session state)

ADK loads the second one itself when it launches the agent. Nothing loads it
here, so this script has to -- the same gap that made hand_runner.py report an
unconfigured lineup on its first run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Windows console is cp1252 by default and the report contains emoji and
# typographic dashes. Without this the job dies on print, not on the work.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

def _load_agent_env() -> None:
    """Read gameday_agent/.env into the environment.

    Inline and not shared with hand_runner.py on purpose: this MUST run before
    `gameday_agent.agent` is imported. That module binds
    MODEL = os.environ.get("GEMINI_MODEL", ...) at import time, so importing a
    shared helper out of this package would trigger the package __init__, which
    imports agent, which reads GEMINI_MODEL -- before the loader had a chance to
    set it. Changing the model in .env would then silently do nothing here while
    working fine under `adk web`. Ordering is the point, so it stays visible.
    """
    env = Path(__file__).resolve().parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_agent_env()

from watch_slate import load_env, send_discord  # noqa: E402

from gameday_agent.agent import root_agent  # noqa: E402

QUERY = "Check my lineup for this Sunday and tell me if anything needs my attention."

# A run that hangs on a wedged HTTP call must not sit there until the next one
# starts. Task Scheduler will not kill it for us.
TIMEOUT_SECONDS = 300


def log(msg: str) -> None:
    """Timestamped, flushed -- this is the only record of an unattended run."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run_agent_remote(resource: str) -> str:
    """Call the agent deployed on Agent Platform and return its final report.

    This is the mode the scheduled job runs in. The agent lives in the cloud;
    this script is a thin client that asks it a question, takes the answer and
    delivers it. Delivery and the fallback stay OUT HERE on purpose -- if the
    deployed agent is down, out of quota or mid-redeploy, something that is
    not the agent has to notice and still send a report.

    The event shape differs from the in-process runner. Locally ADK yields
    Event objects with attribute access; stream_query yields plain dicts. Same
    content, different surface, so the two loops below cannot be shared.
    """
    import vertexai
    from vertexai import agent_engines

    vertexai.init(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "advisor-agent-2026"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    agent = agent_engines.get(resource)

    final, calls = "", []
    for event in agent.stream_query(user_id="scheduled", message=QUERY):
        for part in (event.get("content", {}) or {}).get("parts", []) or []:
            if "function_call" in part:
                calls.append(part["function_call"]["name"])
            if part.get("text"):
                final = part["text"]

    log(f"tools called: {', '.join(calls) or 'none'}")
    return final.strip()


async def run_agent() -> str:
    """Drive the ADK agent IN THIS PROCESS and return its final report.

    Used with --local, for testing a code change before deploying it. The
    deployed agent runs whatever was last pushed, so a local run and a remote
    run can legitimately disagree -- which is the point of being able to do
    both.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=root_agent, app_name="gameday_report")
    session = await runner.session_service.create_session(
        app_name="gameday_report", user_id="scheduled"
    )

    final, tool_calls = "", []
    async for event in runner.run_async(
        user_id="scheduled",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=QUERY)]),
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if getattr(part, "function_call", None):
                tool_calls.append(part.function_call.name)
            if getattr(part, "text", None):
                # Last text wins. ADK streams intermediate drafts as the model
                # revises; only the final one is the report.
                final = part.text

    log(f"tools called: {', '.join(tool_calls) or 'none'}")
    return final.strip()


def main() -> int:
    p = argparse.ArgumentParser(description="Run the gameday agent and report")
    p.add_argument("--dry-run", action="store_true",
                   help="print the report instead of sending it")
    p.add_argument("--no-fallback", action="store_true",
                   help="fail loudly instead of sending the rule-based report")
    p.add_argument("--local", action="store_true",
                   help="run the agent in this process instead of calling the "
                        "deployed one (for testing an unpushed change)")
    a = p.parse_args()

    load_env()  # DISCORD_WEBHOOK_URL and AGENT_ENGINE_ID from the repo-root .env

    # The resource id lives in the REPO-ROOT .env, not the agent's own. Anything
    # in gameday_agent/.env is staged into the deployment by `adk deploy`, and
    # an agent carrying its own resource name around is both pointless and one
    # more value to keep in sync.
    resource = os.environ.get("AGENT_ENGINE_ID", "").strip()
    remote = bool(resource) and not a.local

    try:
        if remote:
            log(f"calling deployed agent {resource.rsplit('/', 1)[-1]}")
            report = run_agent_remote(resource)
        else:
            why = "--local" if a.local else "AGENT_ENGINE_ID unset"
            log(f"running agent in-process ({why})")
            report = asyncio.run(asyncio.wait_for(run_agent(), TIMEOUT_SECONDS))
        if not report:
            raise RuntimeError("agent returned an empty report")
        log(f"agent produced {len(report)} characters")
    except Exception as e:
        log(f"agent failed: {type(e).__name__}: {e}")
        if a.no_fallback:
            raise
        # Same call as src/agent.py makes, for the same reason: a scheduled
        # Sunday job that silently sends nothing is worse than one that sends
        # the deterministic version. The rule-based report needs no model and
        # no Agent Platform quota, so it survives most things that kill the LLM.
        log("falling back to the rule-based report")
        from gameday_report import build_report
        report = build_report(int(os.environ.get("WEEK", 1)))

    if a.dry_run:
        log("dry run -- not sending")
        print("\n" + "-" * 60)
        print(report)
        print("-" * 60)
        return 0

    send_discord(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
