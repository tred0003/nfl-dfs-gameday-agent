"""Headless entry point: run the ADK agent once, send the report to Discord.

This is what the gameday-report Cloud Run job runs every day. Nobody is watching
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

THE LINEUP COMES WITH THE REQUEST. This client reads it from a Google Sheet
(LINEUP_SHEET_ID) and creates the agent's session with it in state, so the
agent checks whatever the sheet says right now. It used to be MY_LINEUP baked
into the deployment, which went three weeks stale. The week is derived from
the slate date (src/nfl_week.py), for the same reason.

Two env files are read, on purpose:
    nfl-dfs/.env                  DISCORD_WEBHOOK_URL, AGENT_ENGINE_ID,
                                  LINEUP_SHEET_ID -- the CLIENT's config,
                                  never staged into the agent
    nfl-dfs/gameday_agent/.env    model + project config, which `adk deploy`
                                  bakes into the deployment. MY_LINEUP and
                                  WEEK must NOT be in it when deploying; see
                                  deploy/REPORTS-CLOUD.md

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

from lineup_sheet import EXPECTED_SLOTS, LineupError, read_lineup  # noqa: E402
from watch_slate import load_env, send_discord  # noqa: E402

# Must match gameday_agent/slate.py LINEUP_STATE_KEY. Not imported from there:
# that would import the agent package, which the scheduled image does not ship.
LINEUP_STATE_KEY = "my_lineup"

QUERY = "Check my lineup for this Sunday and tell me if anything needs my attention."

# A run that hangs on a wedged HTTP call must not sit there until the next one
# starts. The Cloud Run task timeout is the backstop, not the plan.
TIMEOUT_SECONDS = 300


def log(msg: str) -> None:
    """Timestamped, flushed -- this is the only record of an unattended run."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run_agent_remote(resource: str, lineup: list[str]) -> str:
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

    # A fresh session per run, carrying this run's lineup in state. The
    # agent's lineup tools read state before they look at env, so this is
    # what makes a sheet edit take effect with no redeploy.
    session = agent.create_session(user_id="scheduled",
                                   state={LINEUP_STATE_KEY: lineup})
    session_id = session["id"] if isinstance(session, dict) else session.id

    texts, calls = [], []
    try:
        for event in agent.stream_query(user_id="scheduled", session_id=session_id,
                                        message=QUERY):
            # A model error arrives as an EVENT, not an exception: a dict with
            # error_code and no content. Seen on 10/9 -- a 429 from Gemini
            # landed mid-run, the loop skipped the contentless event, and the
            # half-written report would have gone out as if it were finished.
            # Raise, so the caller treats it as an agent failure (fallback/alert).
            if event.get("error_code"):
                raise RuntimeError(f"agent error {event['error_code']}: "
                                   f"{str(event.get('error_message', ''))[:200]}")
            if event.get("partial"):
                continue
            for part in (event.get("content", {}) or {}).get("parts", []) or []:
                if "function_call" in part:
                    calls.append(part["function_call"]["name"])
                if part.get("text"):
                    texts.append(part["text"])
    finally:
        # Managed sessions persist and bill per event. One run, one session,
        # then gone. A failed delete is not worth failing the report over.
        try:
            agent.delete_session(user_id="scheduled", session_id=session_id)
        except Exception as e:  # noqa: BLE001
            log(f"could not delete session {session_id}: {e}")

    log(f"tools called: {', '.join(calls) or 'none'}")
    # Every text part, in order -- not just the last one. The model writes the
    # report in sections BETWEEN tool calls (lineup, then weather, then
    # matchups); keeping only the last text produced a 397-character
    # Vegas-only "report" on 10/9 after the agent had called all five tools.
    # Streaming drafts are excluded by skipping partial events above, which is
    # what the old "last text wins" rule was guarding against.
    return "\n\n".join(t.strip() for t in texts if t.strip())


async def run_agent(lineup: list[str]) -> str:
    """Drive the ADK agent IN THIS PROCESS and return its final report.

    Used with --local, for testing a code change before deploying it. The
    deployed agent runs whatever was last pushed, so a local run and a remote
    run can legitimately disagree -- which is the point of being able to do
    both.
    """
    # Imported here, not at the top: the scheduled job only CALLS the deployed
    # agent, and a top-level import dragged ADK, the tools and every data
    # dependency into a client that needs none of them. The cloud image
    # stays small, and only --local pays for the import.
    from gameday_agent.agent import root_agent
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=root_agent, app_name="gameday_report")
    session = await runner.session_service.create_session(
        app_name="gameday_report", user_id="scheduled",
        state={LINEUP_STATE_KEY: lineup},
    )

    texts, tool_calls = [], []
    async for event in runner.run_async(
        user_id="scheduled",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=QUERY)]),
    ):
        if getattr(event, "error_code", None):
            raise RuntimeError(f"agent error {event.error_code}: "
                               f"{str(getattr(event, 'error_message', ''))[:200]}")
        if getattr(event, "partial", False):
            continue
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if getattr(part, "function_call", None):
                tool_calls.append(part.function_call.name)
            if getattr(part, "text", None):
                texts.append(part.text)

    log(f"tools called: {', '.join(tool_calls) or 'none'}")
    # Every text part, in order -- not just the last one. The model writes the
    # report in sections BETWEEN tool calls (lineup, then weather, then
    # matchups); keeping only the last text produced a 397-character
    # Vegas-only "report" on 10/9 after the agent had called all five tools.
    # Streaming drafts are excluded by skipping partial events above, which is
    # what the old "last text wins" rule was guarding against.
    return "\n\n".join(t.strip() for t in texts if t.strip())


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

    # Read the lineup BEFORE anything else, and stop if it cannot be read.
    #
    # The tradeoff, made on purpose: an unreadable sheet means NO report
    # today, not a report about some other roster. The fallback below exists
    # for when the agent breaks -- it still checks the right players. There is
    # no equivalent for "we do not know the players": a stale roster reads
    # exactly like a real one ("all 9 active"), and that is the failure this
    # whole change exists to remove. A missing report trips the alert and gets
    # fixed; a confidently wrong one went unnoticed for three weeks.
    try:
        lineup = read_lineup(os.environ.get("LINEUP_SHEET_ID", "").strip())
    except LineupError as e:
        log(f"LINEUP UNAVAILABLE, not running: {e}")
        return 1
    log(f"lineup from sheet: {len(lineup)} players")
    if len(lineup) != EXPECTED_SLOTS:
        # Worth a line in the log, not worth stopping for -- a half-edited
        # sheet still names real players, and the report will show who.
        log(f"note: expected {EXPECTED_SLOTS} players, sheet has {len(lineup)}")

    try:
        if remote:
            log(f"calling deployed agent {resource.rsplit('/', 1)[-1]}")
            report = run_agent_remote(resource, lineup)
        else:
            why = "--local" if a.local else "AGENT_ENGINE_ID unset"
            log(f"running agent in-process ({why})")
            report = asyncio.run(asyncio.wait_for(run_agent(lineup), TIMEOUT_SECONDS))
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
        report = build_report(names=lineup)

    if a.dry_run:
        log("dry run -- not sending")
        print("\n" + "-" * 60)
        print(report)
        print("-" * 60)
        return 0

    # send_discord() prints and returns False when no webhook is set. Fine
    # at a terminal; on a schedule it is a silent non-delivery that exits 0.
    # A run that delivered nothing has failed, so say so with the exit code.
    if not send_discord(report):
        log("NOT DELIVERED: no DISCORD_WEBHOOK_URL")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
