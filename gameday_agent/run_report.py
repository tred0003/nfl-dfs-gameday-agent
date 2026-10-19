"""Headless entry point: run the ADK agent once per user, store and send the reports.

This is what the gameday-report Cloud Run job runs every day. Nobody is watching
it, nobody types a question, and there is no second turn -- so delivery is NOT
the model's decision. The model writes the report; this script sends it.

    python gameday_agent/run_report.py --dry-run    # print, never store or send
    python gameday_agent/run_report.py              # every active user
    python gameday_agent/run_report.py --user u2    # one user
    python gameday_agent/run_report.py --no-discord # store, but post nothing
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

THE LINEUP COMES WITH THE REQUEST. This client reads each user's lineup from
Firestore (src/user_store.py, users/{uid}) and creates that user's agent
session with it in state, so the agent checks whatever the user saved right
now. It used to be MY_LINEUP baked into the deployment, which went three weeks
stale, and then one Google Sheet, which could only ever hold one person. The
week is derived from the slate date (src/nfl_week.py), for the same reason.

ONE USER AT A TIME. Users run in sequence, each in their own session with
user_id = their uid, and each report is saved to users/{uid}/reports/{date}.
Sequential on purpose: four agent runs at once is four times the Gemini
requests in the same minute, and 429s are this project's most common failure.
One user failing (no lineup, agent down and fallback down) does not stop the
others; it makes the job exit 1 at the end, so the alert still fires. Only
users with `discord: true` are posted -- the shared webhook is one channel.

Two env files are read, on purpose:
    nfl-dfs/.env                  DISCORD_WEBHOOK_URL, AGENT_ENGINE_ID --
                                  the CLIENT's config, never staged into
                                  the agent
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
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

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

from token_cost import (  # noqa: E402
    add_usage, call_record, cost_usd, new_totals, subcall_cost, subcalls_from_state,
)
from user_store import FirestoreStore, LineupError, User  # noqa: E402
from report_check import audit as audit_report, correct as correct_report  # noqa: E402
from watch_slate import load_env, send_discord  # noqa: E402

# Must match gameday_agent/slate.py LINEUP_STATE_KEY. Not imported from there:
# that would import the agent package, which the scheduled image does not ship.
LINEUP_STATE_KEY = "my_lineup"

QUERY = "Check my lineup for this Sunday and tell me if anything needs my attention."

# A run that hangs on a wedged HTTP call must not sit there until the next one
# starts. The Cloud Run task timeout is the backstop, not the plan.
TIMEOUT_SECONDS = 300

# DraftKings classic NFL: QB, 2 RB, 3 WR, TE, FLEX, DST.
EXPECTED_SLOTS = 9


def log(msg: str) -> None:
    """Timestamped, flushed -- this is the only record of an unattended run."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _record_event(record: dict, usage, function_calls: list[dict],
                  wrote_text: bool) -> None:
    """Add one non-partial agent event to the run record (see run_agent_remote).

    Tool calls go in with their arguments, in order. A model call is any event
    carrying usage_metadata; it is tagged with the tools it asked for and
    whether it wrote text, so the dashboard can say what each call was for.
    """
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    for fc in function_calls:
        record.setdefault("tool_calls", []).append(
            {"name": fc.get("name", ""), "args": dict(fc.get("args") or {})})
    rec = call_record(usage, model, [fc.get("name", "") for fc in function_calls],
                      wrote_text)
    if rec:
        record.setdefault("model_calls", []).append(rec)


def run_agent_remote(resource: str, lineup: list[str], query: str = QUERY,
                     responses: list[dict] | None = None,
                     user_id: str = "scheduled",
                     tokens: dict | None = None,
                     record: dict | None = None) -> str:
    """Call the agent deployed on Agent Platform and return its final report.

    This is the mode the scheduled job runs in. The agent lives in the cloud;
    this script is a thin client that asks it a question, takes the answer and
    delivers it. Delivery and the fallback stay OUT HERE on purpose -- if the
    deployed agent is down, out of quota or mid-redeploy, something that is
    not the agent has to notice and still send a report.

    The event shape differs from the in-process runner. Locally ADK yields
    Event objects with attribute access; stream_query yields plain dicts. Same
    content, different surface, so the two loops below cannot be shared.

    `query` and `responses` exist for the Thursday canary (canary.py): it asks
    a different question and needs to see what each tool RETURNED, not just
    that it was called. The daily report passes neither.

    `user_id` is the Firestore uid. Agent Engine keys sessions by it, so each
    user's session is theirs alone, and the agent's logs say whose run it was.

    `tokens`, if given, is filled with the run's summed Gemini usage
    (src/token_cost.py) -- what this one report cost to write. `record`, if
    given, gets the detail behind it for the eval dashboard: every tool call
    with its arguments, and every model call with its own tokens and cost.
    """
    import vertexai
    from vertexai import agent_engines

    vertexai.init(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "advisor-agent-2026"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    agent = agent_engines.get(resource)

    # A fresh session per run, carrying this user's lineup in state. The
    # agent's lineup tools read state before they look at env, so this is
    # what makes a lineup edit take effect with no redeploy.
    session = agent.create_session(user_id=user_id,
                                   state={LINEUP_STATE_KEY: lineup})
    session_id = session["id"] if isinstance(session, dict) else session.id

    texts, calls = [], []
    try:
        for event in agent.stream_query(user_id=user_id, session_id=session_id,
                                        message=query):
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
            if tokens is not None:
                add_usage(tokens, event.get("usage_metadata"))
            if record is not None:
                record.setdefault("subcalls", []).extend(subcalls_from_state(
                    (event.get("actions") or {}).get("state_delta")))
            parts = (event.get("content", {}) or {}).get("parts", []) or []
            if record is not None:
                _record_event(record, event.get("usage_metadata"),
                              [p["function_call"] for p in parts if p.get("function_call")],
                              any(p.get("text") for p in parts))
            for part in parts:
                if part.get("function_call"):
                    calls.append(part["function_call"]["name"])
                if responses is not None and part.get("function_response"):
                    responses.append(part["function_response"])
                if part.get("text"):
                    texts.append(part["text"])
    finally:
        # Managed sessions persist and bill per event. One run, one session,
        # then gone. A failed delete is not worth failing the report over.
        try:
            agent.delete_session(user_id=user_id, session_id=session_id)
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


async def run_agent(lineup: list[str], responses: list[dict] | None = None,
                    user_id: str = "scheduled", tokens: dict | None = None,
                    record: dict | None = None) -> str:
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
        app_name="gameday_report", user_id=user_id,
        state={LINEUP_STATE_KEY: lineup},
    )

    texts, tool_calls = [], []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=QUERY)]),
    ):
        if getattr(event, "error_code", None):
            raise RuntimeError(f"agent error {event.error_code}: "
                               f"{str(getattr(event, 'error_message', ''))[:200]}")
        if getattr(event, "partial", False):
            continue
        if tokens is not None:
            add_usage(tokens, getattr(event, "usage_metadata", None))
        if record is not None:
            record.setdefault("subcalls", []).extend(subcalls_from_state(
                getattr(getattr(event, "actions", None), "state_delta", None)))
        if not (event.content and event.content.parts):
            continue
        if record is not None:
            fcs = [p.function_call for p in event.content.parts
                   if getattr(p, "function_call", None)]
            _record_event(record, getattr(event, "usage_metadata", None),
                          [{"name": fc.name, "args": dict(fc.args or {})} for fc in fcs],
                          any(getattr(p, "text", None) for p in event.content.parts))
        for part in event.content.parts:
            if getattr(part, "function_call", None):
                tool_calls.append(part.function_call.name)
            fr = getattr(part, "function_response", None)
            if responses is not None and fr:
                responses.append({"name": fr.name, "response": fr.response})
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


def produce_report(user: User, resource: str, local: bool,
                   no_fallback: bool) -> dict:
    """One user's report: agent, fact-check, fallback. Returns the fields to store.

    Raises LineupError when the user has no lineup -- see the tradeoff in
    run_users. Everything else that goes wrong becomes the rule-based fallback,
    unless --no-fallback, in which case it raises too.
    """
    lineup = user.checked_lineup()
    log(f"[{user.uid}] lineup: {len(lineup)} players")
    if len(lineup) != EXPECTED_SLOTS:
        # Worth a line in the log, not worth stopping for -- a half-edited
        # lineup still names real players, and the report will show who.
        log(f"[{user.uid}] note: expected {EXPECTED_SLOTS} players, got {len(lineup)}")

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    # Stored with every report, agent or fallback: who, what they submitted,
    # and which model and agent produced it.
    base = {"lineup": list(lineup), "model": model,
            "agent": resource.rsplit("/", 1)[-1] if resource and not local else "local"}

    # What each tool RETURNED, kept so report_check can hold the report to it.
    responses: list[dict] = []
    tokens = new_totals()
    record: dict = {"tool_calls": [], "model_calls": [], "subcalls": []}
    started = time.monotonic()
    try:
        if resource and not local:
            log(f"[{user.uid}] calling deployed agent {resource.rsplit('/', 1)[-1]}")
            report = run_agent_remote(resource, lineup, responses=responses,
                                      user_id=user.uid, tokens=tokens, record=record)
        else:
            log(f"[{user.uid}] running agent in-process")
            report = asyncio.run(asyncio.wait_for(
                run_agent(lineup, responses, user_id=user.uid, tokens=tokens,
                          record=record),
                TIMEOUT_SECONDS))
        if not report:
            raise RuntimeError("agent returned an empty report")
        log(f"[{user.uid}] agent produced {len(report)} characters")

        # Fact-check before delivery. A failed check does not block the post:
        # the facts the model missed go on top, and the rest of its report
        # stays. "REPORT CHECK FAILED" is the line to build a log metric on.
        checks = audit_report(report, responses)
        problems = [p for row in checks for p in row["problems"]]
        agent_text = report
        if problems:
            log(f"[{user.uid}] REPORT CHECK FAILED ({len(problems)}): "
                + " | ".join(problems))
            report = correct_report(report, problems)
        else:
            log(f"[{user.uid}] report check passed")
        tools = [r.get("name", "") for r in responses]
        cost = cost_usd(tokens, model)
        sub_cost = subcall_cost(record["subcalls"])
        total = None if cost is None else round(cost + sub_cost, 6)
        log(f"[{user.uid}] tokens: {tokens} (agent ${cost} + "
            f"{len(record['subcalls'])} in-tool calls ${sub_cost} = ${total})")
        return {**base, "text": report, "agent_text": agent_text, "source": "agent",
                "problems": problems, "checks": checks, "tools": tools,
                "tokens": tokens, "cost_usd": cost,
                "subcalls": record["subcalls"], "subcall_cost_usd": sub_cost,
                "total_cost_usd": total,
                "model_calls": record["model_calls"],
                "tool_calls": pair_tool_calls(record["tool_calls"], responses),
                "elapsed_s": round(time.monotonic() - started, 1)}
    except Exception as e:
        log(f"[{user.uid}] agent failed: {type(e).__name__}: {e}")
        if no_fallback:
            raise
        # Same call as src/agent.py makes, for the same reason: a scheduled
        # Sunday job that silently sends nothing is worse than one that sends
        # the deterministic version. The rule-based report needs no model and
        # no Agent Platform quota, so it survives most things that kill the LLM.
        log(f"[{user.uid}] falling back to the rule-based report")
        from gameday_report import build_report
        return {**base, "text": build_report(names=lineup), "source": "fallback",
                "problems": [], "tools": [],
                "agent_error": f"{type(e).__name__}: {str(e)[:300]}",
                "elapsed_s": round(time.monotonic() - started, 1)}


# One tool result can be a whole news article list. Firestore caps a document
# at 1 MiB, so each result is stored as capped JSON text: a string, because
# Firestore also rejects arrays nested directly in arrays, which tool results
# are free to contain.
MAX_RESULT_CHARS = 60_000


def pair_tool_calls(calls: list[dict], responses: list[dict]) -> list[dict]:
    """Each tool call with what it returned, in call order, as JSON text.

    Paired by name and order: the Nth call to a tool gets that tool's Nth
    response. Calls in one model response run in parallel, so responses can
    come back in a different order than the calls went out.
    """
    by_name: dict[str, list] = {}
    for r in responses:
        by_name.setdefault(r.get("name", ""), []).append(r.get("response"))
    out = []
    for c in calls:
        queue = by_name.get(c["name"]) or []
        result = queue.pop(0) if queue else None
        text = json.dumps(result, default=str, ensure_ascii=False)
        out.append({"name": c["name"],
                    "args_json": json.dumps(c.get("args") or {}, default=str),
                    "result_json": text[:MAX_RESULT_CHARS],
                    "result_chars": len(text),
                    "truncated": len(text) > MAX_RESULT_CHARS})
    return out


def run_users(users: list[User], store, produce: Callable[[User], dict],
              deliver: Callable[[str], bool] | None, report_id: str,
              dry_run: bool = False) -> int:
    """Run every user in turn. Returns how many FAILED (0 = all good).

    `produce` and `deliver` are passed in so the unit tests can run this loop
    with a MemoryStore and a fake agent -- no credentials, no Gemini, no Discord.
    `deliver` None means --no-discord.

    The tradeoff, kept from the Sheet days: a user with no lineup gets NO
    report today, not a report about some other roster. The fallback exists
    for when the agent breaks -- it still checks the right players. Nothing
    covers "we do not know the players", and a stale or empty roster reads
    exactly like a real one ("all 9 active"). A missing report trips the
    alert and gets fixed; a confidently wrong one went unnoticed for weeks.
    """
    failed = 0
    for user in users:
        try:
            fields = produce(user)
        except LineupError as e:
            log(f"[{user.uid}] LINEUP UNAVAILABLE, no report: {e}")
            failed += 1
            continue
        except Exception as e:  # noqa: BLE001 -- one user must not stop the rest
            log(f"[{user.uid}] FAILED: {type(e).__name__}: {e}")
            failed += 1
            continue

        if dry_run:
            log(f"[{user.uid}] dry run -- not storing or sending")
            print("\n" + "-" * 60 + f"\n{user.uid} ({user.name or user.email})")
            print(fields["text"])
            print("-" * 60)
            continue

        # Stored before it is posted: the stored copy is the record, and the
        # web page (step 2) reads it whether or not Discord worked.
        store.save_report(user.uid, report_id, fields)
        log(f"[{user.uid}] saved users/{user.uid}/reports/{report_id} "
            f"({fields['source']})")

        if user.discord and deliver is not None:
            # send_discord() prints and returns False when no webhook is set.
            # Fine at a terminal; on a schedule it is a silent non-delivery
            # that exits 0. A run that delivered nothing has failed.
            if not deliver(fields["text"]):
                log(f"[{user.uid}] NOT DELIVERED: no DISCORD_WEBHOOK_URL")
                failed += 1
    return failed


def main() -> int:
    p = argparse.ArgumentParser(description="Run the gameday agent for each user")
    p.add_argument("--dry-run", action="store_true",
                   help="print the reports instead of storing or sending them")
    p.add_argument("--no-fallback", action="store_true",
                   help="fail loudly instead of sending the rule-based report")
    p.add_argument("--local", action="store_true",
                   help="run the agent in this process instead of calling the "
                        "deployed one (for testing an unpushed change)")
    p.add_argument("--user", metavar="UID",
                   help="run one user (active or not) instead of every active user")
    p.add_argument("--no-discord", action="store_true",
                   help="store the reports but post nothing to Discord")
    a = p.parse_args()

    load_env()  # DISCORD_WEBHOOK_URL and AGENT_ENGINE_ID from the repo-root .env

    # The resource id lives in the REPO-ROOT .env, not the agent's own. Anything
    # in gameday_agent/.env is staged into the deployment by `adk deploy`, and
    # an agent carrying its own resource name around is both pointless and one
    # more value to keep in sync.
    resource = os.environ.get("AGENT_ENGINE_ID", "").strip()
    if not resource and not a.local:
        log("AGENT_ENGINE_ID unset -- running the agent in-process")

    # Read the users BEFORE anything else, and stop if they cannot be read:
    # no users means no reports, and that has to reach the alert as exit 1.
    try:
        store = FirestoreStore()
        if a.user:
            one = store.get_user(a.user)
            users = [one] if one else []
        else:
            users = store.active_users()
    except Exception as e:  # noqa: BLE001 -- auth, network, missing database
        log(f"USERS UNAVAILABLE, not running: {type(e).__name__}: {e}")
        return 1
    if not users:
        log(f"no {'user ' + a.user if a.user else 'active users'} in Firestore")
        return 1
    log(f"{len(users)} user(s): {', '.join(u.uid for u in users)}")

    # One report per user per day, named by the ET date. A rerun the same day
    # replaces that day's report rather than piling up copies.
    report_id = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    failed = run_users(
        users, store,
        produce=lambda u: produce_report(u, resource, a.local, a.no_fallback),
        deliver=None if a.no_discord else send_discord,
        report_id=report_id, dry_run=a.dry_run)
    log(f"done: {len(users) - failed}/{len(users)} users OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
