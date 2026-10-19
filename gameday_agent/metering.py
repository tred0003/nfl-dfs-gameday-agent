"""What the Gemini calls INSIDE tools cost, carried back to whoever ran the agent.

The agent's own model calls arrive as events with usage_metadata, and the
drivers (run_report.py, eval/run_eval.py) sum those. The calls a tool makes
-- the injection screen, news extraction, check_claims, search_news -- do
not: they happen inside a function body, and nothing about them leaves it.
Until 2026-10-19 that made the quoted cost per report a floor, with the news
pipeline (the biggest prompts in a run) entirely missing.

How the numbers get out, in three steps:

  start_subcalls    before_tool_callback  -- a fresh list in a ContextVar
  generate()        every in-tool Gemini call goes through it and appends
                    its usage to that list. ADK runs a sync tool in an
                    executor with a COPY of the context, so the copy holds
                    the same list and the append is visible afterwards.
  collect_subcalls  after_tool_callback   -- write the list to session state
                    under a key unique to this function call

A key per call, not one shared list: parallel tool calls each get their own
state delta, and two of them read-modify-writing one key would keep only the
last. State deltas travel with events, so the remote driver sees them in
stream_query's dicts and the local ones in event.actions.

A call made outside any tool (eval/injection_check.py calling screen()
directly) has no list to append to and is simply not recorded.
"""

from __future__ import annotations

import contextvars

from token_cost import SUBCALL_STATE_PREFIX, subcall_record

_CALLS: contextvars.ContextVar[list | None] = contextvars.ContextVar("llm_subcalls", default=None)


def generate(client, label: str, *, model: str, contents, config=None):
    """client.models.generate_content, with its usage recorded under `label`."""
    resp = client.models.generate_content(model=model, contents=contents, config=config)
    record(label, model, getattr(resp, "usage_metadata", None))
    return resp


def record(label: str, model: str, usage) -> None:
    calls = _CALLS.get()
    rec = subcall_record(label, model, usage)
    if calls is not None and rec is not None:
        calls.append(rec)


def start_subcalls(tool, args, tool_context):
    _CALLS.set([])
    return None


def collect_subcalls(tool, args, tool_context, tool_response):
    calls = _CALLS.get()
    if calls:
        key = f"{SUBCALL_STATE_PREFIX}{tool_context.function_call_id or tool.name}"
        tool_context.state[key] = [dict(c, tool=tool.name) for c in calls]
    _CALLS.set(None)
    return None  # never alters what the model sees
