"""Things the report must always do, done in code instead of asked of the model.

The instruction already says to copy every `source_warnings` entry verbatim as
the first line of the report. The eval measured how often that holds: on
2026-10-15, prompt_injection run 5 times per endpoint left the warning out in
1 of 5 on the global endpoint -- the injection was ignored, but the reader was
never told it happened. A "must" the model keeps 80% of the time is a "should".

So the instruction stays (it is still the best place for the wording), and
these callbacks make sure of it:

  reset_warnings      before_agent_callback  -- start each run with none
  collect_warnings    after_tool_callback    -- every tool's source_warnings,
                                                whichever tool returned them
  prepend_warnings    after_model_callback   -- on the final report, put any
                                                warning it does not already
                                                contain on top, verbatim

Generic on purpose: a new tool that returns `source_warnings` is covered
without touching this file. Same principle as the decision not to build a
lineup agent -- a step with one right answer belongs in code, not in a model.
"""

from __future__ import annotations

from google.genai import types

STATE_KEY = "run_source_warnings"


def reset_warnings(callback_context):
    # A plain key, reset per run, rather than ADK's "temp:" prefix: temp keys
    # are trimmed from the event delta as they are appended, and this value has
    # to survive from a tool call to a later model call in the same run.
    callback_context.state[STATE_KEY] = []
    return None


def collect_warnings(tool, args, tool_context, tool_response):
    warnings = tool_response.get("source_warnings") if isinstance(tool_response, dict) else None
    if warnings:
        seen = list(tool_context.state.get(STATE_KEY) or [])
        # Reassigned, not appended in place: ADK records a state change only
        # on assignment, and an in-place append would never reach the callback.
        tool_context.state[STATE_KEY] = seen + [w for w in warnings if w not in seen]
    return None  # never alters what the model sees


def prepend_warnings(callback_context, llm_response):
    content = llm_response.content
    if llm_response.partial or not content or not content.parts:
        return None
    # A turn that still calls tools is not the report. The instruction says
    # write nothing between tool calls; this does not rely on it.
    if any(getattr(p, "function_call", None) for p in content.parts):
        return None
    text = "".join(p.text or "" for p in content.parts if getattr(p, "text", None))
    if not text.strip():
        return None
    missing = [w for w in (callback_context.state.get(STATE_KEY) or []) if w not in text]
    if not missing:
        return None
    report = "\n".join(missing) + "\n\n" + text
    return llm_response.model_copy(update={
        "content": types.Content(role=content.role or "model", parts=[types.Part(text=report)]),
    })
