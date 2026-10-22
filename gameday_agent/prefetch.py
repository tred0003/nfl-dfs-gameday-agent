"""Ask for the tools every report needs without paying a model call to ask.

Every saved run since the multi-user rollout opens the same way: the first
model call reads the instruction and the tool schemas (~5.5K tokens), thinks,
and asks for the same seven tools in one parallel batch. Nothing is being
decided -- the instruction says to call all of them, and the eval's
trajectory checks fail a run that skips one -- so that call is ~40% of the
agent loop's cost (2026-10-19) spent producing a constant.

prefetch_core_tools is a before_model_callback. On the first model call of a
run it answers in the model's place with those core function calls, so ADK
runs the tools through its normal path -- every tool callback (the
source_warnings guard, usage metering) and every event, so report_check,
the eval's trajectory and the stored tool_calls see exactly what they saw
before. The model's first real call is the one that reads the results. It
can still call find_replacements or search_news, which take arguments only
it can choose.

Only for the standard report question. Any other message -- the canary's,
a user asking for a specific check -- gets the model's own first call. The
first CI eval gate after this shipped (2026-10-19) caught why: told "Use
search_news to confirm Rashad Feeney's status", the agent skipped the search
in 1 of 3 local runs and in the gate, because it never saw the request
before the results arrived and they already looked conclusive. Without
prefetch it searched 3 of 3. A request the model never got to act on is not
a constant.

PREFETCH_TOOLS=0 restores the old behaviour.
"""

from __future__ import annotations

import os
import re

from google.adk.models import LlmResponse
from google.genai import types

# The ones the instruction requires on every run, in the order the model
# itself most often asked for them (get_player_history, added 2026-10-22,
# goes last). find_replacements, search_news and search_news_archive are
# left to the model: their arguments are judgments.
CORE_TOOLS = [
    "get_lineup_status",
    "get_cb_matchups",
    "get_slate_weather",
    "get_vegas_context",
    "get_notable_scratches",
    "get_value_picks",
    "get_team_news",
    "get_player_history",
]


# Must match run_report.QUERY (the scheduled report and the web app) --
# tests/test_prefetch.py fails if they drift. A drift only turns prefetch off.
STANDARD_QUERY = "Check my lineup for this Sunday and tell me if anything needs my attention."


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _asked(callback_context) -> str:
    content = getattr(callback_context, "user_content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", None) or "" for p in parts)


def enabled() -> bool:
    return os.environ.get("PREFETCH_TOOLS", "1") != "0"


def prefetch_core_tools(callback_context, llm_request):
    if not enabled() or _norm(_asked(callback_context)) != _norm(STANDARD_QUERY):
        return None
    # Only before anything has been called. Any function call or response in
    # the request means this run is past its first model call.
    for content in llm_request.contents or []:
        for part in content.parts or []:
            if getattr(part, "function_call", None) or getattr(part, "function_response", None):
                return None
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name=name, args={}))
        for name in CORE_TOOLS
    ]))
