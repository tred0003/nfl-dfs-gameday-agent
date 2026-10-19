"""Count Gemini tokens across an agent run and turn them into dollars.

Every Gemini response carries usage_metadata. An agent run is several model
calls (decide, call a tool, read the result, decide again, write), so the
cost of one report is the SUM over every model response in the run. Both the
eval harness (ADK Event objects) and the daily report (Agent Engine dicts)
feed their events through add_usage(), so the two report the same thing.

Gemini calls made INSIDE a tool -- the news extractor, check_claims, the
injection screen, search_news -- never appear as agent events. Until
2026-10-19 they were not counted at all, so the quoted cost was a floor.
gameday_agent/metering.py now records each one into session state, and
subcalls_from_state() reads them back out, so a run's bill is the agent's
own calls plus every sub-call. What is still not counted: the Google Search
grounding fee search_news pays per request, which is billed per request,
not per token.

Cached input. Gemini 2.5 caches a repeated prompt prefix on its own
(implicit caching) and bills those tokens at a 90% discount.
`prompt_token_count` INCLUDES them; `cached_content_token_count` says how
many. Counting the whole prompt at the full rate overstated input cost on
every later turn of a run.

Prices: Vertex AI standard tier, USD per 1M tokens, prompts under 200K,
checked 2026-10-16 at cloud.google.com/vertex-ai/generative-ai/pricing.
Cached input is 10% of the input rate (the 90% implicit-cache discount on
2.5 models); flash-lite was added 2026-10-19 for the sub-call experiments.
Output is billed INCLUDING thinking tokens, which 2.5 models spend a lot of
and which usage_metadata reports separately -- leave them out and the cost
comes out several times too low.
"""

from __future__ import annotations

PRICES_PER_M = {
    "gemini-2.5-flash": {"input": 0.30, "cached_input": 0.030, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "cached_input": 0.010, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
}
PRICES_CHECKED = "2026-10-16"

# usage_metadata field -> our name. Agent Engine returns snake_case dicts;
# some SDK paths serialise camelCase. Both are accepted.
_FIELDS = {
    "prompt_token_count": "input", "promptTokenCount": "input",
    "candidates_token_count": "output", "candidatesTokenCount": "output",
    "thoughts_token_count": "thinking", "thoughtsTokenCount": "thinking",
    "cached_content_token_count": "cached", "cachedContentTokenCount": "cached",
}


def new_totals() -> dict:
    return {"input": 0, "cached": 0, "output": 0, "thinking": 0, "model_calls": 0}


def add_usage(totals: dict, usage) -> None:
    """Add one response's usage_metadata (object or dict) into totals."""
    if not usage:
        return
    if not isinstance(usage, dict):
        usage = {k: getattr(usage, k, None) for k in
                 ("prompt_token_count", "candidates_token_count", "thoughts_token_count",
                  "cached_content_token_count")}
    counted = False
    for field, name in _FIELDS.items():
        v = usage.get(field)
        if isinstance(v, int):
            totals[name] = totals.get(name, 0) + v
            counted = counted or name != "cached"
    if counted:
        totals["model_calls"] += 1


def call_record(usage, model: str, requested_tools: list[str] | None = None,
                wrote_text: bool = False) -> dict | None:
    """One model call's usage and cost, for the per-call breakdown.

    `requested_tools` are the function calls that response asked for (one
    response can ask for several at once -- parallel function calling), and
    `wrote_text` says whether it wrote report text. Together they say what
    that call was FOR: deciding which tools to run, or writing the report.
    """
    one = new_totals()
    add_usage(one, usage)
    if not one["model_calls"]:
        return None
    return {"input": one["input"], "cached": one["cached"], "output": one["output"],
            "thinking": one["thinking"], "cost_usd": cost_usd(one, model),
            "requested_tools": list(requested_tools or []), "wrote_text": wrote_text}


def cost_usd(totals: dict, model: str) -> float | None:
    """Dollar cost of the totals at `model`'s price, or None for an unknown model."""
    price = PRICES_PER_M.get(model)
    if price is None or not totals:
        return None
    cached = min(totals.get("cached", 0), totals.get("input", 0))
    billed_out = totals.get("output", 0) + totals.get("thinking", 0)
    return round(((totals.get("input", 0) - cached) * price["input"]
                  + cached * price["cached_input"]
                  + billed_out * price["output"]) / 1_000_000, 6)


# ---------------------------------------------------------------- sub-calls
# Must match gameday_agent/metering.py. Not imported from there: the
# scheduled image ships src/ but not the agent package.
SUBCALL_STATE_PREFIX = "llm_subcalls:"


def subcall_record(label: str, model: str, usage) -> dict | None:
    """One Gemini call made inside a tool: what it was for, its tokens and cost."""
    one = new_totals()
    add_usage(one, usage)
    if not one["model_calls"]:
        return None
    return {"label": label, "model": model, "input": one["input"], "cached": one["cached"],
            "output": one["output"], "thinking": one["thinking"],
            "cost_usd": cost_usd(one, model)}


def subcalls_from_state(state_delta) -> list[dict]:
    """The sub-call records in one event's state delta (dict, or None)."""
    out = []
    for key, value in (state_delta or {}).items():
        if key.startswith(SUBCALL_STATE_PREFIX) and isinstance(value, list):
            out += [v for v in value if isinstance(v, dict)]
    return out


def subcall_cost(subcalls: list[dict]) -> float:
    """Summed dollars over sub-call records. An unpriced model counts as 0."""
    return round(sum(c.get("cost_usd") or 0 for c in subcalls), 6)
