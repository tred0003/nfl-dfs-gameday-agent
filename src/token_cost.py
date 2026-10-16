"""Count Gemini tokens across an agent run and turn them into dollars.

Every Gemini response carries usage_metadata. An agent run is several model
calls (decide, call a tool, read the result, decide again, write), so the
cost of one report is the SUM over every model response in the run. Both the
eval harness (ADK Event objects) and the daily report (Agent Engine dicts)
feed their events through add_usage(), so the two report the same thing.

What it does NOT count: Gemini calls made INSIDE a tool -- the news
extractor, check_claims, the injection screen. Those run in the tool, not in
the agent's loop, so their usage never appears as an agent event. The number
here is the agent's own reasoning cost, a floor rather than the full bill.

Prices: Vertex AI standard tier, USD per 1M tokens, prompts under 200K,
checked 2026-10-16 at cloud.google.com/vertex-ai/generative-ai/pricing.
Output is billed INCLUDING thinking tokens, which 2.5 models spend a lot of
and which usage_metadata reports separately -- leave them out and the cost
comes out several times too low.
"""

from __future__ import annotations

PRICES_PER_M = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
}
PRICES_CHECKED = "2026-10-16"

# usage_metadata field -> our name. Agent Engine returns snake_case dicts;
# some SDK paths serialise camelCase. Both are accepted.
_FIELDS = {
    "prompt_token_count": "input", "promptTokenCount": "input",
    "candidates_token_count": "output", "candidatesTokenCount": "output",
    "thoughts_token_count": "thinking", "thoughtsTokenCount": "thinking",
}


def new_totals() -> dict:
    return {"input": 0, "output": 0, "thinking": 0, "model_calls": 0}


def add_usage(totals: dict, usage) -> None:
    """Add one response's usage_metadata (object or dict) into totals."""
    if not usage:
        return
    if not isinstance(usage, dict):
        usage = {k: getattr(usage, k, None) for k in
                 ("prompt_token_count", "candidates_token_count", "thoughts_token_count")}
    counted = False
    for field, name in _FIELDS.items():
        v = usage.get(field)
        if isinstance(v, int):
            totals[name] += v
            counted = True
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
    return {"input": one["input"], "output": one["output"], "thinking": one["thinking"],
            "cost_usd": cost_usd(one, model), "requested_tools": list(requested_tools or []),
            "wrote_text": wrote_text}


def cost_usd(totals: dict, model: str) -> float | None:
    """Dollar cost of the totals at `model`'s price, or None for an unknown model."""
    price = PRICES_PER_M.get(model)
    if price is None or not totals:
        return None
    billed_out = totals.get("output", 0) + totals.get("thinking", 0)
    return round((totals.get("input", 0) * price["input"]
                  + billed_out * price["output"]) / 1_000_000, 6)
