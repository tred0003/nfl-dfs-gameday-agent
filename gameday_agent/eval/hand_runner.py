"""The runner, by hand, with no ADK. ~30 lines of actual loop.

ADK's `Runner` is not a mysterious engine. It is a while loop that keeps a
list of messages and calls your functions when the model asks. This file is
that loop, written out, driving the SAME instruction and the SAME five tools
as the real agent -- so you can read it once and stop wondering what ADK is
doing on your behalf.

What ADK adds on top of this loop, and why you'd still use it in production:
sessions and state, streaming, callbacks, tracing, retries, parallel tool
calls, auth, and a deployment story. None of that changes the shape below.

    python gameday_agent/eval/hand_runner.py

Compare its output to `adk run gameday_agent "..."`. Same model, same prompt,
same tools, same job -- one written by you, one written by the framework.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# ADK reads gameday_agent/.env for you. Nothing here does, so the first run of
# this file reported "your lineup is not configured" -- MY_LINEUP simply wasn't
# in the environment. A small, real example of what a framework is doing on
# your behalf while you assume it's the model's behaviour.
for line in (ROOT / "gameday_agent" / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "advisor-agent-2026")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

from google import genai  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402
from google.genai import types  # noqa: E402

from gameday_agent.agent import INSTRUCTION, MODEL  # noqa: E402
from gameday_agent.tools import TOOLS  # noqa: E402

QUERY = "Check my lineup for this Sunday and tell me if anything needs my attention."
MAX_TURNS = 8

# Name -> the actual Python function, so we can run what the model asks for.
FUNCS = {f.__name__: f for f in TOOLS}

# The JSON schemas the model sees. Borrowed from ADK purely so the schemas
# match the real agent exactly -- generating them is not the interesting part.
DECLARATIONS = [FunctionTool(f)._get_declaration() for f in TOOLS]


def main() -> None:
    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
    )

    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTION,
        tools=[types.Tool(function_declarations=DECLARATIONS)],
        temperature=0.2,
        # Off on purpose. With this enabled the SDK would run the loop for us,
        # which is exactly the thing this file exists to make visible.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # THE CONVERSATION. Every turn appends to it, and the whole thing is re-sent
    # each time -- the model remembers nothing on its own.
    contents = [types.Content(role="user", parts=[types.Part(text=QUERY)])]

    # ------------------------------------------------------------------ LOOP
    for turn in range(1, MAX_TURNS + 1):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )
        parts = response.candidates[0].content.parts or []
        calls = [p.function_call for p in parts if p.function_call]

        if not calls:
            text = "".join(p.text for p in parts if p.text)
            print(f"\n--- turn {turn}: model answered, loop ends ---\n")
            print(text.strip())
            usage = response.usage_metadata
            print(f"\n[{turn} model calls, "
                  f"{usage.prompt_token_count:,} prompt tokens on the last one]")
            return

        # The model asked for tools. Keep its request in the transcript...
        contents.append(response.candidates[0].content)

        for call in calls:
            args = dict(call.args or {})
            print(f"--- turn {turn}: model asked for {call.name}({args}) ---")
            result = FUNCS[call.name](**args)          # <-- YOUR code runs. Not the model's.
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=call.name, response=result)],
            ))
    # ---------------------------------------------------------------- END LOOP

    print(f"\nHit MAX_TURNS ({MAX_TURNS}) without a final answer.")


if __name__ == "__main__":
    main()
