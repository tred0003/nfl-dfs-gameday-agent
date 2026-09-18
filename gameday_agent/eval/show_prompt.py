"""Print exactly what the model receives — and what it doesn't.

There is a persistent confusion worth killing with evidence: the model does
not read your source file. It does not see module docstrings, comments, or
function bodies. It sees three things, and this script prints all three:

    1. the `instruction` string on the agent      (the system prompt)
    2. one declaration per tool, built from the   (the tool spec)
       function name, its parameters, and its
       docstring
    3. whatever the user says, plus tool results  (at runtime)

That is the entire surface. Everything else in these files is for humans.

Run it after changing any tool docstring, because changing a tool docstring
IS a prompt change and this is the diff you actually shipped:

    python gameday_agent/eval/show_prompt.py
    python gameday_agent/eval/show_prompt.py --tool get_lineup_status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from google.adk.tools import FunctionTool  # noqa: E402

from gameday_agent.agent import root_agent  # noqa: E402

RULE = "=" * 78


def _declaration(tool):
    """The FunctionDeclaration ADK hands to Gemini for one tool."""
    if not isinstance(tool, FunctionTool):
        tool = FunctionTool(tool)
    return tool._get_declaration()


def _render_params(schema, indent: str = "    ") -> list[str]:
    """Flatten a parameter schema into readable lines."""
    if schema is None or not getattr(schema, "properties", None):
        return [f"{indent}(no parameters)"]

    required = set(getattr(schema, "required", None) or [])
    out = []
    for pname, p in schema.properties.items():
        ptype = getattr(p.type, "name", str(p.type)).lower() if p.type else "?"
        flag = "required" if pname in required else "optional"
        out.append(f"{indent}{pname}: {ptype}  ({flag})")
        if getattr(p, "description", None):
            out.append(f"{indent}    ↳ {p.description}")
    return out


def main(only: str | None) -> None:
    print(RULE)
    print(f"MODEL: {root_agent.model}")
    print(f"AGENT NAME: {root_agent.name}")
    temp = getattr(root_agent.generate_content_config, "temperature", None)
    print(f"TEMPERATURE: {temp}")
    print(RULE)

    print("\nSYSTEM PROMPT — the `instruction` field, sent verbatim every run\n")
    print(root_agent.instruction.rstrip())

    tools = list(root_agent.tools)
    print()
    print(RULE)
    print(f"TOOL DECLARATIONS — {len(tools)} tools, built from signature + docstring")
    print(RULE)

    for tool in tools:
        decl = _declaration(tool)
        if only and decl.name != only:
            continue
        print(f"\n--- {decl.name} " + "-" * max(0, 70 - len(decl.name)))
        print("\nDESCRIPTION (this is the docstring, and it is prompt):")
        for line in (decl.description or "(none)").strip().splitlines():
            print(f"  {line}")
        print("\nPARAMETERS:")
        for line in _render_params(decl.parameters):
            print(line)

    print()
    print(RULE)
    print("NOT SENT TO THE MODEL")
    print(RULE)
    print("""
  - module docstrings (the text at the top of agent.py, tools.py, slate.py)
  - function bodies, and every comment in them
  - helper functions that are not registered as tools
  - the DataFrame work, the caching, the error handling

  If you want the model to know something, it goes in the instruction or in a
  tool docstring. Nowhere else counts.
""".rstrip())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Show what the model actually receives")
    p.add_argument("--tool", help="show only this tool's declaration")
    main(p.parse_args().tool)
