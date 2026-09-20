"""ADK package for the Sleeper dynasty strategy agent.

ADK *imports* this package to find `root_agent` -- it never runs it as a
script. The path bootstrap lives here rather than in agent.py because it has to
happen before anything imports sleeper_client out of src/.

Walks up to find src/ rather than assuming a fixed depth, for the reason
gameday_agent/__init__.py documents at length: ADK 2.8 stages the agent at
/app/agents/<name>/ where earlier versions used /app/<name>/, so a hardcoded
parent.parent lands on a directory with no src/ in it and the deployed agent
dies on its first import.

    laptop      nfl-dfs/sleeper_agent/      ->  nfl-dfs/src/
    ADK 2.8     /app/agents/sleeper_agent/  ->  /app/src/

Still pass `--extra_packages src` on every deploy -- without it there is no
src/ to find at any depth.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_src() -> Path | None:
    """Nearest ancestor directory holding src/sleeper_client.py."""
    here = Path(__file__).resolve()
    for base in here.parents:
        candidate = base / "src"
        if (candidate / "sleeper_client.py").is_file():
            return candidate
    return None


_SRC = _find_src()
if _SRC is not None and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from . import agent  # noqa: E402  -- must follow the path bootstrap above

__all__ = ["agent"]
