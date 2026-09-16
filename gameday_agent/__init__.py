"""ADK package for the gameday agent.

ADK *imports* this package to find `root_agent` -- it never runs it as a script.
That is why the path bootstrap lives here instead of in agent.py: it has to
happen before anything imports the project's shared modules out of src/.

The path is derived from __file__ so it resolves the same way in both places
this package has to run:

    laptop      nfl-dfs/gameday_agent/  ->  nfl-dfs/src/
    deployed    /app/gameday_agent/     ->  /app/src/

That second layout is what `adk deploy agent_engine --extra_packages src`
produces -- each extra package is staged at /app/<basename>. Keep that flag on
every deploy, or the agent imports cleanly on the laptop and dies in the cloud.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from . import agent  # noqa: E402  -- must follow the path bootstrap above

__all__ = ["agent"]
