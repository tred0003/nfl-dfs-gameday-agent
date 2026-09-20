"""ADK package for the Sleeper dynasty strategy agent.

ADK *imports* this package to find `root_agent` -- it never runs it as a
script. The path bootstrap lives here rather than in agent.py because it has to
happen before anything imports sleeper_client out of src/.

Same two layouts as gameday_agent:

    laptop      nfl-dfs/sleeper_agent/  ->  nfl-dfs/src/
    deployed    /app/sleeper_agent/     ->  /app/src/

That second layout is what `adk deploy agent_engine --extra_packages src`
produces. Keep that flag on every deploy or the agent imports cleanly on the
laptop and dies in the cloud.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from . import agent  # noqa: E402  -- must follow the path bootstrap above

__all__ = ["agent"]
