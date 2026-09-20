"""ADK package for the gameday agent.

ADK *imports* this package to find `root_agent` -- it never runs it as a script.
That is why the path bootstrap lives here instead of in agent.py: it has to
happen before anything imports the project's shared modules out of src/.

The path is derived from __file__, but NOT by assuming a fixed depth. It walks
up looking for the real src/ instead, because the depth is not stable:

    laptop      nfl-dfs/gameday_agent/      ->  nfl-dfs/src/
    ADK <=2.7   /app/gameday_agent/         ->  /app/src/
    ADK 2.8     /app/agents/gameday_agent/  ->  /app/src/

This file used to hardcode `parent.parent / "src"` and warn that dropping
`--extra_packages src` would kill the deploy. The flag was right; the layout
moved. ADK 2.8 stages the agent under agents/, so parent.parent resolved to
/app/agents and the deployed agent died with:

    ModuleNotFoundError: Fail to load 'gameday_agent' module.
    No module named 'predict_slate'

Walking upward finds src/ at either depth and at any future one. Still keep
`--extra_packages src` on every deploy -- without it there is no src/ to find.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_src() -> Path | None:
    """Nearest ancestor directory holding src/predict_slate.py."""
    here = Path(__file__).resolve()
    for base in here.parents:
        candidate = base / "src"
        if (candidate / "predict_slate.py").is_file():
            return candidate
    return None


_SRC = _find_src()
if _SRC is not None and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from . import agent  # noqa: E402  -- must follow the path bootstrap above

__all__ = ["agent"]
