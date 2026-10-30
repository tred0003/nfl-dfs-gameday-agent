"""The web app's browser scripts must parse: python -m unittest tests.test_static_js

On 2026-10-30 a "\\n" in app.js became a real line break inside a "..."
string. One syntax error in a module script stops the whole script, so the
page showed its title and nothing else -- no sign-in, no lineup -- and every
Python test passed. `node --check app.js` did not catch it either: the
scripts are ES modules, and node only parses them as modules under a .mjs
name. So each is copied to .mjs first.

Skipped where node is not installed (the CI image has no node).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "webapp" / "static"
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "node is not installed")
class StaticJsTest(unittest.TestCase):
    def test_every_page_script_parses_as_a_module(self):
        scripts = sorted(STATIC.glob("*.js"))
        self.assertTrue(scripts)
        with tempfile.TemporaryDirectory() as tmp:
            for js in scripts:
                with self.subTest(script=js.name):
                    copy = Path(tmp) / f"{js.stem}.mjs"
                    shutil.copyfile(js, copy)
                    r = subprocess.run([NODE, "--check", str(copy)],
                                       capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
