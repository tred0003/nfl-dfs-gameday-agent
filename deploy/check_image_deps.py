"""Does the training image actually contain everything train.py imports?

A Dockerfile's COPY line is a dependency list that nothing keeps in sync. On
2026-09-16 `train.py` gained `import calibrate` and the COPY line did not; the
next Cloud Build would have produced a perfectly good image that died on its
first line with ModuleNotFoundError -- after a five-minute build and a job
submission, with the error buried in Cloud Logging.

This catches it in under a second, locally, before the build.

Run:  python deploy/check_image_deps.py
Exit: 0 if the image is complete, 1 if a module is missing.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DOCKERFILE = ROOT / "deploy" / "Dockerfile.train"
ENTRYPOINT = "train.py"


def local_imports(start: Path) -> set[str]:
    """Every project module reachable from `start`, followed transitively.

    Transitive matters: train.py imports calibrate, and if calibrate ever
    imports a third module the image needs that too.
    """
    seen: set[str] = set()
    queue = [start]
    while queue:
        f = queue.pop()
        if not f.exists():
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in seen or not (SRC / f"{n}.py").exists():
                    continue
                seen.add(n)
                queue.append(SRC / f"{n}.py")
    return seen


def copied_modules() -> set[str]:
    text = DOCKERFILE.read_text(encoding="utf-8")
    copied = set()
    for line in text.splitlines():
        if line.strip().startswith("COPY") and "src/" in line:
            copied |= {Path(m).stem for m in re.findall(r"src/([\w./]+\.py)", line)}
    return copied


def main() -> int:
    needed = local_imports(SRC / ENTRYPOINT) | {Path(ENTRYPOINT).stem}
    have = copied_modules()
    missing = sorted(needed - have)
    extra = sorted(have - needed)

    print(f"entrypoint : src/{ENTRYPOINT}")
    print(f"needs      : {', '.join(sorted(needed))}")
    print(f"image has  : {', '.join(sorted(have))}")

    if missing:
        print(f"\nMISSING from {DOCKERFILE.relative_to(ROOT)}: {', '.join(missing)}")
        print("The build will succeed and the job will die on import. Add them to "
              "the COPY line.")
        return 1
    if extra:
        # not an error -- just image weight on a container that should be small
        print(f"\nnote: copied but not imported: {', '.join(extra)}")
    print("\nimage dependency list is complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
