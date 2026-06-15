#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent / "runtime"
RUNTIME_ENTRYPOINT = RUNTIME_ROOT / "src" / "inference" / "infer_observer_scores.py"


def main() -> None:
    if not RUNTIME_ENTRYPOINT.exists():
        raise FileNotFoundError(f"Final observer runtime entrypoint not found: {RUNTIME_ENTRYPOINT}")

    # Make the bundled runtime's `src/...` imports resolve before the project's own `src`.
    sys.path.insert(0, str(RUNTIME_ROOT))
    runpy.run_path(str(RUNTIME_ENTRYPOINT), run_name="__main__")


if __name__ == "__main__":
    main()

