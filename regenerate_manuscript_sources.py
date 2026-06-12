"""Regenerate all manuscript-source reports, figures, and READMEs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript_regenerate.run_all import main

if __name__ == "__main__":
    main()
