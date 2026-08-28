from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_SRC = Path(__file__).resolve().parents[1] / "plugin" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))
