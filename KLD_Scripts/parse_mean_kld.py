#!/usr/bin/env python3
"""Extract a single mean KLD scalar from score_mode_kld.py stdout/stderr (best-effort)."""
from __future__ import annotations

import re
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("", end="")
        return 2
    path = sys.argv[1]
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        print("", end="")
        return 1

    # vLLM score_mode_kld.py summary (see examples/offline_inference/score_mode_kld.py):
    #   Results:
    #     Mean KLD: 0.013013
    #     Total positions: 204700
    #     ...
    num = r"([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)"
    patterns = [
        # Exact label first (avoids confusing Mean KLD with Positions/second, etc.)
        rf"(?m)^\s*Mean\s+KLD\s*:\s*{num}\s*$",
        rf"(?m)Mean\s+KLD\s*:\s*{num}",
        r"mean\s*kld\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)",
        r"mean\s*kl\s*(?:divergence)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)",
        r"kld\s*\(mean\)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)",
        r"average\s*kld\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)",
        r"Mean\s*KL\s*[Dd]ivergence\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            print(m.group(1), end="")
            return 0

    print("", end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
