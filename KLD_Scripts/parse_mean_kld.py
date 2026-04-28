#!/usr/bin/env python3
"""Extract a single mean KLD scalar from score_mode_kld.py stdout/stderr (best-effort)."""
from __future__ import annotations

import re
import sys


def _mean_kld_from_results_block(text: str) -> str | None:
    """Same idea as RunPod run_kld_benchmark.sh: grep -A4 '^Results:' then pick Mean KLD."""
    lines = text.splitlines()
    num = r"([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)"
    for i, raw in enumerate(lines):
        if raw.strip().startswith("Results:"):
            chunk = "\n".join(lines[i : min(i + 12, len(lines))])
            m = re.search(rf"Mean\s+KLD\s*:\s*{num}", chunk, flags=re.IGNORECASE)
            if m:
                return m.group(1)
    return None


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

    # TTY-style color codes break line-anchored regexes when stderr is merged into the log.
    text = re.sub(r"\x1b\[[0-9;:]*[ -/]*[@-~]", "", text)

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
        # After ANSI strip, still allow junk before the label on the same line
        rf"Mean\s+KLD\s*:\s*{num}",
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

    from_block = _mean_kld_from_results_block(text)
    if from_block:
        print(from_block, end="")
        return 0

    print("", end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
