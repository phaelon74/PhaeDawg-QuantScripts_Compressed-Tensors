#!/usr/bin/env bash
# Create (or refresh) the phaelon74/exllamav3 sm86-decode fork checkout as a
# submodule of this workstream. Requires GitHub access to phaelon74.
#
#   export EXL3=...
#   bash "$EXL3/scripts/fork_exllamav3.sh"
set -euo pipefail
EXL3="${EXL3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$EXL3/../.." && pwd)"
PIN="${EXLLAMAV3_COMMIT:-0c49587a7c235e6303a6bbedc8b665272ad3a2ea}"
FORK_URL="${EXLLAMAV3_FORK_URL:-https://github.com/phaelon74/exllamav3.git}"
UPSTREAM_URL="https://github.com/turboderp-org/exllamav3.git"
BRANCH="${EXLLAMAV3_FORK_BRANCH:-sm86-decode}"
DEST="$EXL3/exllamav3"

cd "$REPO_ROOT"
if [[ ! -d "$DEST/.git" && ! -f "$DEST/.git" ]]; then
  if git ls-remote "$FORK_URL" HEAD >/dev/null 2>&1; then
    git submodule add -b "$BRANCH" "$FORK_URL" \
      "Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM/exllamav3" || \
      git clone -b "$BRANCH" "$FORK_URL" "$DEST"
  else
    echo "Fork $FORK_URL not reachable. Cloning upstream pin and applying overlay."
    git clone "$UPSTREAM_URL" "$DEST"
    git -C "$DEST" checkout "$PIN"
    git -C "$DEST" checkout -B "$BRANCH"
    python3 "$EXL3/kernel/overlay/apply_overlay.py" "$DEST"
    echo "Push this tree to $FORK_URL when the GitHub fork exists:"
    echo "  git -C \"$DEST\" remote add fork $FORK_URL"
    echo "  git -C \"$DEST\" push -u fork $BRANCH"
    exit 0
  fi
fi

git -C "$DEST" fetch --all --tags || true
python3 "$EXL3/kernel/overlay/apply_overlay.py" "$DEST"
echo "Submodule/checkout ready at $DEST"
echo "Rebuild with: bash \"$EXL3/scripts/build_exllamav3_ext.sh\""
