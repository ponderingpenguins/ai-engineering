#!/usr/bin/env bash
# Copy remote reports, figures, checkpoints and result files into local results/.
# Usage: scripts/remote_pc/fetch-results.sh [-n]
# Extra arguments go to rsync. Existing matching files may be updated; none are deleted.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_pc
pc_bash <<< "[ -d '$PC_DIR/results' ] && [ -n \"\$(ls -A '$PC_DIR/results')\" ]" || { echo "no results on the PC yet"; exit 0; }
mkdir -p "$REPO/results"
pc_rsync -a --itemize-changes --info=stats1 "$@" "$PC_HOST:$PC_DIR/results/" "$REPO/results/" \
  | grep -vE '^(Number|Total|File list|sent|total)' || true
echo "results fetched into $REPO/results"
