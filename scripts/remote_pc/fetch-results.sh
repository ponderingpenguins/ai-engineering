#!/usr/bin/env bash
# Copies result folders (report, figures, checkpoints, results.json) from the PC into ./results here.
# Never deletes anything locally. Extra arguments go to rsync, e.g. -n for a dry run.
#   scripts/remote_pc/fetch-results.sh [-n]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_pc
pc_bash <<< "[ -d '$PC_DIR/results' ] && [ -n \"\$(ls -A '$PC_DIR/results')\" ]" || { echo "no results on the PC yet"; exit 0; }
mkdir -p "$REPO/results"
pc_rsync -a --itemize-changes --info=stats1 "$@" "$PC_HOST:$PC_DIR/results/" "$REPO/results/" \
  | grep -vE '^(Number|Total|File list|sent|total)' || true
echo "results fetched into $REPO/results"
