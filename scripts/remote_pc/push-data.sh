#!/usr/bin/env bash
# Copy one dataset from local data/baked/ to the remote PC.
# Usage: scripts/remote_pc/push-data.sh [dataset-folder-name]
# Default: floor_posture. Re-run after an interruption to resume the transfer.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
name="${1:-floor_posture}"
[ -d "$REPO/data/baked/$name" ] || die "no folder data/baked/$name here"
require_pc
pc_bash <<< "mkdir -p '$PC_DIR/data/baked/$name'"
pc_rsync -a --partial --info=progress2 \
  "$REPO/data/baked/$name/" "$PC_HOST:$PC_DIR/data/baked/$name/"
echo "data/baked/$name synced"
