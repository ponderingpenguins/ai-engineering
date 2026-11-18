#!/usr/bin/env bash
# Copies the baked dataset (about 3 GB) to the PC. Resumable: rerun it if it is interrupted.
# Pass a different folder name to copy another one, e.g. floor_posture_person_split.
#   scripts/pc/push-data.sh [dataset-folder-name]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
name="${1:-floor_posture}"
[ -d "$REPO/data/processed/$name" ] || die "no folder data/processed/$name here"
require_pc
pc_bash <<< "mkdir -p '$PC_DIR/data/processed/$name'"
pc_rsync -a --partial --info=progress2 \
  "$REPO/data/processed/$name/" "$PC_HOST:$PC_DIR/data/processed/$name/"
echo "data/processed/$name synced"
