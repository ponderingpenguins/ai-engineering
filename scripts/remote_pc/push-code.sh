#!/usr/bin/env bash
# Copy the checkout and local recorder dependency to the remote PC.
# Usage: scripts/remote_pc/push-code.sh [-n]
# Exclude Git metadata, environments, notebooks, docs, outputs and raw recordings.
# Use push-data.sh for baked data; extra arguments go to rsync (-n previews changes).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_pc
pc_bash <<< "mkdir -p '$PC_DIR'"
# Include the local recorder dependency even though the rest of the raw-data tree is excluded.
pc_rsync -az --itemize-changes "$@" \
  --include='data/raw/floor_posture_dataset/recorder/***' \
  --exclude='/.git' --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
  --exclude='/results' --exclude='/notebooks' --exclude='/logs' --exclude='/docs' \
  --exclude='/data/raw/floor_posture_dataset/*' \
  --exclude='/data/baked/' \
  "$REPO/" "$PC_HOST:$PC_DIR/"
case " $* " in *" -n "*|*--dry-run*) echo "dry run: nothing was copied";; *) echo "code synced to $PC_HOST:$PC_DIR";; esac
