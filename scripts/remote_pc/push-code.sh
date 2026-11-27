#!/usr/bin/env bash
# Copy the repository code to the remote PC.
# Usage: scripts/remote_pc/push-code.sh [-n]
# Exclude Git metadata, environments, notebooks, docs, outputs and raw recordings.
# Use push-data.sh for baked data; extra arguments go to rsync (-n previews changes).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_pc
pc_bash <<< "mkdir -p '$PC_DIR'"
pc_rsync -az --itemize-changes "$@" \
  --exclude='/.git' --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
  --exclude='/results' --exclude='/notebooks' --exclude='/logs' --exclude='/docs' \
  --exclude='/data/raw/floor_posture_dataset/*' \
  --exclude='/data/baked/' \
  --exclude='/recording/' \
  "$REPO/" "$PC_HOST:$PC_DIR/"
case " $* " in *" -n "*|*--dry-run*) echo "dry run: nothing was copied";; *) echo "code synced to $PC_HOST:$PC_DIR";; esac
