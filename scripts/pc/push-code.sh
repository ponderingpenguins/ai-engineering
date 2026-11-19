#!/usr/bin/env bash
# Copies the code to the PC. Leaves out .git, notebooks, docs, results, virtualenvs and the big data folders
# (use push-data.sh for the baked dataset). Extra arguments go to rsync, e.g. -n for a dry run.
#   scripts/pc/push-code.sh [-n]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_pc
pc_bash <<< "mkdir -p '$PC_DIR'"
# The recorder package is a local dependency of pyproject.toml, so it must go along.
pc_rsync -az --itemize-changes "$@" \
  --include='data/raw/floor_posture_dataset/recorder/***' \
  --exclude='/.git' --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
  --exclude='/results' --exclude='/notebooks' --exclude='/logs' --exclude='/docs' \
  --exclude='/data/raw/floor_posture_dataset/*' \
  --exclude='/data/processed/floor_posture*' \
  "$REPO/" "$PC_HOST:$PC_DIR/"
case " $* " in *" -n "*|*--dry-run*) echo "dry run: nothing was copied";; *) echo "code synced to $PC_HOST:$PC_DIR";; esac
