#!/usr/bin/env bash
# Starts `python main.py <mode> <args>` on the PC as a detached job (see job.sh). Pass cv, tune or train-final, then key=value overrides.
#   scripts/pc/train.sh cv training.validation=false training.epochs=3
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
JOB_NAME=train "$(dirname "${BASH_SOURCE[0]}")/job.sh" "uv run python main.py $(printf '%q ' "$@")"
