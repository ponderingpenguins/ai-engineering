#!/usr/bin/env bash
# Start cv, tune or train-final as a remote background job through job.sh.
# Usage: scripts/remote_pc/train.sh cv training.validation=false training.epochs=3
# Follow the run with logs.sh; keep this computer awake and connected until it finishes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
JOB_NAME=train "$(dirname "${BASH_SOURCE[0]}")/job.sh" "uv run python main.py $(printf '%q ' "$@")"
