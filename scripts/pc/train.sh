#!/usr/bin/env bash
# Starts `python main.py <args>` on the PC as a detached job (see job.sh). Arguments are main.py's key=value overrides.
#   scripts/pc/train.sh train.validation=false model.rnn.epochs=3
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
JOB_NAME=train "$(dirname "${BASH_SOURCE[0]}")/job.sh" "uv run python main.py $(printf '%q ' "$@")"
