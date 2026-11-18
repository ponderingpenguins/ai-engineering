#!/usr/bin/env bash
# Runs one command on the PC in the project folder and streams its output. Stops when the command ends
# or when you press Ctrl-C. For long jobs use job.sh or train.sh, which keep running if you disconnect.
#   scripts/pc/run.sh uv run pytest -q
#   scripts/pc/run.sh nvidia-smi
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ $# -gt 0 ] || die "usage: run.sh <command> [args...]"
require_pc
cmd="$(printf '%q ' "$@")"
pc_bash <<REMOTE
$REMOTE_PRELUDE
cd "$PC_DIR" && $cmd
REMOTE
