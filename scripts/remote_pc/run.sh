#!/usr/bin/env bash
# Run a command in the remote checkout and stream its output to this terminal.
# Usage: scripts/remote_pc/run.sh uv run pytest -q
# For background work, use job.sh or train.sh; their keep-alive SSH connection must stay up.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ $# -gt 0 ] || die "usage: run.sh <command> [args...]"
require_pc
cmd="$(printf '%q ' "$@")"
pc_bash <<REMOTE
$REMOTE_PRELUDE
cd "$PC_DIR" && $cmd
REMOTE
