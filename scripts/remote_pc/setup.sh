#!/usr/bin/env bash
# Prepare the remote environment: install uv if needed, then run uv sync.
# Usage: scripts/remote_pc/setup.sh
# Run push-code.sh first to copy pyproject.toml, uv.lock and the recorder dependency.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_pc
pc_bash <<REMOTE
set -e
$REMOTE_PRELUDE
mkdir -p "$PC_DIR"
command -v uv >/dev/null || /usr/bin/pip3 install --user uv
uv --version
cd "$PC_DIR" && uv sync
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
REMOTE
