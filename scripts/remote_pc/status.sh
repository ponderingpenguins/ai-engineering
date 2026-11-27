#!/usr/bin/env bash
# Check connectivity, GPU and disk use, jobs, recent outputs, dataset and environment.
# Usage: scripts/remote_pc/status.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_pc
pc_bash <<REMOTE
$REMOTE_PRELUDE
echo "host: \$(hostname)   up: \$(uptime -p)"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "gpu: nvidia-smi not available"
echo "disk: \$(df -h ~ | awk 'NR==2{print \$4" free of "\$2" ("\$5" used)"}')   memory: \$(free -g | awk '/Mem/{print \$7"G available of "\$2"G"}')"
cd "$PC_DIR" 2>/dev/null || { echo "no checkout at $PC_DIR yet: run scripts/remote_pc/push-code.sh"; exit 0; }
echo "--- running jobs"
systemctl --user list-units 'pcjob-*' --all --no-legend --plain 2>/dev/null | awk '{print \$1, \$3, \$4}' | grep . || echo "none"
echo "--- newest log"
log=\$(ls -t logs/*.log 2>/dev/null | head -1)
if [ -n "\$log" ]; then echo "\$log"; tr '\r' '\n' < "\$log" | grep -E '^\[' | tail -3 | cut -c1-160; else echo "none"; fi
echo "--- newest results"
ls -dt results/*/ 2>/dev/null | head -3 || true
echo "--- data / venv"
[ -d data/baked/floor_posture ] && echo "baked dataset: \$(du -sh data/baked/floor_posture | cut -f1)" || echo "baked dataset: missing (scripts/remote_pc/push-data.sh)"
[ -x .venv/bin/python ] && .venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -1 || echo "venv: not ready (scripts/remote_pc/setup.sh)"
REMOTE
