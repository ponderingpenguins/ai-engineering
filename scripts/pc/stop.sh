#!/usr/bin/env bash
# Stops the background jobs started with job.sh / train.sh (and any stray main.py / utils.compare). Asks first.
#   scripts/pc/stop.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_pc
running="$(pc_bash <<< "systemctl --user list-units 'pcjob-*' --no-legend --plain 2>/dev/null | awk '{print \$1}'; pgrep -af 'main\.py|utils\.compare' | grep -v -e pgrep -e 'bash -s' || true")"
[ -n "$running" ] || { echo "nothing running"; exit 0; }
echo "$running"
read -r -p "Stop these? [y/N] " answer || answer=n
[ "$answer" = "y" ] || { echo "left running"; exit 0; }
pc_bash <<< "systemctl --user stop 'pcjob-*' 2>/dev/null; pkill -f 'main\.py|utils\.compare' || true; echo stopped"
