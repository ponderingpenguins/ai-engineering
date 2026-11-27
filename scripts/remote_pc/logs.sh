#!/usr/bin/env bash
# Show filtered log lines from the newest remote job, omitting progress-bar noise.
# Usage: scripts/remote_pc/logs.sh [-f] [train_20261118-153000.log]
# Pass a filename from remote logs/ to select an older log; -f follows until Ctrl+C.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
follow=0; [ "${1:-}" = "-f" ] && { follow=1; shift; }
want="${1:-}"
require_pc
pc_bash <<REMOTE
cd "$PC_DIR"
log="$want"; [ -n "\$log" ] && log="logs/\$log"
[ -n "\$log" ] || log=\$(ls -t logs/*.log 2>/dev/null | head -1)
[ -n "\$log" ] || { echo "no logs yet"; exit 0; }
echo "== \$log"
if [ "$follow" = 1 ]; then
  tail -n 40 -f "\$log" | tr '\r' '\n' | grep --line-buffered -E '^\[|Traceback|Error'
else
  tr '\r' '\n' < "\$log" | grep -E '^\[|Traceback|Error' | tail -40 | cut -c1-200
fi
REMOTE
