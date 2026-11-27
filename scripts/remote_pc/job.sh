#!/usr/bin/env bash
# Start one quoted command as a remote background job; JOB_NAME defaults to job.
# Usage: JOB_NAME=sweep scripts/remote_pc/job.sh 'uv run python main.py cv seed=1'
# Save command and output as logs/<name>_<timestamp>.sh and .log on the PC.
# Keep this computer awake and connected through Tailscale: a background SSH session
# keeps WSL alive for the job. Losing that connection can terminate the remote job.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ $# -eq 1 ] || die "usage: job.sh '<command line>'   (one quoted string)"
name="${JOB_NAME:-job}"
[[ "$name" =~ ^[A-Za-z0-9-]+$ ]] || die "JOB_NAME may only contain letters, digits and dashes"
require_pc

job_text="$1"
stamp="$(date +%Y%m%d-%H%M%S)"
base="logs/${name}_${stamp}"
unit="pcjob-${name}-${stamp}"
keeper_log="${TMPDIR:-/tmp}/${unit}.keeper.log"

# Keep SSH open until the job ends; this PC's WSL setup depends on a live session.
# A systemd unit lets status.sh inspect the job and stop.sh stop it as a group.
launch() {
  pc_bash <<REMOTE
cd "$PC_DIR" && mkdir -p logs
{ echo 'export PATH="\$HOME/.local/bin:\$PATH"'; echo "cd '$PC_DIR'"; } > "$base.sh"
cat >> "$base.sh" <<'JOBTEXT'
$job_text
JOBTEXT
systemd-run --user --wait --collect --quiet --unit="$unit" \
  -p StandardOutput=append:"$PC_DIR/$base.log" -p StandardError=append:"$PC_DIR/$base.log" \
  bash "$PC_DIR/$base.sh"
echo "job \$? finished"
REMOTE
}
( launch </dev/null > "$keeper_log" 2>&1 ) &
disown

# Check startup briefly so common launch errors appear in the calling terminal.
for _ in 1 2 3 4 5 6 7 8; do
  sleep 1
  grep -qiE "error|failed|denied" "$keeper_log" 2>/dev/null && { cat "$keeper_log" >&2; die "the job did not start"; }
  pc_bash <<< "systemctl --user is-active --quiet '$unit'" 2>/dev/null && break || true
done
echo "started $unit -> $PC_DIR/$base.log"
echo "follow it: scripts/remote_pc/logs.sh -f     stop it: scripts/remote_pc/stop.sh     (keep this Mac awake and Tailscale on until it ends)"
