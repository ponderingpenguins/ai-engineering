#!/usr/bin/env bash
# Starts a shell command line on the PC as a background job and returns immediately. Output goes to
# logs/<name>_<timestamp>.log in the project folder on the PC (follow it with logs.sh).
#
# How it stays alive: WSL shuts the whole Linux instance down about 15 seconds after the last ssh session
# closes, and that kills every process, including nohup/setsid/tmux jobs and systemd services. So this
# script leaves one ssh session open in the background on THIS machine for exactly as long as the job runs
# (it ends by itself when the job ends). Consequences: this machine must stay awake and connected (Tailscale
# on) until the job finishes; the script does not stop the Mac from sleeping. If the connection drops, the job dies about 15 seconds later.
# A permanent fix would be on the PC (Windows side), e.g. an `instanceIdleTimeout` in .wslconfig.
#
# The command text is saved as logs/<name>_<timestamp>.sh on the PC. Quote the whole command as one argument.
#   scripts/pc/job.sh 'for s in 0 1 2; do uv run python main.py cv seed=$s; done'
#   JOB_NAME=sweep scripts/pc/job.sh '...'      (letters, digits and dashes only)
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

# `systemd-run --wait` stays in the foreground until the job ends, which is what keeps this ssh session open.
# The unit (rather than a plain child process) gives us status.sh / stop.sh and a clean stop.
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

# Give it a moment to start, so a failure shows up here instead of silently.
for _ in 1 2 3 4 5 6 7 8; do
  sleep 1
  grep -qiE "error|failed|denied" "$keeper_log" 2>/dev/null && { cat "$keeper_log" >&2; die "the job did not start"; }
  pc_bash <<< "systemctl --user is-active --quiet '$unit'" 2>/dev/null && break || true
done
echo "started $unit -> $PC_DIR/$base.log"
echo "follow it: scripts/pc/logs.sh -f     stop it: scripts/pc/stop.sh     (keep this Mac awake and Tailscale on until it ends)"
