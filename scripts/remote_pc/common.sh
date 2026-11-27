# Shared SSH and rsync helpers for the Tailscale-connected Windows/WSL training PC.
# Source this file from another Bash script; do not run it directly.
# Override PC_HOST or PC_DIR in the environment to select another host or checkout.
# Per-call SSH options bypass the configured RemoteCommand without editing ~/.ssh/config.

PC_HOST="${PC_HOST:-haukurs_pc}"
PC_DIR="${PC_DIR:-/home/haukur/rug/ai-engineering}"   # Absolute path inside WSL.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SSH_OPTS=(-o RemoteCommand=none -o RequestTTY=no -o BatchMode=yes -o ConnectTimeout=15)

die() { echo "error: $*" >&2; exit 1; }

# Fail early if Tailscale is stopped or the remote PC does not answer.
require_pc() {
  if command -v tailscale >/dev/null 2>&1 && tailscale status 2>&1 | grep -qi "stopped"; then
    die "Tailscale is stopped on this machine. Start it (tailscale up, or the app) and retry."
  fi
  # Prevent SSH from consuming input intended for this script, such as a stop.sh answer.
  ssh "${SSH_OPTS[@]}" "$PC_HOST" wsl.exe --cd '~' true </dev/null >/dev/null 2>&1 \
    || die "cannot reach $PC_HOST. Is the PC on and awake, and Tailscale running on both ends?"
}

# Run stdin as a Bash script inside WSL, starting in the remote home directory.
pc_bash() { ssh "${SSH_OPTS[@]}" "$PC_HOST" wsl.exe --cd '~' bash -s; }

# Sync files using remote WSL rsync; remote paths must be absolute, e.g. $PC_HOST:$PC_DIR/.
pc_rsync() { rsync -e "ssh ${SSH_OPTS[*]}" --rsync-path="wsl.exe rsync" "$@"; }

# Add the usual user-installed uv location to the remote PATH when needed.
REMOTE_PRELUDE='export PATH="$HOME/.local/bin:$PATH"'
