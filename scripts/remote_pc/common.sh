# Shared settings and helpers for the scripts in this folder. Source it, do not run it.
#
# haukurs_pc is a Windows machine on Tailscale whose ssh config drops you straight into WSL
# (RemoteCommand wsl.exe --cd ~). That blocks `ssh host '<cmd>'`, so every call here overrides it
# per call (your ~/.ssh/config is not touched) and pipes the script into bash inside WSL.

PC_HOST="${PC_HOST:-haukurs_pc}"
PC_DIR="${PC_DIR:-/home/haukur/rug/ai-engineering}"   # absolute path inside WSL
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SSH_OPTS=(-o RemoteCommand=none -o RequestTTY=no -o BatchMode=yes -o ConnectTimeout=15)

die() { echo "error: $*" >&2; exit 1; }

# Fails early with a useful message if Tailscale is off or the PC does not answer.
require_pc() {
  if command -v tailscale >/dev/null 2>&1 && tailscale status 2>&1 | grep -qi "stopped"; then
    die "Tailscale is stopped on this machine. Start it (tailscale up, or the app) and retry."
  fi
  # </dev/null: ssh would otherwise read (and swallow) this script's stdin, e.g. an answer piped to stop.sh.
  ssh "${SSH_OPTS[@]}" "$PC_HOST" wsl.exe --cd '~' true </dev/null >/dev/null 2>&1 \
    || die "cannot reach $PC_HOST. Is the PC on and awake, and Tailscale running on both ends?"
}

# Runs the script on stdin in bash inside WSL on the PC, from the home directory.
pc_bash() { ssh "${SSH_OPTS[@]}" "$PC_HOST" wsl.exe --cd '~' bash -s; }

# rsync to or from the PC. Remote paths look like "$PC_HOST:$PC_DIR/...", and must be absolute.
pc_rsync() { rsync -e "ssh ${SSH_OPTS[*]}" --rsync-path="wsl.exe rsync" "$@"; }

# Lines every remote script starts with: uv lives in ~/.local/bin.
REMOTE_PRELUDE='export PATH="$HOME/.local/bin:$PATH"'
