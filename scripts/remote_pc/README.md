# Remote GPU Scripts (`scripts/remote_pc/`)

Bash wrappers for syncing code and baked data to `haukurs_pc` (Windows host running WSL2 with an RTX 3070 over Tailscale), launching background training jobs, monitoring logs, and retrieving results.



## Environment Variables

| Variable     | Default                             | Purpose                                                                    |
| ------------ | ----------------------------------- | -------------------------------------------------------------------------- |
| `PC_HOST`  | `haukurs_pc`                      | Tailscale SSH host alias.                                                  |
| `PC_DIR`   | `/home/haukur/rug/ai-engineering` | Absolute checkout path inside WSL.                                         |
| `JOB_NAME` | `job` (`train` in `train.sh`) | Log and systemd unit prefix for `job.sh` (alphanumeric and dashes only). |



## Scripts

| Script               | Usage                                                | Responsibility                                                                                                                                                                                                                                                         |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `common.sh`        | *(sourced by all scripts)*                         | Defines `PC_HOST`, `PC_DIR`, `SSH_OPTS`, `require_pc`, `pc_bash`, `pc_rsync`, `die`, and `REMOTE_PRELUDE` (adds `~/.local/bin` to `PATH` for `uv`).                                                                                              |
| `status.sh`        | `scripts/remote_pc/status.sh`                      | Reports reachability, GPU utilisation/memory/temperature, disk/RAM, active `pcjob-*` units, latest log and result folders, baked dataset size, and PyTorch CUDA status.                                                                                              |
| `setup.sh`         | `scripts/remote_pc/setup.sh`                       | One-time remote setup: creates `$PC_DIR`, installs `uv` into user space (no `sudo`), runs `uv sync`, and checks CUDA availability. Run `push-code.sh` first.                                                                                                 |
| `push-code.sh`     | `scripts/remote_pc/push-code.sh [-n]`              | Rsyncs repository code and `data/raw/floor_posture_dataset/recorder/` (required editable dependency); excludes `.git`, `.venv`, `__pycache__`, `results`, `notebooks`, `logs`, `docs`, raw recordings, and `data/baked/`. Pass `-n` for a dry run. |
| `push-data.sh`     | `scripts/remote_pc/push-data.sh [name]`            | Resumable rsync (`--partial`) of `data/baked/<name>` (default `floor_posture`).                                                                                                                                                                                  |
| `run.sh`           | `scripts/remote_pc/run.sh <cmd...>`                | Runs one foreground command in `$PC_DIR` and streams output (for example `run.sh uv run pytest -q` or `run.sh nvidia-smi`).                                                                                                                                      |
| `train.sh`         | `scripts/remote_pc/train.sh <mode> [overrides...]` | Convenience wrapper that submits `uv run python main.py <mode> ...` through `job.sh` with `JOB_NAME=train`.                                                                                                                                                      |
| `job.sh`           | `scripts/remote_pc/job.sh '<command>'`             | Runs one quoted shell command as a background `systemd-run --user` unit (`pcjob-<name>-<timestamp>`), writing the command script and combined output to `logs/<name>_<timestamp>.{sh,log}`.                                                                      |
| `logs.sh`          | `scripts/remote_pc/logs.sh [-f] [file.log]`        | Prints the last 40 timestamped/error lines of the newest (or specified) log with progress-bar carriage returns stripped; `-f` follows live output.                                                                                                                   |
| `stop.sh`          | `scripts/remote_pc/stop.sh`                        | Lists running `pcjob-*` units and `main.py` processes and stops them after confirmation.                                                                                                                                                                           |
| `fetch-results.sh` | `scripts/remote_pc/fetch-results.sh [-n]`          | Copies remote `results/` into local `./results/` without deleting local files. Pass `-n` for a dry run.                                                                                                                                                          |



## Operational Walkthrough

### SSH and WSL Behaviour

The remote Windows host's SSH configuration drops interactive sessions directly into WSL (`RemoteCommand wsl.exe --cd ~`), which prevents `ssh host '<cmd>'` from running cleanly. `common.sh` avoids modifying your local `~/.ssh/config` by passing `-o RemoteCommand=none -o RequestTTY=no -o BatchMode=yes` on every call and piping commands into `wsl.exe --cd '~' bash -s` (and passing `--rsync-path="wsl.exe rsync"` for transfers).

WSL shuts down the Linux instance roughly 15 seconds after the last SSH connection closes, terminating background processes, `tmux` sessions, and user systemd units. To keep a training job alive, `job.sh` holds one detached local SSH session open (`systemd-run --user --wait`) for the exact lifetime of the job and exits automatically when the job completes. Keep the local machine awake and Tailscale connected until the job finishes (or configure `instanceIdleTimeout` in Windows `.wslconfig` on the remote PC).

### Standard Routines

```bash
# One-time provisioning
scripts/remote_pc/push-code.sh
scripts/remote_pc/setup.sh
scripts/remote_pc/push-data.sh
scripts/remote_pc/status.sh

# Typical training or tuning run
scripts/remote_pc/push-code.sh
scripts/remote_pc/train.sh tune 'tune.grid=[{model.arch.hidden: 64}, {model.arch.hidden: 128, training.lr: 0.0003}]'
scripts/remote_pc/logs.sh -f
scripts/remote_pc/fetch-results.sh
```

```bash
# Run a multi-seed sweep with a custom log prefix
JOB_NAME=sweep scripts/remote_pc/job.sh 'for s in 0 1 2; do uv run python main.py cv seed=$s; done'
```

Extending

1. **Adding a remote script:** Start with `#!/usr/bin/env bash`, `set -euo pipefail`, and `source "$(dirname "${BASH_SOURCE[0]}")/common.sh"`. Call `require_pc` before executing remote commands via `pc_bash` (include `$REMOTE_PRELUDE` when invoking `uv`) or syncing files via `pc_rsync`.
2. **Changing synced paths:** Edit the `--include` and `--exclude` rules in `push-code.sh` and verify with `scripts/remote_pc/push-code.sh -n`. Keep `--include='data/raw/floor_posture_dataset/recorder/***'` intact so `uv sync` can resolve the local editable `vt-radar-recorder` dependency.
