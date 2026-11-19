# AI Engineering Practical - Valtes

Radar-based floor posture recognition. The default task: does a 3.2 s window of radar frames contain lying? Models are evaluated person by person, because a person must never be in both training and test data.

## Setup

```
uv sync
source .venv/bin/activate
git lfs install && git lfs pull        # install git-lfs first: apt install git-lfs / brew install git-lfs
uv pip install -e ./data/raw/floor_posture_dataset/recorder
```

For the notebooks: `uv pip install -e ".[jupyter]"` and `python -m ipykernel install --user --name floor-posture`.

Type check: `uv run mypy main.py --ignore-missing-imports` (the flag is needed because `vt_radar` has no type stubs). Tests: `uv run pytest`.

## Data

`python -m data.extract` reads every recording once and writes one Hugging Face dataset (one row per frame) to `data/processed/floor_posture` (gitignored, about 4.4 GB, about 20 s with `num_proc=6`). Run it before anything else. Columns, the hand-made `person` and `person_present` labels and the pitfalls (centring the heatmap, empty heatmaps) are in [docs/data_notes/baked-dataset.md](docs/data_notes/baked-dataset.md). The baked data contains identifiable camera feeds.

## Running

```
python main.py [key=value ...]
```

Defaults are in `config/config.yaml`. Command-line `key=value` pairs override them (OmegaConf syntax), and unknown keys are rejected.

| Key | Default | Meaning |
| --- | --- | --- |
| `model.name` | `rnn` | `rnn` (CNN + GRU on heatmaps), `pointnet` (on radar points) or `baseline` (most common label). Hyperparameters are under `model.<name>.*`. |
| `data.baked` | `data/processed/floor_posture` | Baked dataset folder. |
| `data.preprocess` | `[config_4]` | Extra frame filters. Recordings without a known person are always dropped. |
| `data.exclude` | two recordings | Recordings left out of training and testing. The default drops `2311__20231124_152926` and `2311__20231124_153817`, where the camera does not look top down. |
| `eval.folds` | `null` | `null`: one fold per person. An integer `k`: stratified k-fold over people. |
| `eval.window_len` / `eval.stride` | `16` / `8` | Frames per window and step. A frame is about 0.2 s. |
| `eval.label_rule` | `any_lying` | `any_lying`: lying occurred in the window. `last`: label of the last frame. |
| `eval.max_windows_per_person` | `null` | Keep at most this many windows per person, for quick test runs only. |
| `train.validation` | `true` | Hold out one training person for early stopping. `false`: fixed epochs. |
| `train.oversample` | `null` | Repeat rare-class training windows up to this share of the largest class. |
| `tune.grid` | `[]` | Hyperparameter settings to tune over (see below). |
| `train.full_run` | `false` | Train one final model on everyone with the tuned setting. |

```
python main.py model.name=baseline eval.folds=2
python main.py model.name=pointnet model.pointnet.epochs=5
```

Each run writes `results/<timestamp>_<model>/` with `report.md` (settings, pooled, per-fold and per-person tables), the loss figures, `results.json` and `checkpoints/`. Rebuild a report with `python -m utils.run_report results/<run>`.

## Evaluation workflow

There are four known people, so evaluation is nested leave-one-person-out. Tune first, then do the full run. The full run reads the setting that tuning saved.

```
# 1. Tuning: writes results/tuned.yaml and a report with every setting's scores.
python main.py 'tune.grid=[{hidden: 64}, {hidden: 128, lr: 0.0003}]'

# 2. Full run: trains one model on all people. Writes results/<run>_final/final.pt.
python main.py train.full_run=true
```

1. The outer loop holds out one person for testing.
2. The inner loop runs leave-one-person-out on the other people for every setting in `tune.grid`. The setting with the lowest mean class-balanced validation loss wins. The validation classes are weighted the same way for every person, so the losses are comparable and the winner is the best setting overall, not the best for the easiest person.
3. The winner is refit on all the training people for its median best epoch, and scored on the test person.
4. The setting that won the most outer folds goes to `tune.best_path` together with its model and task. The full run stops with an error if they do not match the current run.

The test person never chooses a setting. Write a space after each colon in the grid (`hidden: 64`). Tuning costs (people - 1) trainings per setting and per outer fold, so run it on the PC. More detail, and the caveats of tuning with four people, are in [docs/modeling.md](docs/modeling.md).

## Running on the PC

Training runs on `haukurs_pc` (Windows, RTX 3070, over Tailscale, inside WSL). The scripts in `scripts/pc/` wrap the SSH details. They need Tailscale on this machine and the PC switched on.

| Script | What it does |
| --- | --- |
| `status.sh` | Reachability, GPU, disk, running jobs, newest log, and whether the dataset and torch are ready. |
| `setup.sh` | One time: installs `uv` and runs `uv sync` on the PC (run `push-code.sh` first). |
| `push-code.sh [-n]` | Copies the code to the PC. `-n` is a dry run. |
| `push-data.sh [folder]` | Copies the baked dataset (about 3 GB, resumable). |
| `train.sh <main.py args>` | Starts `python main.py <args>` as a background job. |
| `job.sh '<command>'` | Starts any command as a background job. `JOB_NAME=name` names the log. |
| `logs.sh [-f]` | Shows the newest log (`-f` follows it). |
| `run.sh <command>` | Runs one command on the PC and streams the output. |
| `stop.sh` | Stops the background jobs (asks first). |
| `fetch-results.sh [-n]` | Copies result folders back to `./results`. Never deletes locally. |

Typical run: `push-code.sh`, `train.sh ...`, `logs.sh -f`, `fetch-results.sh`.

WSL stops every process about 15 seconds after the last SSH session closes. So `job.sh` keeps one SSH session open on this machine for as long as the job runs. **Keep this machine awake and connected until the job ends**, or the job dies. `PC_HOST` and `PC_DIR` can be set in the environment.

## Code layout

```
config/config.yaml        default settings
data/                     raw dataset (git-lfs), baked dataset, person labels
docs/                     notes (modeling.md, data_notes/), report, slides
notebooks/                exploration and annotation notebooks
scripts/pc/               helpers for running on the PC
results/                  one folder per run (gitignored)
src/data/                 extract, preprocess, windows, augment, resample
src/models/               RNN, PointNet, baseline, shared training loop, factory
src/utils/                experiment, trainer, tuning, fitting, results, report, task, config
tests/
```

- `main.py` reads the command line and turns errors into exit codes. `utils/experiment.py` checks the config and runs one experiment.
- `utils/task.py` defines the task (target, window length, label rule, `max_points`). Checkpoints and `tuned.yaml` store it and compare it.
- `utils/fitting.py` fits one model on one fold, for both the outer and the tuning loop.
- `utils/results.py` holds the result records. The report and the loss curves read them. `results.json` is only read and written there.
- A model class declares its hyperparameter names (`parameter_names`) and its input type, and `models/factory.py` builds any model from the config.
