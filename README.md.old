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

The repository uses 2 initial stages for raw data and annotations, then a final baked dataset for use in modeling tasks. `data/raw/` holds the raw recordings in Git LFS which were received from Valtes. `data/annotations/` holds the source annotation CSVs as normal Git text files. `data/baked/` holds generated datasets, gitignored to be created on individual machines. 

When running the bake, it automatically joins `person_map.csv`, `presence_events.csv`, and `lying_intervals.csv` from `data/annotations/`. Any annotation tools or labelling notebooks save their source CSVs there. After editing annotations, run `python main.py bake --refresh-annotations` to replace these columns in the existing dataset without parsing raw recordings again. Use `--overwrite` for a full rebuild.

`python main.py bake` reads every recording once and writes one Hugging Face dataset (one row per frame) to `data/baked/floor_posture` (gitignored, about 4.4 GB, about 20 s with `--num-proc 6`). Run it before anything else. Columns, the hand-made `person` and `person_present` labels and the pitfalls (centring the heatmap, empty heatmaps) are in [docs/data_notes/baked-dataset.md](docs/data_notes/baked-dataset.md). The baked data contains identifiable camera feeds.

## Running

```
python main.py cv [key=value ...]
```

Configuration is split by purpose:

- `config/data.yaml`: paths, recording selection, optional denoising and labelled windows. Bake and annotation tools load only this file.
- `config/train.yaml`: model selection, seed, output directory, shared training loop, class weighting/oversampling, cross-validation and tuning.
- `config/models/<name>.yaml`: only the selected model's architecture (`model.arch.*`) and training-only augmentation (`model.augment.*`). Wrong-model keys are rejected.

Typed settings and validation live in `src/config/` (`data.py`, `models.py`, `training.py`); `load_data_config` and `load_run_config` merge defaults, YAML and CLI overrides in that order. Unknown keys and invalid values are rejected. Unknown participants are always filtered by the built-in selection rule.

```
python main.py cv model=baseline cv.folds=2
python main.py cv model=pointnet training.epochs=5 model.augment.shift_m=0.3
python main.py tune 'tune.grid=[{model.arch.hidden: 64}, {model.arch.hidden: 128, training.lr: 0.0003}]'
python main.py train-final
python main.py bake --num-proc 4 --overwrite
python main.py bake --refresh-annotations
python main.py report results/<run>
python main.py annotate --port 8765 --no-browser
python main.py audit --out results/annotation_agreement.md
```

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | `rnn` | Model to run: `rnn`, `pointnet` or `baseline`. Architecture and augmentation settings are under `model.arch.*` and `model.augment.*`. |
| `paths.baked_dir` | `data/baked/floor_posture` | Baked dataset folder. Relative paths start at the repository root. |
| `selection.sensor_config` | `Config_4.cfg` | Sensor config to keep; `null` keeps all sensor configs. Window shape checks still apply. |
| `selection.exclude_recordings` | two recordings | Exclude `2311__20231124_152926` and `2311__20231124_153817`, whose cameras are not top-down. |
| `windowing.target` | `lying_annotation` | Per-frame label column; `label_pose` selects the original pose track. |
| `windowing.window_len` / `windowing.stride` | `16` / `8` | Frames per window and between window starts, about 3.2 s and 1.6 s. |
| `windowing.label_rule` | `any_lying` | `any_lying`: lying occurred in any frame. `last`: use the final frame's label. |
| `windowing.collapse_lying` | `true` | Merge `lying_*` labels into `lying`; other labelled poses become `not_lying`. |
| `windowing.max_points` | `128` | Strongest radar points retained per point-cloud window. |
| `training.validation` | `true` | Hold out one training person for early stopping; `false` trains for fixed epochs. |
| `training.class_weights` | `true` | Use inverse-frequency weights for the training loss. Validation loss remains class-balanced. |
| `training.oversample_ratio` | `null` | Repeat rare training windows up to this fraction of the largest class; `null` disables oversampling. |
| `cv.folds` | `null` | `null`: leave one person out. An integer selects grouped k-fold cross-validation. |
| `cv.max_windows_per_person` | `null` | Cap windows per person for quick checks; `null` keeps all windows. |
| `tune.grid` | `[]` | Candidate overrides for `tune`, using dotted keys such as `model.arch.hidden` and `training.lr`. |

The YAML files show every active default and disabled option.

Bake execution flags (`--num-proc`, `--overwrite`, repeated `--recording`) are not saved in YAML. `annotate` opens the annotation tool; `audit` compares reviewed annotations with the original pose labels. Their data overrides use the same keys in `config/data.yaml`. Each command has its own help: `python main.py <command> --help`. The module commands `python -m annotation_tool`, `python -m evaluation.annotation_audit` and `python -m evaluation.report` also remain available.

Each run writes `results/<timestamp>_<model>/` with `report.md` (settings, frame/window coverage, pooled, per-fold and per-person tables), the loss figures, `results.json` and `checkpoints/`. Rebuild a report with `python main.py report results/<run>`. Reports accept older config layouts within the current result-record format; the coverage section appears when those counts were recorded. The earliest October results use a different record schema and need conversion before they can be rebuilt.

## Evaluation workflow

There are four known people, so evaluation is nested leave-one-person-out. Tune first, then do the full run. The full run reads the setting that tuning saved.

```
# 1. Tuning: writes results/tuned.yaml and a report with every setting's scores.
python main.py tune 'tune.grid=[{model.arch.hidden: 64}, {model.arch.hidden: 128, training.lr: 0.0003}]'

# 2. Full run: trains one model on all people. Writes results/<run>_final/final.pt.
python main.py train-final
```

1. The outer loop holds out one person for testing.
2. The inner loop runs leave-one-person-out on the other people for every setting in `tune.grid`. The setting with the lowest mean class-balanced validation loss wins. The validation classes are weighted the same way for every person, so the losses are comparable and the winner is the best setting overall, not the best for the easiest person.
3. The winner is refit on all the training people for its median best epoch, and scored on the test person.
4. The setting that won the most outer folds goes to `tune.best_path` together with its model and windowing. The full run stops with an error if they do not match the current run.

The test person never chooses a setting. Write a space after each colon in the grid (`model.arch.hidden: 64`). Tuning costs (people - 1) trainings per setting and per outer fold, so run it on the PC. More detail, and the caveats of tuning with four people, are in [docs/modeling.md](docs/modeling.md).

## Running on the PC

Training runs on `haukurs_pc` (Windows, RTX 3070, over Tailscale, inside WSL). The scripts in `scripts/remote_pc/` wrap the SSH details. They need Tailscale on this machine and the PC switched on.

| Script | What it does |
| --- | --- |
| `status.sh` | Reachability, GPU, disk, running jobs, newest log, and whether the dataset and torch are ready. |
| `setup.sh` | One time: installs `uv` and runs `uv sync` on the PC (run `push-code.sh` first). |
| `push-code.sh [-n]` | Copies the code to the PC. `-n` is a dry run. |
| `push-data.sh [folder]` | Copies the baked dataset (about 3 GB, resumable). |
| `train.sh <cv\|tune\|train-final> <overrides>` | Starts `python main.py <mode> <args>` as a background job. |
| `job.sh '<command>'` | Starts any command as a background job. `JOB_NAME=name` names the log. |
| `logs.sh [-f]` | Shows the newest log (`-f` follows it). |
| `run.sh <command>` | Runs one command on the PC and streams the output. |
| `stop.sh` | Stops the background jobs (asks first). |
| `fetch-results.sh [-n]` | Copies result folders back to `./results`. Never deletes locally. |

Typical run: `push-code.sh`, `train.sh cv ...`, `logs.sh -f`, `fetch-results.sh`.

WSL stops every process about 15 seconds after the last SSH session closes. So `job.sh` keeps one SSH session open on this machine for as long as the job runs. **Keep this machine awake and connected until the job ends**, or the job dies. `PC_HOST` and `PC_DIR` can be set in the environment.

## Code layout

```
config/                   data.yaml, train.yaml, models/<name>.yaml
data/                     raw recordings (git-lfs), source annotations, generated baked datasets
docs/                     notes (modeling.md, data_notes/), report, slides
notebooks/                exploration/, labelling/, archive/
scripts/                  remote_pc/, viewer/, archive/
results/                  one folder per run (gitignored)
src/data/                 bake, annotations, preprocess, windowing, augment, resample
src/config/               typed data/model/training settings, loaders and validation
src/models/               RNN, PointNet, baseline, prediction and save/load methods, factory
src/training/             fitting, splitting, tuning, cross_validation, pipelines
src/evaluation/           metrics, records, plots, report, annotation_audit
tests/
```

- `main.py` reads the command line and turns errors into exit codes. `src/config/` loads and validates the config. `training/pipelines.py` exposes `run_cv`, `run_tune` and `run_train_final`; final training fits all windows directly without cross-validation.
- `src/config/data.py` defines `WindowConfig`. Its `signature()` returns the five prediction settings saved under `windowing` in checkpoints and tuned settings. Stride is omitted because it only changes sampling density.
- `training/fitting.py` owns the training loop, batch augmentation and training history. `fit_model(model, x, y, training=..., augment=..., seed=...)` fits a model; `fit_fold` also selects and optionally resamples training windows. `CrossValidator` in `training/cross_validation.py` coordinates held-out folds.
- `evaluation/records.py` holds the result records. The report and the loss curves read them. `results.json` is only read and written there.
- A model class declares its architecture parameters (`parameter_names`) and input type. `models/factory.py` builds it with `build_model(cfg.model)`; training and augmentation settings stay separate. Checkpoints store architecture and learned state, including input statistics, and continue to load legacy formats 1, 2 and 3. Run metadata stores the training settings.
