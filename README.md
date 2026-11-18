# AI Engineering Practical - Valtes

## File Structure

```
├── config/                     # Project configuration files
├── data/                       # Input, intermediate, and other project data.
│   ├── raw/
│   └── processed/
│
├── deployment/                 # Deployment-related files and configurations
├── docs/                       # Project documentation
│   ├── report/                 # Ongoing project report. This folder will be built up over time with each assignment.
│   └── submissions/            # Assignment submission snapshots
│
├── logs/                       # Application and experiment logs
├── notebooks/                  # Jupyter notebooks for exploration and analysis
├── results/                    # Outputs and results from experiments or workflows
│
├── src/                        # Main source code
│   ├── data/                   # Data loading and processing code
│   ├── models/                 # Model definitions and implementations
│   ├── pipelines/              # End-to-end data and model pipelines
│   ├── utils/                  # Shared utility functions
│   └── experiments_folders/    # Experiment-specific codes and configurations. One folder per experiment.
│
├── tests/                      # Automated tests
├── .gitattributes              # Git attribute settings
├── .gitignore                  # Files and directories excluded from Git
├── .python-version             # Project Python version specification
├── pyproject.toml              # Python project metadata and tool configuration
├── README.md                   # Project overview and documentation
└── uv.lock                     # Locked Python dependency versions managed by uv
```

## Dependencies

To work with the jupyter notebooks:

```
uv pip install -e ".[jupyter]"
python -m ipykernel install --user --name floor-posture --display-name "floor-posture"
```

## Environment

Create the environment (first time) and activate it before running anything:

```
uv sync
source .venv/bin/activate
```

## Install Data

Run these commands before running any script. Install `git-lfs` with your package manager first (for example `sudo apt install git-lfs` on Debian/Ubuntu or `brew install git-lfs` on macOS).

```
git lfs install
git lfs pull

uv pip install -e ./data/raw/floor_posture_dataset/recorder
```

## Running `main.py`

```
python main.py [key=value ...]
```

`main.py` loads the baked dataset, so [bake it](#baking-the-dataset) first. It logs the row count, columns, label tracks, a per-recording summary (room, frames, sensor config, empty heatmaps) and one example frame.

Defaults come from `config/config.yaml`; `key=value` arguments on the command line override them (OmegaConf dotlist syntax). Unknown keys are rejected with a usage message.

| Parameter  | Default                        | Description                                                                             |
| ---------- | ------------------------------ | --------------------------------------------------------------------------------------- |
| `data.baked` | `data/processed/floor_posture` | Baked dataset folder. Relative paths are resolved against the repo root.                |
| `data.preprocess` | `[config_4]` | Extra frame filters run once after loading, before windowing (`src/data/preprocess.py`). Recordings without a known person are always dropped first, whatever is listed. |
| `eval.folds` | `null` | Cross-validation folds over people. `null` is one fold per person; an integer `k` gives stratified k-fold. |
| `eval.window_len` / `eval.stride` | `16` / `8` | Window length and step in frames. One frame is about 0.2 s, so this is 3.2 s windows every 1.6 s. The run logs the exact seconds. |
| `eval.label_rule` | `any_lying` | How a window gets its label, see below. |
| `eval.max_windows_per_person` | `null` | Keep at most this many windows per person, for quick test runs: all the person's rarer-class windows first, then random windows of the common class. `null` uses every window. Do not use it for results you report. |

Example:

```
python main.py data.baked=/path/to/floor_posture
python main.py model.name=baseline eval.folds=2 eval.label_rule=last
```

### How windows are labelled

Frames are cut into windows of `eval.window_len` frames (about 0.2 s each, measured from the capture timestamps). Windows that span a gap of dropped frames are skipped.

- **`any_lying` (default):** the classifier predicts whether lying **occurred** within the window's 3.2 s. A window is labelled lying if at least one of its frames is lying, and not lying if none is. If the window has several lying frames, the last one decides between `lying_left` and `lying_right` (when `eval.collapse_lying=false`).
- **`last`:** a window takes the label of its last frame.

Why `any_lying`: in the labelled data, lying is annotated in short runs of 1 to 5 frames (0.2 to 1 s), scattered through recordings that last about 6 minutes. With `last`, a window is only positive when it happens to end on one of those frames: with 16-frame windows and stride 8, that kept 7 lying windows out of 1450. Requiring most of the window to be lying (majority) kept none. With `any_lying`, the same data gives 37 lying windows, and each person has some.

Consequences to keep in mind when reading scores:

- **What a positive means:** lying occurred at least once in the window's 3.2 s. It does not mean the person was lying for the whole window, and a positive window may contain 15 non-lying frames. Do not read lying recall or precision as "how long someone lay" or "how well a lying posture is held". They measure detection of lying occurring.
- **What a negative means:** none of the window's frames is lying. A window is not-lying only if lying never occurred in it, so a window next to a lying episode can be a negative.
- Windows overlap by half, so one lying frame can label two neighbouring windows. The 37 lying windows come from 33 lying frames in 6 recordings, so they are not independent.
- Windows are cut after the preprocessing filters, so recordings without a known person are never windowed.
- Checkpoints record the label rule. Checkpoints saved before the rule existed are treated as `last`.

### Models and their inputs

| `model.name` | Input per window | Notes |
| --- | --- | --- |
| `rnn` | the 16 range-Doppler heatmaps (16 x 256 x 16), log-scaled | CNN per frame, GRU over the frames. About 131k parameters. |
| `pointnet` | the radar **point cloud** of the window | Mini PointNet, about 27k parameters. See below. |
| `baseline` | none | Always predicts the most common training label. |

All models use the same windows, folds, preprocessing, class weighting, validation loss and early stopping (`models/torch_classifier.py`), so their results are comparable. Each model declares what it reads with `input_type`, and `make_windows(inputs=...)` builds it.

**The point-cloud input.** The radar reports 0 to 16 points per frame (median 2; 27% of labelled frames have none, and the radar's tracker never reports a track). For a window the points of all its frames are put into one set of at most `eval.max_points` (default 128) points with the columns `x, y, z, doppler, snr, time` (time is 0 for the first frame of the window and 1 for the last) and a flag marking real points versus padding. A 16-frame window holds about 41 points (99th percentile 105), so 128 truncates 0.3% of windows; if there are more, the strongest by SNR are kept. The radar's noise value and its track id (always 255) are not used.

**The mini PointNet** applies the same small MLP (6 to 32 to 64 to 128 features) to every point, pools the valid points with max and mean (padding never reaches the pooled features, and the order of the points does not matter), adds the log of the number of points (lying windows tend to have more) and classifies with a small head. There is no alignment network (T-Net). x, y, z, doppler and snr are standardised with statistics of the training points. Run it with `python main.py model.name=pointnet`; its hyperparameters are under `model.pointnet.*`. Regularisation (all under `model.pointnet.*`): `dropout` (head, 0.5), `point_dropout` (per-point features, 0.2), `weight_decay` (1e-2), and two augmentations that only act while training: `point_drop` (drops 30% of the points) and `jitter` (noise on the standardised features). Set them to 0 to switch them off.

### Training logs, early stopping and checkpoints

Every `python main.py` run writes a folder `results/<timestamp>_<model>/` containing:

| File | What it is |
| --- | --- |
| `report.md` | Short report: settings, pooled results, per-fold and per-person tables, the loss figures and checkpoint paths. |
| `loss_per_fold.png` | One panel per fold: train loss at every step, validation loss at every evaluation, the step whose weights were kept, and where early stopping fired. |
| `loss_mean.png` | Train and validation loss against epochs as the mean over folds, with a shaded band of ±1 standard deviation across folds. A point needs at least two folds still training. |
| `results.json` | Everything above in machine-readable form, including the full per-step loss history of each fold. |
| `checkpoints/fold<k>_<test person>.pt` | The best weights of each fold (the ones kept after early stopping). |

Rebuild the report and figures of an old run with `python -m utils.run_report results/<run>`. The checkpoints load with `BaseModel.load`.

How training works for the `rnn` model:

- **Validation person:** in every fold, one of the training people is held out as validation data (chosen with `eval.seed`), so the validation loss measures generalisation to an unseen person. The test person is never used for validation. The model therefore trains on one person fewer than the fold's training set, which with 4 people is a real cost. With fewer than two training people there is no validation and the model trains for all epochs.
- **Early stopping:** `model.rnn.patience` (default 5). Training stops after that many validation evaluations in a row without a lower validation loss, and the weights with the lowest validation loss are restored. `model.rnn.epochs` (default 30) is only the upper limit.
- **When validation runs:** `model.rnn.evals_per_epoch` times per epoch (default 1, so patience 5 means 5 epochs). With about 10 to 30 steps per epoch, raising this makes patience cover very few steps; the train loss is always recorded at every step.
- **The loss** in the figures is cross-entropy that treats the classes equally, not the evaluation metric. The train loss weights each class by the inverse of its frequency in the training windows (so it is plain cross-entropy once the training data is balanced). The validation loss is class-balanced: the mean of the per-class mean losses, with class weights that come from `val_weight_labels` (by default the validation set itself). A model that always predicts 50/50 scores ln 2 = 0.693 on it, whatever the class sizes, and that is the level to compare against. During tuning the weights are pooled over all training people of the outer fold (see below). A validation loss hovering around 0.69 means the model has learned nothing that carries over to a new person, and early stopping is then only picking noise.
- **Fixed epochs, no validation** (`train.validation=false`): every fold trains for exactly `model.rnn.epochs` epochs on all its training people, keeps the last weights, and the figures show the train loss only. Use it when one validation person is too noisy to decide anything (with 4 people it is). The epoch count must come from earlier validation curves, never from test scores. Example: `python main.py train.validation=false model.rnn.epochs=3`.
- **Oversampling** (`train.oversample`, default off): repeats randomly chosen training windows of the rarer class until each reaches that fraction of the largest class (`1.0` = balanced, `0.5` = half). It is applied to the training part of each fold only; the validation person and the test person keep their real class balance. It repeats existing windows and adds no new information, so it can make the model memorise the few lying windows. The report's per-fold results and `results.json` record the class counts after resampling.
- Models that do not train iteratively (`baseline`) have no loss curves; their report just omits that section.

## Running on the PC

Training runs on `haukurs_pc` (a Windows machine with an RTX 3070, reached over Tailscale; commands run inside its WSL). Helper scripts in `scripts/pc/` wrap the SSH details. They need Tailscale running on this machine and the PC switched on.

| Script | What it does |
| --- | --- |
| `status.sh` | Reachability, GPU, disk, running jobs, newest log and result folder, and whether the dataset and torch are ready. |
| `setup.sh` | One-time: installs `uv` on the PC and runs `uv sync` there (run `push-code.sh` first). |
| `push-code.sh [-n]` | Copies the code to the PC (not `.git`, notebooks, results or the big data folders). `-n` is a dry run. |
| `push-data.sh [folder]` | Copies the baked dataset (about 3 GB, resumable) to the PC. |
| `train.sh <main.py args>` | Starts `python main.py <args>` as a background job, e.g. `train.sh train.validation=false model.rnn.epochs=3`. |
| `job.sh '<command line>'` | Starts any command line as a background job, e.g. a loop over seeds. `JOB_NAME=name` names the log. |
| `logs.sh [-f] [file]` | Shows the newest log on the PC (`-f` follows it). |
| `run.sh <command>` | Runs one command on the PC in the project folder and streams the output. |
| `stop.sh` | Stops the background jobs (asks first). |
| `fetch-results.sh [-n]` | Copies the result folders back into `./results` here. Never deletes anything locally. |

Typical run: `scripts/pc/push-code.sh`, then `scripts/pc/train.sh ...`, `scripts/pc/logs.sh -f`, and finally `scripts/pc/fetch-results.sh`.

Things worth knowing:

- The PC's ssh config logs you straight into WSL, which blocks `ssh host '<command>'`. The scripts override that per call and pipe the script into `bash` in WSL, so `~/.ssh/config` is untouched. `PC_HOST` and `PC_DIR` (default `/home/haukur/rug/ai-engineering`) can be overridden in the environment.
- WSL shuts the whole Linux instance down about 15 seconds after the last SSH session closes, which kills every process, including `nohup`, `setsid`, `tmux` and systemd services (all tested). So `job.sh` leaves one SSH session open in the background on this machine for exactly as long as the job runs (it ends by itself with the job). **The Mac must stay awake and connected (Tailscale on) until the job finishes**; the scripts deliberately do not stop it from sleeping, so if it sleeps or the connection drops, the job dies about 15 seconds later. Jobs run as `systemd-run --user` units so that `status.sh` and `stop.sh` can see and stop them. A permanent fix would be on the PC itself (Windows side), for example an idle-timeout setting in `.wslconfig` or a keep-alive task; the scripts do not change anything there.

## Baking the dataset

`src/data/extract.py` reads every recording once through `vt_radar` and writes a single Hugging Face dataset (one row per frame) to `data/processed/floor_posture` (gitignored):

```
python -m data.extract [data.root=...] [data.baked=...] [num_proc=4] [overwrite=true] ['recordings=["a","b"]']
```

After extracting the data, synchronize the environment before opening the notebooks:

```bash
uv sync
source .venv/bin/activate
```

Quote recording names: YAML reads an unquoted `2311__20231120_144541` as a number. It takes about 20 s with `num_proc=6`, and the result is about 4.4 GB.

To use the processed dataset:

```python
from datasets import load_from_disk

ds = load_from_disk("data/processed/floor_posture")
ds = ds.select_columns(["recording", "heatmap", "label_pose"])   # or ds.remove_columns([...])
```

Columns: `recording`, `room`, `frame_idx`, `timestamp`, `radar_frame_num`, `image` (camera JPEG), `radar_payload` (raw TLV bytes), `heatmap` (range x doppler, its size varies per recording), `pointcloud`, `n_points`/`n_tracks`/`n_heights`, the sensor config (`range_resolution_m`, `num_range_bins`, ...), and `label_<track>` plus `label_<track>_confidence` for every label track found. Tracks a recording doesn't have are null.

Two things to know about `heatmap`. It is stored raw, with zero velocity at doppler bin 0, so apply `np.fft.fftshift(heatmap, axes=-1)` to centre it (see the notebook). And 8 frames across 7 recordings have an empty heatmap (`[]`, and `heatmap_ok` is `False`), so filter on `heatmap_ok` before stacking.

`notebooks/explore_baked_dataset.ipynb` loads the dataset and plots a frame (camera, heatmap, point cloud), the heatmap over time with a label track, and label distributions.

Keep in mind that the baked data contains identifiable feeds.

## Manual annotations

Two columns of the baked dataset are not in the raw data: they were annotated by hand and are joined in by `src/data/extract.py`. The source CSVs are small and tracked in git (see `.gitignore`), so rebuild the dataset (`overwrite=true`) after changing them.

| Column | Source file | Made with | Meaning |
| ------ | ----------- | --------- | ------- |
| `person` | `data/processed/person_map.csv` (`recording, person`) | `notebooks/label_persons.ipynb` | Pseudonym of the participant in the recording (for example `Bob`), so evaluation can be split by person. Recordings that are not labelled get `"NA"`. |
| `person_present` | `data/processed/presence_events.csv` (`recording, frame_idx, event`) | `notebooks/label_presence.ipynb` | `True` while someone is in view of the camera, `False` for an empty scene. |

- **`person`** is one value per recording. The notebook shows the best face crops of each recording and you pick a name from a fixed list of pseudonyms. Unlabelled recordings are dropped by the `known_person` preprocessor.
- **`person_present`** is stored as `enter` and `leave` events on frame indices. A person is present from each `enter` until the next `leave` (or the end of the recording), so a recording that starts with someone already in view needs an `enter` on frame 0. Recordings without any events get `None`, not `False`, so filter on `person_present.notna()` before training on it. This is separate from the raw `label_presence` track, which is not reliable (it is `False` wherever it exists).

## Evaluation workflow

Everything is split by person, and there are four known people, so we use nested leave-one-person-out cross-validation. **Run the tuning first, then the full run:** the full run reads the setting the tuning run saved.

```bash
# 1. Tuning: nested cross-validation over the grid. Writes results/tuned.yaml and the report.
python main.py 'tune.grid=[{hidden: 64}, {hidden: 128, lr: 0.0003}]'

# 2. Full run: trains one model on all people with the saved setting. Writes results/<run>_final/final.pt.
python main.py train.full_run=true
```

What each step does:

1. **Outer loop (testing).** Hold out one person as the test person.
2. **Inner loop (tuning).** On the remaining people, run leave-one-person-out again: each of them is the validation person once, the others train. Every setting of `tune.grid` is tried this way. The winner is the setting with the lowest mean **class-balanced validation loss** (taken at the weights early stopping kept), with class weights pooled over all people of the outer training fold. The validation macro-F1 is reported next to it but does not decide. The epoch count is the median best epoch of the winner's inner folds.
3. **Refit.** Train on all the remaining people with the winning setting for that many epochs (no validation person is left, so no early stopping), and score on the test person.
4. **Report** the outer scores: the per-person table first (lying recall with its support), then the pooled score over all test predictions. `report.md` has a Tuning section with every setting's inner scores for each outer fold.
5. **Best setting.** The setting that won the most outer folds (ties: lower mean validation loss), with the median of its epoch counts, is written to `tune.best_path` (default `results/tuned.yaml`, a copy goes into the run folder as `tuned.yaml`) together with the model and task it was tuned for.
6. **Full run** (`train.full_run=true`). Skips cross-validation, trains one model on all people with the saved setting and epoch count, and saves `final.pt` and `final.json` in a `<timestamp>_<model>_final` run folder. It stops with an error if `tuned.yaml` is missing, if `tune.grid` is also set, or if the model or task settings (`eval.target`, `collapse_lying`, `window_len`, `label_rule`) differ from the ones it was tuned with: tune again in that case. The final model has no held-out score of its own; we assume that more training data does not make it worse than the cross-validated models.

The test person is never used to choose settings, so the outer scores are not inflated by tuning.

**Why the validation class weights are pooled.** Settings are ranked by the mean validation loss over the inner validation people, so those losses must be on one scale. If each person's loss were weighted by that person's own class mix, they would not be: a person with only `not_lying` windows gets weight 1 on that class and a plain cross-entropy, which is much easier to get low than a balanced loss on a person who also has `lying` windows. The mean would then reward whichever setting does well on the easiest validation people, and the winner would be the best setting for the easiest case instead of the best setting overall. Pooling the class weights over all people of the outer training fold gives every validation person, and every setting, the same weights. The loss is then a fair comparison across people and settings, and a model that predicts 50/50 scores ln 2 = 0.693 for everyone.

`tune.grid` is a list of `model.rnn` overrides (`hidden`, `lr`, `weight_decay`, `epochs`, ...). Write a space after each colon (`hidden: 64`, not `hidden:64`), or YAML reads the whole thing as one key; unknown keys are rejected. Tuning is only defined for `model.name=rnn`, and it replaces the single random validation person of `train.validation`. A single-setting grid is a way to choose only the epoch count. Settings not named in the grid keep the values from the config (or the command line) for the tuning run; the full run applies the saved setting on top of the defaults, so pass the same other overrides (such as `train.oversample`) to both steps.

Caveats:

- With 4 people the inner loop validates on one person at a time, who has only a few lying windows (they overlap, see "How windows are labelled"). Tuning is noisy, so keep the search small.
- The cost is (people - 1) trainings per setting and per outer fold, plus one refit, so 4 x 3 = 12 trainings per setting for 4 people. Run it on the PC (`scripts/pc/train.sh`).
- The outer scores describe the whole procedure (tuning included) on a new person, not the one final model.
- The setting chosen for the final model comes from folds that each saw different people, so it is a vote, not a score.

Recordings from sensor configs other than Config 4, and recordings without a known participant, are dropped
by the preprocessing pipeline (`data.preprocess` in the config) before windowing.

After training a model, save it with the task settings used for that run:

```python
model.save("results/rnn.pt", evaluation={
    "target": cfg.eval.target,
    "collapse_lying": cfg.eval.collapse_lying,
    "window_len": cfg.eval.window_len,
})
```

To load a saved model:

```python
from models.model import BaseModel

model, evaluation = BaseModel.load("results/rnn.pt")
```

RNN checkpoints include weights, normalization, class labels, and model settings.
Existing training entry points do not save checkpoints automatically (possible TODO).

Each run writes `results/comparisons/<timestamp>/comparison.md` with PNG graphs in `figures/`.
Keep the report and figures together. `predictions.csv` includes held-out predictions, timestamps,
and the exact ordered frame IDs for each window; `failures.csv` contains only mistakes.
`participants.csv`, `metrics.json`, and `config.json` contain detailed scores and run settings.
Scores describe test windows, not distinct events. Times cover test prediction only.

## Type checking

Type hints are checked with [mypy](https://mypy-lang.org/), a dev dependency (installed by `uv sync`):

```
uv run mypy main.py --ignore-missing-imports
```

Or, with the environment activated: `mypy main.py --ignore-missing-imports`.

`--ignore-missing-imports` is needed because `vt_radar` ships without type stubs.
