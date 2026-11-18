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
| `data.preprocess` | `[config_4, known_person]` | Frame filters run once after loading, before windowing (`src/data/preprocess.py`). |
| `eval.folds` | `null` | Cross-validation folds over people. `null` is one fold per person; an integer `k` gives stratified k-fold. |
| `eval.window_len` / `eval.stride` | `16` / `8` | Window length and step in frames. One frame is about 0.2 s, so this is 3.2 s windows every 1.6 s. The run logs the exact seconds. |
| `eval.label_rule` | `any_lying` | How a window gets its label, see below. |

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

### Training logs, early stopping and checkpoints

Every `python main.py` run writes a folder `results/<timestamp>_<model>/` containing:

| File | What it is |
| --- | --- |
| `report.md` | Short report: settings, pooled results, per-fold and per-person tables, the loss figures and checkpoint paths. |
| `loss_per_fold.png` | One panel per fold: train loss at every step, validation loss at every evaluation, the step whose weights were kept, and where early stopping fired. |
| `loss_mean.png` | Train and validation loss against epochs as the mean over folds, with a shaded band of ±1 standard deviation across folds. A point needs at least two folds still training. |
| `results.json` | Everything above in machine-readable form, including the full per-step loss history of each fold. |
| `checkpoints/fold<k>_<test person>.pt` | The best weights of each fold (the ones kept after early stopping). |

Rebuild the report and figures of an old run with `python -m utils.run_report results/<run>`. The checkpoints load with `BaseModel.load` and can go straight into `python -m utils.compare`.

How training works for the `rnn` model:

- **Validation person:** in every fold, one of the training people is held out as validation data (chosen with `eval.seed`), so the validation loss measures generalisation to an unseen person. The test person is never used for validation. The model therefore trains on one person fewer than the fold's training set, which with 4 people is a real cost. With fewer than two training people there is no validation and the model trains for all epochs.
- **Early stopping:** `model.rnn.patience` (default 5). Training stops after that many validation evaluations in a row without a lower validation loss, and the weights with the lowest validation loss are restored. `model.rnn.epochs` (default 30) is only the upper limit.
- **When validation runs:** `model.rnn.evals_per_epoch` times per epoch (default 1, so patience 5 means 5 epochs). With about 10 to 30 steps per epoch, raising this makes patience cover very few steps; the train loss is always recorded at every step.
- **The loss** in the figures is class-weighted cross-entropy (rare classes weigh more), for both train and validation, so the two curves are comparable. It is not the evaluation metric.
- Models that do not train iteratively (`baseline`) have no loss curves; their report just omits that section.

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
| `person` | `data/processed/person_map.csv` (`recording, person`) | `notebooks/label_persons.ipynb` | Pseudonym of the participant in the recording (for example `Bob`), so splits can be made by person. Recordings that are not labelled get `"NA"`. |
| `person_present` | `data/processed/presence_events.csv` (`recording, frame_idx, event`) | `notebooks/label_presence.ipynb` | `True` while someone is in view of the camera, `False` for an empty scene. |

- **`person`** is one value per recording. The notebook shows the best face crops of each recording and you pick a name from a fixed list of pseudonyms. Unlabelled recordings always stay in the train set of the person-wise split.
- **`person_present`** is stored as `enter` and `leave` events on frame indices. A person is present from each `enter` until the next `leave` (or the end of the recording), so a recording that starts with someone already in view needs an `enter` on frame 0. Recordings without any events get `None`, not `False`, so filter on `person_present.notna()` before training on it. This is separate from the raw `label_presence` track, which is not reliable (it is `False` wherever it exists).

## Evaluation workflow

Everything is split by person, and there are four known people, so we use nested leave-one-person-out cross-validation.

1. **Outer loop (testing).** Hold out one person as the test person.
2. **Inner loop (tuning).** On the remaining people, run leave-one-person-out again: each of them is the validation person once, the others train. Compare hyperparameters on the pooled validation predictions and pick the best settings. Early stopping and the epoch count also come from the inner loop.
3. **Refit.** Train on all the remaining people with the best settings and score on the test person.
4. **Report** the outer scores: the per-person table first (lying recall with its support), then the pooled score over all test predictions.
5. **Final model.** Train on all people with the settings that won most often in the inner loops. There is no validation person for this run, so it uses the median epoch count from the inner loops. It has no held-out score of its own; we assume that more training data does not make it worse than the cross-validated models.

The test person is never used to choose settings, so the outer scores are not inflated by tuning.

Caveats:

- With 4 people the inner loop validates on one person at a time, who has only a few lying windows (they overlap, see "How windows are labelled"). Tuning is noisy, so keep the search small.
- The cost is (people - 1) trainings per setting and per outer fold, plus one refit, so 4 x 3 = 12 trainings per setting for 4 people.
- The outer scores describe the whole procedure (tuning included) on a new person, not the one final model.
- The nested loop is not implemented yet: `main.py` currently runs only the outer loop, with no tuning.

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
