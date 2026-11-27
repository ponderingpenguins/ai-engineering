# Configuration - (`config/` and `src/config/`)

Runtime configuration combines typed Python dataclasses in `src/config/`, YAML default files in `config/`, and command-line `key=value` overrides parsed by OmegaConf.  Unknown keys raise an error immediately since we are using strict struct mode.



## Files

| Path                                                    | Responsibility                                                                                                           |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `config/data.yaml`, `src/config/data.py`            | Dataset paths, recording selection, optional heatmap denoising, and windowing rules (`DataConfig`).                   |
| `config/train.yaml`, `src/config/training.py`       | Active model selection, random seed, output directory, optimiser loop, CV, and tuning settings (`RunConfig`).          |
| `config/models/<name>.yaml`, `src/config/models.py` | Per-model architecture and batch-augmentation settings (`rnn`, `pointnet`, `baseline`) and `MODEL_SCHEMAS`.      |
| `src/config/loader.py`                                | Public loaders (`load_data_config`, `load_run_config`) and validators (`check_data_config`, `check_run_config`). |
| `config/radar/`                                       | Bundled TI mmWave sensor `.cfg` profiles used with `python main.py record --cfg-file ...`.                       |



## Loading Walkthrough

Two separate loaders isolate data tools from training and model definitions:

1. **Data tools (`bake`, `annotate`, `audit`)** call `config.loader.load_data_config(root, overrides)`.
   - Builds the `DataConfig` dataclass schema.
   - Merges `config/data.yaml`, then CLI overrides.
   - Runs `check_data_config()`. Training and model YAML files are never loaded.
2. **Training pipelines (`cv`, `tune`, `train-final`)** call `config.loader.load_run_config(root, overrides)`.
   - Reads the default model name from `config/train.yaml`, overridden by CLI `model=<name>` (`model.name=<name>` is rejected).
   - Instantiates a strict `RunConfig` schema bound to that model's `(arch, augment)` classes from `MODEL_SCHEMAS`.
   - Merges `config/data.yaml`, `config/train.yaml`, `config/models/<name>.yaml`, and the remaining CLI overrides, in that order.
   - Runs `check_run_config()`, which calls `check_data_config()` and enforces all training and augmentation constraints.



## Configuration Reference

| Section / Key                                                             | Default                                                                                  | Role and Constraints                                                                                                                                                                    |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `paths.raw_root` / `baked_dir` / `annotations_dir`                  | `data/raw/floor_posture_dataset` / `data/baked/floor_posture` / `data/annotations` | Relative paths resolve against the repository root.                                                                                                                                     |
| `selection.sensor_config`                                               | `Config_4.cfg`                                                                         | Keeps recordings matching this `.cfg` file (`null` keeps all). Unknown participants (`"NA"`) are always dropped (`require_known_person` is fixed to `True`).                  |
| `selection.exclude_recordings`                                          | `2311__20231124_152926`, `2311__20231124_153817`                                     | Recordings removed before windowing (non-top-down camera mounts).                                                                                                                       |
| `denoising.range_normalise`                                             | `enabled: false`, `percentile: 10.0`, `smooth_bins: 41`                            | Non-causal per-recording noise-floor subtraction. Requires `percentile` in `[0, 100]` and positive odd `smooth_bins`. Cannot be combined with `model.augment.floor_jitter > 0`. |
| `denoising.temporal_background`                                         | `enabled: false`, `tau_frames: 300.0`                                                | Causal exponential moving average subtraction. Requires `tau_frames >= 1`.                                                                                                            |
| `windowing.target`                                                      | `lying_annotation`                                                                     | Per-frame label column (`lying_annotation` or `label_pose`).                                                                                                                        |
| `windowing.label_rule`                                                  | `any_lying`                                                                            | `any_lying` or `last`. `model.augment.time_reverse=true` requires `any_lying`.                                                                                                  |
| `windowing.collapse_lying`                                              | `true`                                                                                 | Collapses any `lying*` label to `lying` and all other labels to `not_lying`. Required when PointNet `yaw_deg > 0` or `mirror=true`.                                           |
| `windowing.window_len` / `stride` / `max_points`                    | `16` / `8` / `128`                                                                 | All must be positive integers.                                                                                                                                                          |
| `training.lr` / `weight_decay`                                        | `0.001` / `0.01`                                                                     | `lr > 0`, `weight_decay >= 0`.                                                                                                                                                      |
| `training.epochs` / `batch_size` / `patience` / `evals_per_epoch` | `30` / `32` / `5` / `1`                                                          | All must be positive integers.                                                                                                                                                          |
| `training.validation`                                                   | `true`                                                                                 | Holds out one training participant per fold for early stopping.                                                                                                                         |
| `training.class_weights`                                                | `true`                                                                                 | Inverse-frequency class weighting on training cross-entropy.                                                                                                                            |
| `training.oversample_ratio`                                             | `null`                                                                                 | `null` disables oversampling; a value in `(0, 1]` tops up rarer training classes to that fraction of the majority class.                                                            |
| `model.augment.*`                                                       | See `config/models/<name>.yaml`                                                        | All numeric magnitudes must be non-negative; `point_drop` must be in `[0, 1]`. Any geometric transform requires `selection.sensor_config=Config_4.cfg`.                           |
| `cv.folds` / `cv.max_windows_per_person`                              | `null` / `null`                                                                      | `folds=null` runs leave-one-person-out (`>= 2` if integer). `max_windows_per_person` caps windows per participant for quick checks.                                               |
| `tune.grid` / `tune.best_path`                                        | `[]` / `results/tuned.yaml`                                                          | Dotted hyperparameter overrides searched by `tune`; `train-final` reads `best_path`.                                                                                              |

### Windowing Compatibility Signature

`WindowConfig.signature()` returns `{target, collapse_lying, window_len, label_rule, max_points}`. It intentionally excludes `stride`, which only changes window sampling density. Every saved `.pt` checkpoint and `tuned.yaml` stores this signature and refuses to load when the active run's signature differs.



## Usage

```bash
# Override data, model, and training keys from the CLI
python main.py cv model=pointnet training.epochs=15 model.augment.shift_m=0.3

# Run nested hyperparameter tuning with dotted keys
python main.py tune 'tune.grid=[{model.arch.hidden: 64}, {model.arch.hidden: 128, training.lr: 0.0003}]'
```

```python
from pathlib import Path
from config.loader import load_data_config, load_run_config

data_cfg = load_data_config(Path("."), ["windowing.window_len=32"])
run_cfg = load_run_config(Path("."), ["model=rnn", "training.epochs=10"])
```



## Extending

### Adding a Configuration Field

1. Add the typed field and default value to the corresponding dataclass in `src/config/data.py`, `src/config/models.py`, or `src/config/training.py`.
2. Add the same key and an inline comment to the matching file in `config/`.
3. Add range or cross-field validation to `check_data_config()` or `check_run_config()` in `src/config/loader.py`.
4. Add a unit test under `tests/config/`.

### Adding a Model Configuration

1. Define `<Name>Architecture` and `<Name>AugmentConfig` dataclasses in `src/config/models.py` and register `"<name>": (<Name>Architecture, <Name>AugmentConfig)` in `MODEL_SCHEMAS`.
2. Create `config/models/<name>.yaml` with default values.
3. Add any model-specific validation rules in `check_run_config()` inside `src/config/loader.py`.
4. Register the architecture itself in `src/models/` (see [src/models/README.md](../src/models/README.md)).
