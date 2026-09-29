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

To use the custom `recorder` package by Valtes:

```
uv pip install -e ./data/raw/floor_posture_dataset/recorder
```

## Environment

Create the environment (first time) and activate it before running anything:

```
uv sync
source .venv/bin/activate
```

`main.py` exits with "Did you forget to activate your environment?" if the `vt_radar` package (see above) can't be imported.

## Running `main.py`

```
python main.py [key=value ...]
```

Defaults come from `config/config.yaml`; `key=value` arguments on the command line override them (OmegaConf dotlist syntax). Unknown keys are rejected with a usage message.

| Parameter   | Default                            | Description                                                                                                                   |
| ----------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `data.root` | `data/raw/floor_posture_dataset`   | Dataset folder containing `data/`, `MANIFEST.csv` and `HEATMAP_SHAPES.csv`. Relative paths are resolved against the repo root. |

Example:

```
python main.py data.root=/path/to/floor_posture_dataset
```

## Baking the dataset

`src/data/extract.py` reads every recording once through `vt_radar` and writes a single Hugging Face dataset (one row per frame) to `data/processed/floor_posture` (gitignored):

```
python -m data.extract [data.root=...] [data.out=...] [num_proc=4] [overwrite=true] ['recordings=["a","b"]']
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

## Type checking

Type hints are checked with [mypy](https://mypy-lang.org/), a dev dependency (installed by `uv sync`):

```
uv run mypy main.py --ignore-missing-imports
```

Or, with the environment activated: `mypy main.py --ignore-missing-imports`.

`--ignore-missing-imports` is needed because `vt_radar` ships without type stubs.
