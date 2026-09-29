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

## Type checking

Type hints are checked with [mypy](https://mypy-lang.org/), which is a dev dependency (installed by `uv sync`):

```
uv run mypy main.py --ignore-missing-imports
```

Or, with the environment activated: `mypy main.py --ignore-missing-imports`.

`--ignore-missing-imports` is needed because `vt_radar` ships without type stubs.