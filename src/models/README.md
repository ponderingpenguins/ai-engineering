# Models (`src/models/`)

Defines the posture classifier architectures, batched inference, and versioned checkpoint save/load logic. Optimisers, loss weighting, early stopping, and training-batch augmentations are intentionally kept outside this package in [src/training/README.md](../training/README.md).



## Modules

| Module          | Responsibility                                                                                                                          |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `model.py`    | `BaseModel` interface (`save`, `load`, `predict`), `TorchModel` base class, and `pick_device()` (CUDA, then MPS, then CPU). |
| `rnn.py`      | `HeatmapEncoder`, `HeatmapGRU`, and the `RNN` `TorchModel` wrapper.                                                             |
| `pointnet.py` | `MiniPointNet` and the `PointNet` `TorchModel` wrapper.                                                                           |
| `baseline.py` | `MajorityClass` label-only baseline.                                                                                                  |
| `factory.py`  | `MODEL_CLASSES`, `MODELS`, and `build_model(ModelConfig)`.                                                                        |



## Model Contract and Checkpoints (`model.py`)

Every classifier subclasses `BaseModel` and declares two class attributes:

- `model_name`: registry key in `factory.MODEL_CLASSES` (`"rnn"`, `"pointnet"`, `"baseline"`).
- `input_type`: tells `data.windowing.build_windows()` which tensor representation to construct (`"heatmap"`, `"pointcloud"`, or `"none"`).

Neural classifiers subclass `TorchModel`, which adds:

- `preprocessing`: version tag verified on checkpoint load to block pipeline mismatches.
- `parameter_names`: architecture keyword arguments stored in checkpoints so the network can be reconstructed without the original YAML file.
- `_build_net(n_classes)`: instantiates the fresh `nn.Module` at the start of training or checkpoint restore.
- `_set_input_stats(x)`: computes training-only standardisation buffers on `net` and returns the scale tensor so `src/training/fitting.py` can scale additive noise consistently.
- `predict(x, batch_size=32)`: runs `@torch.no_grad()` batched inference and maps class indices back through `classes_`.

### Checkpoint Format

`BaseModel.save(path, windowing)` writes plain dictionaries and tensors via `torch.save` with `format_version: 3`:

```python
{
    "format_version": 3,
    "model": "<model_name>",
    "windowing": { ... },        # WindowConfig.signature()
    "preprocessing": "<tag>",    # TorchModel subclasses
    "parameters": { ... },       # architecture kwargs only
    "classes": ["lying", ...],
    "state_dict": { ... },       # weights + input standardisation buffers
}
```

`BaseModel.load(path)` loads with `weights_only=True`, verifies the format version (v1–v3 are accepted for backward compatibility), checks the `windowing` signature keys, dispatches to the registered subclass (or checks exact type when called as `RNN.load(path)`), verifies `preprocessing`, restores weights in `.eval()` mode, and returns `(model, windowing)`.



## Architectures

### 1. `rnn` (`HeatmapGRU`, ~131k parameters)

- **Input:** `float32 (B, T, 1, 256, 16)` `log1p`-scaled range-Doppler heatmaps (`preprocessing = "log1p_raw_heatmap_v1"`).
- **Standardisation:** Global scalar buffers `input_mean` and `input_std` fit on training windows: $\hat{x} = (x - \mu) / \max(\sigma, 10^{-6})$.
- **Frame encoder (`HeatmapEncoder`):** Applied identically to all $B \cdot T$ frames:
  1. `Conv2d(1 -> 16, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d((2, 1))` $\rightarrow$ `(16, 128, 16)`
  2. `Conv2d(16 -> 32, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d((2, 1))` $\rightarrow$ `(32, 64, 16)`
  3. `Conv2d(32 -> 64, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d((2, 2))` $\rightarrow$ `(64, 32, 8)`
  4. `AdaptiveAvgPool2d((4, 2)) -> Flatten` (512 features) `-> Linear(512 -> feat_dim) -> Dropout(feature_dropout)`
- **Sequence head (`HeatmapGRU`):** `GRU(feat_dim -> hidden, batch_first=True)` reads the sequence of frame embeddings; its final hidden state $h_{:, -1}$ passes through `Dropout(dropout) -> Linear(hidden -> n_classes)`.



### 2. `pointnet` (`MiniPointNet`, ~27k parameters)

- **Input:** `float32 (B, P, 7)` radar points `[x, y, z, doppler, snr, t_norm, valid]` (`preprocessing = "pointcloud_xyz_doppler_snr_time_v1"`).
- **Standardisation:** Per-feature buffers `feat_mean` and `feat_std` $\in \mathbb{R}^5$ computed over valid training points (`valid > 0.5`) for `[x, y, z, doppler, snr]`; `t_norm` remains in $[0, 1]$.
- **Per-point shared MLP** (width $W$, default `128`):
$$
  \text{Linear}(6 \to W/4) \to \text{ReLU} \to \text{Linear}(W/4 \to W/2) \to \text{ReLU} \to \text{Linear}(W/2 \to W) \to \text{ReLU} \to \text{Dropout}(\text{point\_dropout})
$$
- **Masked symmetric pooling:** Over valid points $M = \{p : \text{valid}_p > 0.5\}$:
  - Masked max pool along points (set to $0$ when $|M| = 0$).
  - Masked mean pool along points (denominator clamped to $\ge 1$).
  - Count feature $c = \log(1 + n_{\text{points}}) / 5.0$, where `forward(x, n_points=...)` accepts the pre-`point_drop` count during training.
  - Concatenated vector $\in \mathbb{R}^{2W + 1}$.
- **Classifier head:** `Linear(2W + 1 -> W/2) -> ReLU -> Dropout(dropout) -> Linear(W/2 -> n_classes)`.



### 3. `baseline` (`MajorityClass`)

- **Input:** Ignored (`input_type = "none"`).
- **Behaviour:** `fit(x, y)` stores the most frequent training label in `label_`; `predict(x)` returns that label for every sample.



## Usage

```python
from config.loader import load_run_config
from models.factory import build_model
from models.model import BaseModel

# Instantiate from config
cfg = load_run_config(repo_root, ["model=pointnet"])
model = build_model(cfg.model)

# Load a saved checkpoint for inference
trained, windowing = BaseModel.load("results/<run>/checkpoints/fold1_Bob.pt")
predictions = trained.predict(x_test)
```



## Extending

### Adding a New Model Architecture

1. Create `src/models/<name>.py` subclassing `TorchModel` (or `BaseModel` for non-PyTorch models). Set `model_name`, `input_type`, `preprocessing`, and `parameter_names`, and implement `_build_net()` and `_set_input_stats()`.
2. Register `"<name>": <Class>` in `MODEL_CLASSES` inside `src/models/factory.py`.
3. Add architecture and augmentation dataclasses to `MODEL_SCHEMAS` in `src/config/models.py` and create `config/models/<name>.yaml` (see [config/README.md](../../config/README.md)).
4. If the model uses batch augmentation, register its augment class in `build_augment()` inside `src/training/fitting.py` and add any validation rules in `check_run_config()` inside `src/config/loader.py`.
5. Add unit tests (forward shapes, empty/padded edge cases, and checkpoint save/load round-trip) under `tests/models/`.
