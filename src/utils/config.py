from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class DataConfig:
    # relative paths are resolved against the repo root
    root: str = MISSING  # raw dataset bundle (MANIFEST.csv, data/, ...)
    baked: str = "data/processed/floor_posture"  # baked HF dataset (gitignored)
    # per-recording sensor config (constant within a recording)
    cfg_columns: list[str] = field(
        default_factory=lambda: [
            "cfg_file",
            "range_resolution_m",
            "range_max_m",
            "num_range_bins",
            "num_doppler_bins",
            "doppler_resolution_m_s",
            "max_velocity_m_s",
            "num_tx",
            "num_rx",
        ]
    )


@dataclass
class RNNConfig:
    hidden: int = 128  # GRU hidden size
    feat_dim: int = 64  # size of the per-frame feature vector from the CNN
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1
    batch_size: int = 32
    seed: int = 0


@dataclass
class ModelConfig:
    name: str = "rnn"  # "rnn" or "baseline", see models/factory.py
    rnn: RNNConfig = field(default_factory=RNNConfig)


@dataclass
class EvalConfig:
    folds: int = 5  # k for the person-wise k-fold cross-validation
    target: str = "label_pose"  # label column to predict
    collapse_lying: bool = True  # True: lying vs. not lying, False: the original pose classes
    window_len: int = 16  # frames per window
    stride: int = 8  # frames between window starts
    out_dir: str = "results"  # each run gets its own folder in here


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
