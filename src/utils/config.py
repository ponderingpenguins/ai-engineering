from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class DataConfig:
    # relative paths are resolved against the repo root
    root: str = MISSING  # raw dataset bundle (MANIFEST.csv, data/, ...)
    baked: str = "data/processed/floor_posture"  # baked HF dataset (gitignored)
    # preprocessors run in order on the loaded frames, see data/preprocess.py
    preprocess: list[str] = field(default_factory=lambda: ["config_4", "known_person"])


@dataclass
class RNNConfig:
    hidden: int = 128  # GRU hidden size
    feat_dim: int = 64  # size of the per-frame feature vector from the CNN
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30  # upper limit; early stopping usually ends training sooner
    batch_size: int = 32
    seed: int = 0
    patience: int = 5  # stop after this many validation evaluations without improvement
    evals_per_epoch: int = 1  # validation loss evaluations per epoch; patience counts these


@dataclass
class ModelConfig:
    name: str = "rnn"  # "rnn" or "baseline", see models/factory.py
    rnn: RNNConfig = field(default_factory=RNNConfig)


@dataclass
class TrainConfig:
    # None: off. A ratio in (0, 1] repeats training windows of the rarer classes until each reaches
    # that share of the largest class (1.0 = balanced). Training data only, see data/resample.py.
    oversample: Optional[float] = None
    # True: each fold holds out one training person for validation (early stopping, best weights).
    # False: no validation; every fold trains for exactly model.rnn.epochs on all its training people.
    validation: bool = True
    # True: skip cross-validation and train one model on all people with the setting saved by the tuning run
    # (tune.best_path), then save it as final.pt. Run the tuning first. Cannot be combined with tune.grid.
    full_run: bool = False


@dataclass
class TuneConfig:
    # Nested cross-validation: a list of hyperparameter overrides of model.rnn, e.g.
    # [{hidden: 64}, {hidden: 128, lr: 0.0003}]. Every outer fold tries each setting with leave-one-person-out
    # over its training people, keeps the best and refits it for the median best epoch. Empty: no tuning.
    grid: list[dict[str, Any]] = field(default_factory=list)
    # Where the best setting is written after a tuning run, and read from by train.full_run.
    # Relative paths are resolved against the repo root.
    best_path: str = "results/tuned.yaml"


@dataclass
class EvalConfig:
    folds: Optional[int] = None  # k for person-wise k-fold cross-validation; None = one fold per person
    seed: int = 0  # shuffles the fold assignment
    target: str = "label_pose"  # label column to predict
    collapse_lying: bool = True  # True: lying occurred vs. not (with any_lying), False: the original pose classes
    window_len: int = 16  # frames per window
    label_rule: str = "any_lying"  # "any_lying": predict whether lying occurred within the window; "last": label of its last frame
    stride: int = 8  # frames between window starts
    out_dir: str = "results"  # each run gets its own folder in here


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
