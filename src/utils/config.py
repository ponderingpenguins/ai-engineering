from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

from omegaconf import MISSING, OmegaConf


@dataclass
class DataConfig:
    # relative paths are resolved against the repo root
    root: str = MISSING  # raw dataset bundle (MANIFEST.csv, data/, ...)
    baked: str = "data/baked/floor_posture"  # baked HF dataset (gitignored)
    # extra preprocessors run in order on the loaded frames, see data/preprocess.py (unknown persons are always dropped first)
    preprocess: list[str] = field(default_factory=lambda: ["config_4"])
    # recordings left out of training and testing: the camera in these does not look top down
    exclude: list[str] = field(default_factory=lambda: ["2311__20231124_152926", "2311__20231124_153817"])


@dataclass
class AugmentConfig:
    """Training-time heatmap augmentation, see data/augment.py. Everything is off by default.

    Sized for Config_4 only (0.0237 m range bins, 256x16 heatmaps), the one sensor config used for model development.
    """

    range_shift_bins: float = 0.0  # largest random shift along range, in bins (one bin is 0.0237 m, so 13 is about 0.3 m)
    floor_jitter: float = 0.0  # std of a random smooth baseline along range, in raw heatmap units (try 50)
    time_reverse: bool = False  # play windows backwards half of the time; needs eval.label_rule=any_lying


@dataclass
class PointAugmentConfig:
    """Training-time point cloud augmentation, see data/augment.py. Everything is off by default.

    Sized for Config_4 only. The points are in the sensor frame: y is the boresight (the distance to the floor),
    and the floor plane is x-z. Dropping points and jitter are `model.pointnet.point_drop` and `jitter`.
    """

    shift_m: float = 0.0  # random move along the floor (x and z), up to this many metres each (try 0.3)
    height_m: float = 0.0  # random move along the boresight (y), up to this many metres (try 0.2)
    yaw_deg: float = 0.0  # random rotation about the boresight, up to this angle; needs eval.collapse_lying=true
    tilt_deg: float = 0.0  # random sensor tilt about x and z, up to this angle (try 5)
    mirror: bool = False  # mirror half of the windows (x -> -x); needs eval.collapse_lying=true
    ghost_points: float = 0.0  # mean number of false-alarm points added per window (try 3)
    snr_jitter_db: float = 0.0  # std of one SNR offset per window, in dB (try 2)


@dataclass
class RNNConfig:
    hidden: int = 128  # GRU hidden size
    feat_dim: int = 64  # size of the per-frame feature vector from the CNN
    dropout: float = 0.5  # on the GRU output, before the classification layer
    feature_dropout: float = 0.3  # on the per-frame CNN features
    noise: float = 0.05  # augmentation: noise std on the standardised heatmap while training
    augment: AugmentConfig = field(default_factory=AugmentConfig)  # more augmentation, applied to training batches only
    lr: float = 1e-3  # Adam learning rate
    weight_decay: float = 1e-2  # Adam weight decay (L2 penalty)
    epochs: int = 30  # upper limit; early stopping usually ends training sooner
    batch_size: int = 32  # for training and for prediction
    seed: int = 0  # torch seed, set at the start of each fit
    patience: int = 5  # stop after this many validation evaluations without improvement
    evals_per_epoch: int = 1  # validation loss evaluations per epoch; patience counts these


@dataclass
class PointNetConfig:
    width: int = 128  # features per point after the shared MLP
    dropout: float = 0.5  # classifier head
    point_dropout: float = 0.2  # per-point features, before pooling
    point_drop: float = 0.3  # augmentation: probability of dropping each point while training
    jitter: float = 0.05  # augmentation: noise std on the standardised point features while training
    lr: float = 1e-3  # Adam learning rate
    weight_decay: float = 1e-2  # Adam weight decay (L2 penalty)
    epochs: int = 30  # upper limit; early stopping usually ends training sooner
    batch_size: int = 32  # for training and for prediction
    seed: int = 0  # torch seed, set at the start of each fit
    patience: int = 5  # stop after this many validation evaluations without improvement
    evals_per_epoch: int = 1  # validation loss evaluations per epoch; patience counts these
    augment: PointAugmentConfig = field(default_factory=PointAugmentConfig)  # geometric augmentation, training batches only


@dataclass
class ModelConfig:
    name: str = "rnn"  # "rnn", "pointnet" or "baseline", see models/factory.py
    rnn: RNNConfig = field(default_factory=RNNConfig)
    pointnet: PointNetConfig = field(default_factory=PointNetConfig)


@dataclass
class TrainConfig:
    # None: off. A ratio in (0, 1] repeats training windows of the rarer classes until each reaches
    # that share of the largest class (1.0 = balanced). Training data only, see data/resample.py.
    oversample: Optional[float] = None
    # True: each fold holds out one training person for validation (early stopping, best weights).
    # False: no validation; every fold trains for exactly the model's `epochs` on all its training people.
    validation: bool = True
    # True: skip cross-validation and train one model on all people with the setting saved by the tuning run
    # (tune.best_path), then save it as final.pt. Run the tuning first. Cannot be combined with tune.grid.
    full_run: bool = False


@dataclass
class TuneConfig:
    # Nested cross-validation: a list of hyperparameter overrides of the chosen model (model.rnn or model.pointnet), e.g.
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
    target: str = "lying_annotation"  # label column to predict: the hand-made lying annotation (ground truth); "label_pose" is the original pose label
    collapse_lying: bool = True  # True: lying occurred vs. not (with any_lying), False: the original pose classes
    window_len: int = 16  # frames per window
    max_points: int = 128  # radar points kept per window by the point-cloud models (the strongest by SNR)
    label_rule: str = "any_lying"  # "any_lying": predict whether lying occurred within the window; "last": label of its last frame
    stride: int = 8  # frames between window starts
    out_dir: str = "results"  # each run gets its own folder in here
    # None: use every window. n: keep at most n windows per person (all rarer-class windows first), for quick test runs
    max_windows_per_person: Optional[int] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(root: Path, overrides: Sequence[str] = ()) -> Config:
    """The run configuration: the defaults, then `config/config.yaml`, then `key=value` overrides.

    Raises:
        ConfigKeyError, ValidationError and other `OmegaConfBaseException`: For an unknown key or a wrong type.
    """
    merged = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(root / "config" / "config.yaml"),
                             OmegaConf.from_dotlist(list(overrides)))
    return cast(Config, OmegaConf.to_object(merged))
