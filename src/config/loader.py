"""Typed data/run loaders; data tools never load training or model files."""
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf

from .data import DataConfig
from .models import MODEL_SCHEMAS, ModelConfig
from .training import RunConfig


def load_data_config(root: Path, overrides: Sequence[str] = ()) -> DataConfig:
    merged = OmegaConf.merge(OmegaConf.structured(DataConfig), OmegaConf.load(root / "config/data.yaml"),
                             OmegaConf.from_dotlist(list(overrides)))
    cfg = cast(DataConfig, OmegaConf.to_object(merged))
    check_data_config(cfg)
    return cfg


def load_run_config(root: Path, overrides: Sequence[str] = ()) -> RunConfig:
    training = OmegaConf.to_container(OmegaConf.load(root / "config/train.yaml"), resolve=True)
    assert isinstance(training, dict)
    name = training.pop("model", "rnn")
    remaining = []
    for override in overrides:
        key, separator, value = override.partition("=")
        if key == "model.name":
            raise ValueError("select the model with model=<name>, not model.name")
        if key == "model" and separator:
            name = OmegaConf.from_dotlist([override]).model
        else:
            remaining.append(override)
    if name not in MODEL_SCHEMAS:
        raise ValueError(f"model must be one of {tuple(MODEL_SCHEMAS)}, got {name!r}")
    schema = OmegaConf.structured(RunConfig)
    schema.model = OmegaConf.structured(ModelConfig(name=str(name)))
    OmegaConf.set_struct(schema, True)
    merged = OmegaConf.merge(schema, OmegaConf.load(root / "config/data.yaml"), training,
                             {"model": OmegaConf.load(root / f"config/models/{name}.yaml")},
                             OmegaConf.from_dotlist(remaining))
    cfg = cast(RunConfig, OmegaConf.to_object(merged))
    check_run_config(cfg)
    return cfg


def check_data_config(cfg: DataConfig) -> None:
    if cfg.windowing.label_rule not in ("any_lying", "last"):
        raise ValueError("windowing.label_rule must be any_lying or last")
    for name in ("window_len", "stride", "max_points"):
        if getattr(cfg.windowing, name) < 1:
            raise ValueError(f"windowing.{name} must be positive")
    rn, tb = cfg.denoising.range_normalise, cfg.denoising.temporal_background
    if not 0 <= rn.percentile <= 100 or rn.smooth_bins < 1 or rn.smooth_bins % 2 != 1:
        raise ValueError("denoising.range_normalise needs percentile in [0, 100] and positive odd smooth_bins")
    if tb.tau_frames < 1:
        raise ValueError("denoising.temporal_background.tau_frames must be at least 1")


def check_run_config(cfg: RunConfig) -> None:
    check_data_config(cfg)
    if cfg.model.name not in MODEL_SCHEMAS:
        raise ValueError(f"model must be one of {tuple(MODEL_SCHEMAS)}, got {cfg.model.name!r}")
    if cfg.cv.folds is not None and cfg.cv.folds < 2:
        raise ValueError("cv.folds must be at least 2 or null")
    if cfg.cv.max_windows_per_person is not None and cfg.cv.max_windows_per_person < 1:
        raise ValueError("cv.max_windows_per_person must be positive or null")
    for name in ("epochs", "batch_size", "patience", "evals_per_epoch"):
        if getattr(cfg.training, name) < 1:
            raise ValueError(f"training.{name} must be positive")
    if cfg.training.lr <= 0 or cfg.training.weight_decay < 0:
        raise ValueError("training.lr must be positive and weight_decay non-negative")
    ratio = cfg.training.oversample_ratio
    if ratio is not None and not 0 < ratio <= 1:
        raise ValueError("training.oversample_ratio must be in (0, 1] or null")
    aug = asdict(cfg.model.augment) if cfg.model.name != "baseline" else {}
    if any(value < 0 for value in aug.values() if not isinstance(value, bool)):
        raise ValueError("model.augment sizes must be non-negative")
    if cfg.model.name == "rnn":
        if cfg.denoising.range_normalise.enabled and aug["floor_jitter"] > 0:
            raise ValueError("model.augment.floor_jitter requires denoising.range_normalise.enabled=false "
                             "because range normalisation removes the floor that jitter varies")
        geometric = aug["range_shift_bins"] or aug["floor_jitter"] or aug["time_reverse"]
        if aug["time_reverse"] and cfg.windowing.label_rule != "any_lying":
            raise ValueError("model.augment.time_reverse needs windowing.label_rule=any_lying")
    elif cfg.model.name == "pointnet":
        geometric = any(v for k, v in aug.items() if k not in ("point_drop", "jitter"))
        if aug["point_drop"] > 1:
            raise ValueError("model.augment.point_drop must be in [0, 1]")
        if (aug["yaw_deg"] or aug["mirror"]) and not cfg.windowing.collapse_lying:
            raise ValueError("model.augment.yaw_deg and mirror need windowing.collapse_lying=true")
    else:
        geometric = False
    if geometric and cfg.selection.sensor_config != "Config_4.cfg":
        raise ValueError("geometric augmentation sizes are for Config_4; set selection.sensor_config=Config_4.cfg")
