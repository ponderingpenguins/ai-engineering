import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from datasets import Dataset

from data.preprocess import CONFIG_4
from config import ModelConfig, RunConfig, WindowConfig, check_run_config
from utils.experiment import run, tuned_model_config
from utils.results import BestSetting
from utils.tuning import save_tuned

PERIOD = 0.2


def tiny_dataset(path: Path, people=("Bob", "Dave", "Frank", "Ian"), frames=48) -> None:
    """One recording per person with a short lying run, saved like the baked dataset (no heatmaps needed)."""
    rows: dict[str, list] = {k: [] for k in ("recording", "person", "frame_idx", "timestamp", "label_pose", "lying_annotation",
                                              "heatmap_ok", "num_range_bins", "num_doppler_bins", "cfg_file")}
    start = datetime(2026, 1, 1)
    for person in people:
        for i in range(frames):
            rows["recording"].append(f"rec_{person}")
            rows["person"].append(person)
            rows["frame_idx"].append(i)
            rows["timestamp"].append(start + timedelta(seconds=PERIOD * i))
            lying = 20 <= i < 22
            rows["label_pose"].append("lying_left" if lying else "standing")
            rows["lying_annotation"].append("lying" if lying else "not_lying")
            rows["heatmap_ok"].append(True)
            rows["num_range_bins"].append(256)
            rows["num_doppler_bins"].append(16)
            rows["cfg_file"].append(CONFIG_4)
    Dataset.from_dict(rows).save_to_disk(str(path))


def make_cfg(tmp_path: Path, **eval_overrides) -> RunConfig:
    cfg = RunConfig()
    cfg.paths.raw_root = "unused"
    cfg.paths.baked_dir = str(tmp_path / "baked")
    cfg.model.name = "baseline"
    cfg.out_dir = str(tmp_path / "results")
    for key, value in eval_overrides.items():
        setattr(cfg.cv, key, value)
    return cfg


# ---- check_run_config ----------------------------------------------------------------------------------

def test_the_default_config_is_valid():
    check_run_config(RunConfig())


def test_unknown_model_is_refused():
    cfg = RunConfig()
    cfg.model.name = "nope"
    with pytest.raises(ValueError, match="model must be one of"):
        check_run_config(cfg)


def test_geometric_augmentation_needs_config_4_when_its_modality_is_selected():
    cfg = RunConfig()
    cfg.model = ModelConfig(name="pointnet")
    cfg.model.augment.shift_m = 0.3
    check_run_config(cfg)
    cfg.selection.sensor_config = None
    with pytest.raises(ValueError, match="Config_4"):
        check_run_config(cfg)


def test_label_dependent_augmentations_need_matching_window_labels():
    cfg = RunConfig()
    cfg.model.augment.time_reverse = True
    cfg.windowing.label_rule = "last"
    with pytest.raises(ValueError, match="any_lying"):
        check_run_config(cfg)
    cfg = RunConfig()
    cfg.model = ModelConfig(name="pointnet")
    cfg.model.augment.mirror = True
    cfg.windowing.collapse_lying = False
    with pytest.raises(ValueError, match="collapse_lying"):
        check_run_config(cfg)


# ---- tuned_model_config ----------------------------------------------------------------------------

WINDOWING = RunConfig().windowing.signature()
BEST = BestSetting({"model.arch.hidden": 16}, 3, 1, 1, 0.5)


def test_tuned_config_needs_a_tuned_file_and_no_grid(tmp_path):
    cfg = RunConfig()
    cfg.tune.best_path = "tuned.yaml"
    with pytest.raises(ValueError, match="run tuning first"):
        tuned_model_config(cfg, tmp_path)


def test_tuned_config_applies_the_saved_setting_and_checks_windowing(tmp_path):
    cfg = RunConfig()
    cfg.tune.best_path = "tuned.yaml"
    save_tuned(tmp_path / "tuned.yaml", BEST, "rnn", WINDOWING)
    model_cfg, tuned = tuned_model_config(cfg, tmp_path)
    assert model_cfg.model.arch.hidden == 16 and model_cfg.training.epochs == 3 and tuned["wins"] == 1
    cfg.windowing.window_len = 8
    with pytest.raises(ValueError, match="tune again"):
        tuned_model_config(cfg, tmp_path)


def test_the_baseline_has_no_tuned_setting(tmp_path):
    cfg = RunConfig()
    cfg.model.name = "baseline"
    # final mode is selected by the command
    with pytest.raises(ValueError, match="only defined for"):
        tuned_model_config(cfg, tmp_path)


# ---- run -------------------------------------------------------------------------------------------

def test_run_cross_validates_and_writes_the_results(tmp_path):
    tiny_dataset(tmp_path / "baked")
    results = run(make_cfg(tmp_path), tmp_path)
    assert len(results.folds) == 4 and results.pooled.n > 0
    out = next((tmp_path / "results").iterdir())
    config = json.loads((out / "results.json").read_text())["config"]
    assert config["input_type"] == "none"
    assert config["window_coverage"]["frames_total"] == config["window_coverage"]["frames_kept"] == 192
    assert config["window_coverage"]["windows_kept"] == results.pooled.n
    assert config["window_coverage"]["windows_gap_skipped"] == 0
    assert (out / "report.md").is_file()


def test_run_reports_a_missing_dataset(tmp_path):
    with pytest.raises(ValueError, match="No baked dataset"):
        run(make_cfg(tmp_path), tmp_path)


def test_run_reports_too_many_folds(tmp_path):
    tiny_dataset(tmp_path / "baked")
    with pytest.raises(ValueError):
        run(make_cfg(tmp_path, folds=9), tmp_path)


def test_run_limits_the_windows_per_person(tmp_path):
    tiny_dataset(tmp_path / "baked")
    results = run(make_cfg(tmp_path, max_windows_per_person=2), tmp_path)
    assert results.pooled.n == 4 * 2
    assert results.config["window_coverage"]["windows_kept"] == 4 * 2


def test_run_leaves_out_the_excluded_recordings(tmp_path):
    tiny_dataset(tmp_path / "baked")
    cfg = make_cfg(tmp_path)
    cfg.selection.exclude_recordings = ["rec_Bob"]
    results = run(cfg, tmp_path)
    assert len(results.folds) == 3  # one fold per remaining person
    assert all("Bob" not in fold.test_groups for fold in results.folds)


def test_the_report_names_the_target_and_the_recordings_it_used(tmp_path, caplog):
    tiny_dataset(tmp_path / "baked")
    ds = Dataset.load_from_disk(str(tmp_path / "baked"))
    # a second target in which only three of the four recordings are labelled, like a partly reviewed track
    labels = [None if rec == "rec_Ian" else "not_lying" for rec in ds["recording"]]
    ds.remove_columns("lying_annotation").add_column("lying_annotation", labels).save_to_disk(str(tmp_path / "with_annotation"))
    cfg = make_cfg(tmp_path)
    cfg.paths.baked_dir = str(tmp_path / "with_annotation")
    cfg.windowing.target = "lying_annotation"
    with caplog.at_level("WARNING"):
        results = run(cfg, tmp_path)
    assert results.config["recordings_used"] == ["rec_Bob", "rec_Dave", "rec_Frank"]
    assert results.config["recordings_unlabelled"] == ["rec_Ian"]
    coverage = results.config["window_coverage"]
    assert coverage["recordings_used"] == results.config["recordings_used"]
    assert coverage["recordings_unlabelled"] == ["rec_Ian"]
    assert coverage["frames_total"] == 192 and coverage["frames_unlabelled"] == 48
    assert coverage["frames_kept"] == 144
    assert "rec_Ian" in caplog.text and "not used" in caplog.text
    report = (next((tmp_path / "results").iterdir()) / "report.md").read_text()
    assert "Target: `lying_annotation`. 3 recordings used" in report
    assert "no labels for this target: rec_Ian" in report
