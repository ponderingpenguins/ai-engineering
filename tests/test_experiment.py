import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from datasets import Dataset

from data.preprocess import CONFIG_4
from utils.config import Config, EvalConfig
from utils.experiment import check_config, run, tuned_model_config
from utils.results import BestSetting
from utils.task import Task
from utils.tuning import save_tuned

PERIOD = 0.2


def tiny_dataset(path: Path, people=("Bob", "Dave", "Frank", "Ian"), frames=48) -> None:
    """One recording per person with a short lying run, saved like the baked dataset (no heatmaps needed)."""
    rows: dict[str, list] = {k: [] for k in ("recording", "person", "frame_idx", "timestamp", "label_pose",
                                              "heatmap_ok", "num_range_bins", "num_doppler_bins", "cfg_file")}
    start = datetime(2026, 1, 1)
    for person in people:
        for i in range(frames):
            rows["recording"].append(f"rec_{person}")
            rows["person"].append(person)
            rows["frame_idx"].append(i)
            rows["timestamp"].append(start + timedelta(seconds=PERIOD * i))
            rows["label_pose"].append("lying_left" if 20 <= i < 22 else "standing")
            rows["heatmap_ok"].append(True)
            rows["num_range_bins"].append(256)
            rows["num_doppler_bins"].append(16)
            rows["cfg_file"].append(CONFIG_4)
    Dataset.from_dict(rows).save_to_disk(str(path))


def make_cfg(tmp_path: Path, **eval_overrides) -> Config:
    cfg = Config()
    cfg.data.root = "unused"
    cfg.data.baked = str(tmp_path / "baked")
    cfg.model.name = "baseline"
    cfg.eval.out_dir = str(tmp_path / "results")
    for key, value in eval_overrides.items():
        setattr(cfg.eval, key, value)
    return cfg


# ---- check_config ----------------------------------------------------------------------------------

def test_the_default_config_is_valid():
    check_config(Config())


def test_unknown_model_is_refused():
    cfg = Config()
    cfg.model.name = "nope"
    with pytest.raises(ValueError, match="model.name must be one of"):
        check_config(cfg)


def test_augmentation_must_match_the_chosen_model():
    cfg = Config()
    cfg.model.pointnet.augment.shift_m = 0.3  # the model is the rnn
    with pytest.raises(ValueError, match="only applies to model.name=pointnet"):
        check_config(cfg)
    cfg.model.name = "pointnet"
    check_config(cfg)
    cfg.data.preprocess = []
    with pytest.raises(ValueError, match="Config_4"):
        check_config(cfg)


def test_label_dependent_augmentations_need_the_right_task():
    cfg = Config()
    cfg.model.rnn.augment.time_reverse = True
    cfg.eval.label_rule = "last"
    with pytest.raises(ValueError, match="any_lying"):
        check_config(cfg)
    cfg = Config()
    cfg.model.name = "pointnet"
    cfg.model.pointnet.augment.mirror = True
    cfg.eval.collapse_lying = False
    with pytest.raises(ValueError, match="collapse_lying"):
        check_config(cfg)


# ---- tuned_model_config ----------------------------------------------------------------------------

TASK = Task.from_eval(Config().eval)
BEST = BestSetting({"hidden": 16}, 3, 1, 1, 0.5)


def test_tuned_config_needs_a_tuned_file_and_no_grid(tmp_path):
    cfg = Config()
    cfg.tune.best_path = "tuned.yaml"
    with pytest.raises(ValueError, match="run the tuning first"):
        tuned_model_config(cfg, TASK, tmp_path)
    cfg.tune.grid = [{"hidden": 8}]
    with pytest.raises(ValueError, match="cannot be combined"):
        tuned_model_config(cfg, TASK, tmp_path)


def test_tuned_config_applies_the_saved_setting_and_checks_the_task(tmp_path):
    cfg = Config()
    cfg.tune.best_path = "tuned.yaml"
    save_tuned(tmp_path / "tuned.yaml", BEST, "rnn", TASK)
    model_cfg, tuned = tuned_model_config(cfg, TASK, tmp_path)
    assert model_cfg.rnn.hidden == 16 and model_cfg.rnn.epochs == 3 and tuned["wins"] == 1
    with pytest.raises(ValueError, match="tune again"):
        tuned_model_config(cfg, Task.from_eval(EvalConfig(window_len=8)), tmp_path)


def test_the_baseline_has_no_tuned_setting(tmp_path):
    cfg = Config()
    cfg.model.name = "baseline"
    cfg.train.full_run = True
    with pytest.raises(ValueError, match="only defined for"):
        tuned_model_config(cfg, TASK, tmp_path)


# ---- run -------------------------------------------------------------------------------------------

def test_run_cross_validates_and_writes_the_results(tmp_path):
    tiny_dataset(tmp_path / "baked")
    results = run(make_cfg(tmp_path), tmp_path)
    assert len(results.folds) == 4 and results.pooled.n > 0
    out = next((tmp_path / "results").iterdir())
    assert json.loads((out / "results.json").read_text())["config"]["input_type"] == "none"
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


def test_run_leaves_out_the_excluded_recordings(tmp_path):
    tiny_dataset(tmp_path / "baked")
    cfg = make_cfg(tmp_path)
    cfg.data.exclude = ["rec_Bob"]
    results = run(cfg, tmp_path)
    assert len(results.folds) == 3  # one fold per remaining person
    assert all("Bob" not in fold.test_groups for fold in results.folds)


def test_the_report_names_the_target_and_the_recordings_it_used(tmp_path, caplog):
    tiny_dataset(tmp_path / "baked")
    ds = Dataset.load_from_disk(str(tmp_path / "baked"))
    # a second target in which only three of the four recordings are labelled, like a partly reviewed track
    labels = [None if rec == "rec_Ian" else "not_lying" for rec in ds["recording"]]
    ds.add_column("lying_annotation", labels).save_to_disk(str(tmp_path / "with_annotation"))
    cfg = make_cfg(tmp_path)
    cfg.data.baked = str(tmp_path / "with_annotation")
    cfg.eval.target = "lying_annotation"
    with caplog.at_level("WARNING"):
        results = run(cfg, tmp_path)
    assert results.config["recordings_used"] == ["rec_Bob", "rec_Dave", "rec_Frank"]
    assert results.config["recordings_unlabelled"] == ["rec_Ian"]
    assert "rec_Ian" in caplog.text and "not used" in caplog.text
    report = (next((tmp_path / "results").iterdir()) / "report.md").read_text()
    assert "Target: `lying_annotation`. 3 recordings used" in report
    assert "no labels for this target: rec_Ian" in report
