import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from datasets import Dataset

from data.preprocess import CONFIG_4
from config.models import ModelConfig
from config.training import RunConfig
from config.data import WindowConfig
from config.loader import check_run_config
from training.pipelines import run_cv, run_train_final, run_tune, tuned_model_config
from evaluation.records import BestSetting
from training.tuning import save_tuned

PERIOD = 0.2


def tiny_dataset(path: Path, people=("Bob", "Dave", "Frank", "Ian"), frames=48, pointcloud=False) -> None:
    """One recording per person with a short lying run, saved like the baked dataset (no heatmaps needed)."""
    rows: dict[str, list] = {k: [] for k in ("recording", "person", "frame_idx", "timestamp", "label_pose", "lying_annotation",
                                              "heatmap_ok", "num_range_bins", "num_doppler_bins", "cfg_file")}
    start = datetime(2026, 1, 1)
    if pointcloud:
        rows["pointcloud"] = []
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
            if pointcloud:
                rows["pointcloud"].append([[0.1, 2.0, 0.2 if lying else 1.2, 0.0, 20.0, 40.0, 255.0]])
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
    results = run_cv(make_cfg(tmp_path), tmp_path)
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
        run_cv(make_cfg(tmp_path), tmp_path)


def test_run_reports_too_many_folds(tmp_path):
    tiny_dataset(tmp_path / "baked")
    with pytest.raises(ValueError):
        run_cv(make_cfg(tmp_path, folds=9), tmp_path)


def test_run_limits_the_windows_per_person(tmp_path):
    tiny_dataset(tmp_path / "baked")
    results = run_cv(make_cfg(tmp_path, max_windows_per_person=3), tmp_path)
    assert results.pooled.n == 4 * 3
    assert results.config["window_coverage"]["windows_kept"] == 4 * 3


def test_run_leaves_out_the_excluded_recordings(tmp_path):
    tiny_dataset(tmp_path / "baked")
    cfg = make_cfg(tmp_path)
    cfg.selection.exclude_recordings = ["rec_Bob"]
    results = run_cv(cfg, tmp_path)
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
        results = run_cv(cfg, tmp_path)
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


@pytest.mark.parametrize("grid,match", [
    ([], "non-empty"),
    ([{"training.unknown": 1}], "unknown hyperparameter"),
    ([{"training.epochs": 0}], "epochs"),
    ([{"model.augment.jitter": -0.1}], "non-negative"),
])
def test_tuning_rejects_invalid_candidates_before_loading_data(tmp_path, monkeypatch, grid, match):
    import training.pipelines as pipelines

    cfg = make_cfg(tmp_path)
    cfg.model = ModelConfig(name="pointnet")
    cfg.tune.grid = grid
    monkeypatch.setattr(pipelines, "load_from_disk", lambda *a: pytest.fail("must validate before loading data"))
    with pytest.raises(ValueError, match=match):
        run_tune(cfg, tmp_path)


@pytest.mark.parametrize("mismatch", [False, True])
def test_final_validates_tuned_settings_before_loading_data(tmp_path, monkeypatch, mismatch):
    import training.pipelines as pipelines

    cfg = make_cfg(tmp_path)
    cfg.model = ModelConfig(name="rnn")
    cfg.tune.best_path = "tuned.yaml"
    if mismatch:
        save_tuned(tmp_path / "tuned.yaml", BEST, "pointnet", cfg.windowing.signature())
    monkeypatch.setattr(pipelines, "load_from_disk", lambda *a: pytest.fail("must validate before loading data"))
    with pytest.raises(ValueError, match="tune again" if mismatch else "run tuning first"):
        run_train_final(cfg, tmp_path)


def pointnet_cfg(tmp_path):
    cfg = make_cfg(tmp_path, folds=2)
    cfg.model = ModelConfig(name="pointnet")
    cfg.model.arch.width = 8
    cfg.model.augment.point_drop = cfg.model.augment.jitter = 0
    cfg.training.epochs = 1
    cfg.seed = 7
    cfg.tune.best_path = "tuned.yaml"
    return cfg


def test_tuning_applies_all_candidate_settings_to_inner_fits_and_outer_refits(tmp_path, monkeypatch):
    import training.fitting as fitting
    from models.model import BaseModel
    from training.tuning import load_tuned

    tiny_dataset(tmp_path / "baked", pointcloud=True)
    cfg = pointnet_cfg(tmp_path)
    cfg.tune.grid = [
        {"model.arch.width": 8, "model.augment.jitter": 0.02, "training.lr": 0.002, "training.class_weights": False},
        {"model.arch.width": 16, "model.augment.jitter": 0.04, "training.lr": 0.004, "training.class_weights": True},
    ]
    seen = []
    fit_torch = fitting.fit_torch

    def record(model, x, y, **kwargs):
        training, augment = kwargs["training"], kwargs["augment"]
        seen.append((model.width, augment.jitter, training.lr, training.class_weights,
                     training.epochs, kwargs["seed"], kwargs["val"], kwargs["val_weight_labels"]))
        return fit_torch(model, x, y, **kwargs)

    monkeypatch.setattr(fitting, "fit_torch", record)
    results = run_tune(cfg, tmp_path)
    assert len(results.folds) == 2 and results.tuned is not None and len(seen) == 10
    inner = [row for row in seen if row[6] is not None]
    refits = [row for row in seen if row[6] is None]
    assert {row[:6] for row in inner} == {(8, 0.02, 0.002, False, 1, 7), (16, 0.04, 0.004, True, 1, 7)}
    assert all(len(row[7]) == 2 * len(row[6][1]) for row in inner)  # pooled training-person labels
    assert len(refits) == 2
    out = next((tmp_path / "results").iterdir())
    assert load_tuned(tmp_path / "tuned.yaml") == load_tuned(out / "tuned.yaml")
    for row, fold in zip(refits, results.folds):
        setting = fold.tuning.setting
        assert row[:6] == (setting["model.arch.width"], setting["model.augment.jitter"],
                           setting["training.lr"], setting["training.class_weights"], fold.tuning.epochs, 7)
        restored, _ = BaseModel.load(out / fold.checkpoint)
        assert restored.width == setting["model.arch.width"]


def test_final_fits_all_windows_saves_tuned_metadata_and_bypasses_cv(tmp_path, monkeypatch):
    import training.cross_validation as cv
    import training.splitting as splitting
    from models.model import BaseModel

    tiny_dataset(tmp_path / "baked", pointcloud=True)
    cfg = pointnet_cfg(tmp_path)
    cfg.cv.folds = 99  # irrelevant to final training
    cfg.training.oversample_ratio = 1.0
    best = BestSetting({"model.arch.width": 16, "training.lr": 0.002}, 2, 1, 1, 0.5)
    save_tuned(tmp_path / "tuned.yaml", best, "pointnet", cfg.windowing.signature())
    monkeypatch.setattr(cv.CrossValidator, "cross_validate", lambda *a: pytest.fail("final training must bypass CV"))
    monkeypatch.setattr(splitting.GroupKFoldSplitter, "split", lambda *a: pytest.fail("must not split people"))
    monkeypatch.setattr(splitting.GroupKFoldSplitter, "n_splits", lambda *a: pytest.fail("must not validate folds"))
    final = run_train_final(cfg, tmp_path)
    out = next((tmp_path / "results").iterdir())
    assert out.name.endswith("_final") and final.checkpoint == "final.pt"
    assert final.n_windows == 20 and final.n_train_after_resampling > final.n_windows
    assert set(final.train_classes) == {"lying", "not_lying"}
    assert len(set(final.train_classes.values())) == 1
    assert final.history.val_steps == [] and len(final.history.train_steps) == 2
    stored = json.loads((out / "final.json").read_text())
    assert stored["config"]["training"]["epochs"] == 2
    assert stored["config"]["model"]["arch"]["width"] == 16
    assert stored["config"]["tuned"]["epochs"] == 2
    restored, windowing = BaseModel.load(out / "final.pt")
    assert restored.width == 16 and windowing == cfg.windowing.signature()
    assert not (out / "results.json").exists()
    assert cfg.model.arch.width == 8 and cfg.training.epochs == 1
