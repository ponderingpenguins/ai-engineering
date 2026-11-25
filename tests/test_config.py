from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil

import pytest
from datasets import Dataset
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from config import DataConfig, ModelConfig, RunConfig, WindowConfig, load_data_config, load_run_config
from data import extract
from data.annotation import Marks, write_annotations
from data.preprocess import DropUnknownPerson, RangeNormalise, TemporalBackground, configured_pipeline
from models.factory import build_model
from utils.tuning import with_setting

REPO = Path(__file__).resolve().parents[1]


def test_split_files_list_every_default_without_duplicate_training_settings():
    assert OmegaConf.to_container(OmegaConf.load(REPO / "config/data.yaml")) == asdict(DataConfig())
    train = OmegaConf.to_container(OmegaConf.load(REPO / "config/train.yaml"))
    assert train.pop("model") == "rnn"
    expected = asdict(RunConfig())
    for key in (*asdict(DataConfig()), "model"):
        expected.pop(key)
    assert train == expected
    for name in ("rnn", "pointnet", "baseline"):
        cfg = load_run_config(REPO, [f"model={name}"])
        assert cfg.model.name == name
        if name != "baseline":
            assert set(OmegaConf.load(REPO / f"config/models/{name}.yaml")) == {"arch", "augment"}
    assert not (REPO / "config/config.yaml").exists()


def test_data_tools_do_not_need_training_or_model_files(tmp_path):
    (tmp_path / "config").mkdir()
    shutil.copyfile(REPO / "config/data.yaml", tmp_path / "config/data.yaml")
    cfg = load_data_config(tmp_path, ["windowing.window_len=8", "paths.baked_dir=elsewhere"])
    assert cfg.windowing.window_len == 8 and cfg.paths.baked_dir == "elsewhere"
    assert not hasattr(cfg, "model") and not hasattr(cfg, "training")


def test_cli_overrides_win_over_model_and_training_yaml():
    cfg = load_run_config(REPO, ["model=pointnet", "model.arch.width=32", "training.lr=0.02", "seed=12"])
    assert cfg.model.arch.width == 32 and cfg.training.lr == 0.02 and cfg.seed == 12
    assert not hasattr(cfg.model.arch, "hidden")


@pytest.mark.parametrize("override", [
    "model.augment.shift_m=0.3", "model.arch.width=32", "training.epochs=nope",
    "training.imbalance.class_weights=false", "data.bake.overwrite=true", "model.rnn.hidden=32",
    "selection.require_known_person=false", "windowing.unknown=1",
])
def test_unknown_wrong_modality_or_wrong_type_overrides_are_rejected(override):
    with pytest.raises(OmegaConfBaseException):
        load_run_config(REPO, [override])


@pytest.mark.parametrize("override", [
    "windowing.window_len=0", "windowing.stride=0", "windowing.max_points=0", "cv.folds=1",
    "training.lr=0", "training.batch_size=0", "training.oversample_ratio=1.1",
    "denoising.range_normalise.smooth_bins=2", "denoising.temporal_background.tau_frames=0",
    "model.augment.noise=-1",
])
def test_invalid_values_fail_during_loading(override):
    with pytest.raises(ValueError):
        load_run_config(REPO, [override])


def test_model_selector_cannot_bypass_the_selected_schema():
    with pytest.raises(ValueError, match="model=<name>"):
        load_run_config(REPO, ["model.name=pointnet"])
    with pytest.raises(OmegaConfBaseException):
        load_run_config(REPO, ["model=baseline", "model.augment.noise=0.2"])


@pytest.mark.parametrize("normalise,jitter", [(False, 0), (False, 50), (True, 0), (True, 50)])
def test_range_normalisation_and_floor_jitter_cannot_be_enabled_together(normalise, jitter):
    overrides = [f"denoising.range_normalise.enabled={str(normalise).lower()}",
                 f"model.augment.floor_jitter={jitter}"]
    if normalise and jitter > 0:
        with pytest.raises(ValueError, match="floor_jitter requires denoising.range_normalise.enabled=false"):
            load_run_config(REPO, overrides)
    else:
        cfg = load_run_config(REPO, overrides)
        assert cfg.denoising.range_normalise.enabled == normalise and cfg.model.augment.floor_jitter == jitter


def test_range_normalisation_guard_does_not_block_pointnet():
    cfg = load_run_config(REPO, ["model=pointnet", "denoising.range_normalise.enabled=true"])
    assert cfg.denoising.range_normalise.enabled


def test_existing_preprocessing_keeps_its_steps_order_and_parameters():
    cfg = load_data_config(REPO, ["selection.sensor_config=null", "selection.exclude_recordings=[]",
        "denoising.range_normalise.enabled=true", "denoising.range_normalise.percentile=20",
        "denoising.range_normalise.smooth_bins=9", "denoising.temporal_background.enabled=true",
        "denoising.temporal_background.tau_frames=50"])
    pipeline = configured_pipeline(cfg.selection, cfg.denoising)
    assert len(pipeline.steps) == 3 and isinstance(pipeline.steps[0], DropUnknownPerson)
    rn, tb = pipeline.steps[1:]
    assert isinstance(rn, RangeNormalise) and (rn.percentile, rn.smooth_bins) == (20, 9)
    assert isinstance(tb, TemporalBackground) and tb.tau_frames == 50
    assert len(configured_pipeline(cfg.selection).steps) == 1


@pytest.mark.parametrize("name", ["rnn", "pointnet"])
def test_shared_training_options_and_selected_augmentations_reach_model(name):
    cfg = load_run_config(REPO, [f"model={name}", "training.epochs=2", "training.class_weights=false", "seed=5"])
    model = build_model(cfg.model, cfg.training, seed=cfg.seed)
    assert model.epochs == 2 and model.seed == 5 and not model.class_weights
    if name == "rnn":
        assert model.noise == 0.05 and model.augment is None
    else:
        assert model.point_drop == 0.3 and model.jitter == 0.05 and model.augment is None


def test_dotted_tuning_settings_are_typed_and_do_not_mutate_the_original():
    cfg = load_run_config(REPO, ["model=pointnet"])
    tuned = with_setting(cfg, {"model.arch.width": 64, "model.augment.jitter": 0.02, "training.lr": 0.005})
    assert tuned.model.arch.width == 64 and tuned.model.augment.jitter == 0.02 and tuned.training.lr == 0.005
    assert cfg.model.arch.width == 128 and cfg.model.augment.jitter == 0.05
    with pytest.raises(OmegaConfBaseException):
        with_setting(cfg, {"training.lr": "not-a-number"})


def test_bake_execution_options_are_flags_not_yaml(monkeypatch):
    seen = []
    monkeypatch.setattr(extract, "extract", lambda cfg, **options: seen.append((cfg, options)))
    monkeypatch.setattr("sys.argv", ["extract", "--num-proc", "1", "--overwrite", "--recording", "example"])
    extract.main()
    cfg, options = seen[0]
    assert isinstance(cfg, DataConfig) and not hasattr(cfg, "bake")
    assert options == {"num_proc": 1, "overwrite": True, "recordings": ["example"]}


def test_annotation_join_reads_a_configured_input_directory(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "person_map.csv").write_text("recording,person\na,NewPerson\n")
    (annotations / "presence_events.csv").write_text("recording,frame_idx,event\na,0,enter\na,1,leave\n")
    write_annotations({"a": Marks(((1, 1),), True)}, annotations / "lying_intervals.csv")
    ds = Dataset.from_dict({"recording": ["a", "a"], "frame_idx": [0, 1],
                            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 0, 1)]})
    ds = extract._add_person(ds, annotations)
    ds = extract._add_presence(ds, annotations)
    ds = extract._add_lying_annotation(ds, annotations)
    assert ds["person"] == ["NewPerson", "NewPerson"]
    assert ds["person_present"] == [True, False]
    assert ds["lying_annotation"] == ["not_lying", "lying"]


def test_annotation_tool_uses_configured_annotations_directory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from annotation_tool import __main__ as tool

    cfg = DataConfig()
    cfg.paths.baked_dir = str(tmp_path / "baked")
    cfg.paths.annotations_dir = str(tmp_path / "custom_annotations")
    Path(cfg.paths.baked_dir).mkdir()
    seen = []

    def stop():
        raise KeyboardInterrupt

    server = SimpleNamespace(server_port=8765, serve_forever=stop, server_close=lambda: None)
    monkeypatch.setattr(tool, "load_data_config", lambda *args: cfg)
    monkeypatch.setattr(tool, "load_from_disk", lambda path: object())
    monkeypatch.setattr(tool, "configured_pipeline", lambda selection: lambda ds: ds)
    monkeypatch.setattr(tool, "Catalogue", lambda ds: SimpleNamespace(names=["a"]))
    monkeypatch.setattr(tool, "Annotator", lambda catalogue, path: seen.append(path))
    monkeypatch.setattr(tool, "make_server", lambda *args: server)
    monkeypatch.setattr("sys.argv", ["annotation_tool", "--no-browser"])
    tool.main()
    assert seen == [tmp_path / "custom_annotations/lying_intervals.csv"]
