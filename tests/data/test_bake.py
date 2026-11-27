from datetime import datetime, timedelta
from pathlib import Path

import pytest
from datasets import Dataset, load_from_disk

from config.data import DataConfig
from data import bake
from data.annotations import Marks, attach_annotations, write_annotations


@pytest.fixture
def baked(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "person_map.csv").write_text("recording,person\na,OldPerson\n")
    (annotations / "presence_events.csv").write_text("recording,frame_idx,event\na,0,enter\n")
    write_annotations({"a": Marks((), True)}, annotations / "lying_intervals.csv")
    ds = Dataset.from_dict({
        "recording": ["a", "a", "a", "b"],
        "frame_idx": [0, 1, 2, 0],
        "timestamp": [datetime(2026, 1, 1) + timedelta(seconds=i) for i in range(4)],
        "radar_payload": [b"\x00\x01", b"\x02\x03", b"\x04\x05", b"\x06\x07"],
        "heatmap": [[[i, i + 1]] for i in range(4)],
        "pointcloud": [[[1.0, 2.0, 3.0]]] * 4,
        "label_pose": ["standing", "sitting", "lying", None],
    })
    cfg = DataConfig()
    cfg.paths.baked_dir = str(tmp_path / "baked")
    cfg.paths.annotations_dir = str(annotations)
    cfg.paths.raw_root = str(tmp_path / "raw_does_not_exist")
    attach_annotations(ds, annotations).save_to_disk(cfg.paths.baked_dir)
    return cfg, ds.to_dict(), ds.features


def test_refresh_round_trip_preserves_raw_columns_and_uses_updated_annotations(baked, monkeypatch):
    cfg, raw, raw_features = baked
    annotations = Path(cfg.paths.annotations_dir)
    (annotations / "person_map.csv").write_text("recording,person\na,NewPerson\n")
    (annotations / "presence_events.csv").write_text("recording,frame_idx,event\na,1,enter\na,2,leave\n")
    write_annotations({"a": Marks(((1, 1),), True)}, annotations / "lying_intervals.csv")
    monkeypatch.setattr(bake, "_recordings", lambda *_: pytest.fail("must not scan raw recordings"))
    monkeypatch.setattr(bake, "_rows", lambda *_: pytest.fail("must not parse raw recordings"))

    bake.refresh_annotations(cfg)
    bake.refresh_annotations(cfg)  # re-running refresh must replace existing annotation columns
    out = load_from_disk(cfg.paths.baked_dir)
    assert out.select_columns(list(raw)).to_dict() == raw
    assert out.select_columns(list(raw)).features == raw_features
    assert out["person"] == ["NewPerson"] * 3 + ["NA"]
    assert out["person_present"] == [False, True, False, None]
    assert out["lying_annotation"] == ["not_lying", "lying", "not_lying", None]
    assert out.features["person_present"].dtype == "bool"
    assert out.features["lying_annotation"].dtype == "string"
    assert {path.name for path in Path(cfg.paths.baked_dir).parent.iterdir()} == {"annotations", "baked"}


def test_refresh_save_failure_preserves_original_and_cleans_partial_copy(baked, monkeypatch):
    cfg, _, _ = baked
    original = load_from_disk(cfg.paths.baked_dir).to_dict()

    def fail_save(self, path):
        Path(path).mkdir()
        (Path(path) / "partial").write_text("incomplete")
        raise OSError("disk full")

    monkeypatch.setattr(Dataset, "save_to_disk", fail_save)
    with pytest.raises(OSError, match="disk full"):
        bake.refresh_annotations(cfg)
    assert load_from_disk(cfg.paths.baked_dir).to_dict() == original
    assert not (Path(cfg.paths.baked_dir).parent / ".baked.refresh_tmp").exists()
    assert not (Path(cfg.paths.baked_dir).parent / ".baked.old").exists()


def test_refresh_swap_failure_restores_original(baked, monkeypatch):
    cfg, _, _ = baked
    original = load_from_disk(cfg.paths.baked_dir).to_dict()
    rename = Path.rename

    def fail_install(self, target):
        if self.name == ".baked.refresh_tmp":
            raise PermissionError("cannot install refreshed dataset")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_install)
    with pytest.raises(PermissionError, match="cannot install"):
        bake.refresh_annotations(cfg)
    assert load_from_disk(cfg.paths.baked_dir).to_dict() == original
    assert not (Path(cfg.paths.baked_dir).parent / ".baked.old").exists()
    assert (Path(cfg.paths.baked_dir).parent / ".baked.refresh_tmp").is_dir()


@pytest.mark.parametrize("suffix", ["refresh_tmp", "old"])
def test_refresh_preserves_existing_recovery_folders(baked, suffix):
    cfg, _, _ = baked
    recovery = Path(cfg.paths.baked_dir).parent / f".baked.{suffix}"
    recovery.mkdir()
    marker = recovery / "keep"
    marker.write_text("recovery data")
    with pytest.raises(FileExistsError, match="recovery folder already exists"):
        bake.refresh_annotations(cfg)
    assert marker.read_text() == "recovery data"


def test_refresh_requires_an_existing_bake(tmp_path):
    cfg = DataConfig()
    cfg.paths.baked_dir = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="run bake first"):
        bake.refresh_annotations(cfg)
