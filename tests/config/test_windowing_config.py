from dataclasses import replace
import numpy as np
import pytest
import torch
from config.data import WindowConfig
from models.baseline import MajorityClass
from models.model import BaseModel


def test_signature_omits_stride_without_modifying_config():
    cfg = WindowConfig()
    assert cfg.signature() == replace(cfg, stride=1).signature()
    assert cfg.signature() != replace(cfg, window_len=8).signature()
    assert cfg.stride == 8 and "stride" not in cfg.signature()


def test_new_checkpoints_store_plain_windowing_settings(tmp_path):
    model = MajorityClass()
    model.fit(np.zeros((3, 1)), np.array(["lying", "lying", "not_lying"]))
    signature = WindowConfig().signature()
    path = tmp_path / "model.pt"
    model.save(path, signature)
    stored = torch.load(path, weights_only=True)
    assert stored["format_version"] == 3 and stored["windowing"] == signature
    assert "window_spec" not in stored
    restored, values = BaseModel.load(path)
    assert values == signature and isinstance(values, dict)
    assert restored.predict(np.zeros((1, 1))).tolist() == ["lying"]


@pytest.mark.parametrize("version,key", [(1, "evaluation"), (2, "window_spec")])
def test_previous_checkpoint_versions_still_load(tmp_path, version, key):
    model = MajorityClass()
    model.fit(np.zeros((3, 1)), np.array(["lying", "lying", "not_lying"]))
    signature = WindowConfig().signature()
    path = tmp_path / "legacy.pt"
    torch.save({**model._checkpoint_state(), "format_version": version,
                "model": "baseline", key: signature}, path)
    restored, values = BaseModel.load(path)
    assert values == signature and restored.predict(np.zeros((1, 1))).tolist() == ["lying"]


@pytest.mark.parametrize("values", [None, [], {"target": "label_pose"}])
def test_malformed_saved_windowing_settings_are_rejected(tmp_path, values):
    path = tmp_path / "invalid.pt"
    torch.save({"format_version": 3, "model": "baseline", "windowing": values}, path)
    with pytest.raises(ValueError, match="windowing settings"):
        BaseModel.load(path)
