import pytest
from datasets import Dataset

from data.preprocess import CONFIG_4, DropUnknownPerson, KeepConfig4, build_pipeline
from data.extract import NO_PERSON


@pytest.fixture
def ds():
    return Dataset.from_dict({
        "recording": ["r1", "r2", "r3", "r4"],
        "cfg_file": [CONFIG_4, CONFIG_4, "Config_3.cfg", CONFIG_4],
        "person": ["Bob", NO_PERSON, "Dave", "Ian"],
        "heatmap": [[0.0], [0.0], [0.0], [0.0]],
    })


def test_keep_config_4(ds):
    assert KeepConfig4()(ds)["recording"] == ["r1", "r2", "r4"]


def test_drop_unknown_person(ds):
    assert DropUnknownPerson()(ds)["recording"] == ["r1", "r3", "r4"]


def test_pipeline_applies_steps_in_order_and_keeps_other_columns(ds):
    out = build_pipeline(["config_4", "known_person"])(ds)
    assert out["recording"] == ["r1", "r4"]
    assert out.column_names == ds.column_names


def test_empty_pipeline_only_drops_unknown_persons(ds):
    assert build_pipeline([])(ds)["recording"] == ["r1", "r3", "r4"]


def test_unknown_step_name_raises():
    with pytest.raises(ValueError, match="unknown preprocessor"):
        build_pipeline(["nope"])


def test_unknown_persons_are_dropped_even_if_not_requested(ds):
    out = build_pipeline(["config_4"])(ds)
    assert "NA" not in out["person"]
    assert "NA" not in build_pipeline([])(ds)["person"]
