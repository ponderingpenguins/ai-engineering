import sys

import pytest

import main as cli
from data import bake


@pytest.mark.parametrize("mode", ["cv", "tune", "train-final"])
def test_explicit_modes_dispatch_independently_of_grid_contents(mode, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "run", lambda cfg, root, command: seen.append((cfg, command)))
    monkeypatch.setattr(sys, "argv", ["main.py", mode, "model=rnn",
                                     'tune.grid=[{"model.arch.hidden":16}]'])
    cli.main()
    cfg, command = seen[0]
    assert command == mode and cfg.tune.grid == [{"model.arch.hidden": 16}]
    assert not hasattr(cfg.training, "full_run")


def test_tune_without_candidates_fails_before_running(monkeypatch):
    monkeypatch.setattr(cli, "run", lambda *args: pytest.fail("must not start an empty tuning run"))
    monkeypatch.setattr(sys, "argv", ["main.py", "tune"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_bake_forwards_execution_flags_and_data_overrides(monkeypatch):
    seen = []
    monkeypatch.setattr(bake, "bake_dataset", lambda cfg, **options: seen.append((cfg, options)))
    monkeypatch.setattr(sys, "argv", ["main.py", "bake", "--overwrite", "--num-proc", "2",
                                     "--recording", "one", "--recording", "two", "paths.baked_dir=temporary"])
    cli.main()
    cfg, options = seen[0]
    assert cfg.paths.baked_dir == "temporary"
    assert options == {"num_proc": 2, "overwrite": True, "recordings": ["one", "two"]}


def test_bake_help_displays_its_own_execution_flags(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "bake", "--help"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--num-proc" in help_text and "--overwrite" in help_text and "--include-unknown" not in help_text
    assert "--refresh-annotations" in help_text


def test_bake_refresh_dispatches_with_data_overrides(monkeypatch):
    seen = []
    monkeypatch.setattr(bake, "bake_dataset", lambda *args, **kwargs: pytest.fail("must not rebuild raw data"))
    monkeypatch.setattr(bake, "refresh_annotations", seen.append)
    monkeypatch.setattr(sys, "argv", ["main.py", "bake", "--refresh-annotations",
                                     "paths.baked_dir=temporary", "paths.annotations_dir=new_labels"])
    cli.main()
    assert seen[0].paths.baked_dir == "temporary"
    assert seen[0].paths.annotations_dir == "new_labels"


@pytest.mark.parametrize("flags", [["--overwrite"], ["--recording", "one"]])
def test_bake_refresh_rejects_rebuild_flags_before_doing_work(flags, monkeypatch, capsys):
    monkeypatch.setattr(bake, "load_data_config", lambda *args: pytest.fail("must reject conflicting flags first"))
    monkeypatch.setattr(sys, "argv", ["main.py", "bake", "--refresh-annotations", *flags])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err
