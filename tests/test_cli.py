import json
import subprocess
import sys
from importlib import import_module

import pytest

import main as cli
from data import bake
from evaluation.records import FoldResult, Pooled, RunResults
from training import pipelines


@pytest.mark.parametrize("mode", ["cv", "tune", "train-final"])
def test_explicit_modes_dispatch_independently_of_grid_contents(mode, monkeypatch):
    seen = []
    function = {"cv": "run_cv", "tune": "run_tune", "train-final": "run_train_final"}[mode]
    monkeypatch.setattr(pipelines, function, lambda cfg, root: seen.append((cfg, mode)))
    monkeypatch.setattr(sys, "argv", ["main.py", mode, "model=rnn",
                                     'tune.grid=[{"model.arch.hidden":16}]'])
    cli.main()
    cfg, command = seen[0]
    assert command == mode and cfg.tune.grid == [{"model.arch.hidden": 16}]
    assert not hasattr(cfg.training, "full_run")


def test_tune_without_candidates_fails_before_running(monkeypatch):
    monkeypatch.setattr(pipelines, "run_tune", lambda *args: pytest.fail("must not start an empty tuning run"))
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


@pytest.mark.parametrize("command,arguments", [
    ("bake", ["--num-proc", "2", "paths.baked_dir=temporary"]),
    ("report", ["results/a run"]),
    ("annotate", ["--port", "9000", "--csv", "marks.csv", "--lan", "--no-browser", "paths.baked_dir=temporary"]),
    ("audit", ["--csv", "marks.csv", "--out", "audit.md", "--port", "9000", "windowing.window_len=8"]),
    ("record", ["--cfg-file", "config/radar/Config_4.cfg", "--camera-port", "0", "--out-dir", "tmp_rec"]),
])
def test_tool_commands_forward_arguments_without_loading_training_config(command, arguments, monkeypatch):
    module = import_module(cli.TOOL_COMMANDS[command])
    seen = []
    monkeypatch.setattr(module, "main", lambda: seen.append(sys.argv.copy()))
    monkeypatch.setattr(cli, "load_run_config", lambda *a: pytest.fail("tools must use their own config loader"))
    original = ["main.py", command, *arguments]
    monkeypatch.setattr(sys, "argv", original)
    cli.main()
    assert seen == [[f"main.py {command}", *arguments]]
    assert sys.argv is original


@pytest.mark.parametrize("command,expected", [
    ("bake", "--refresh-annotations"),
    ("report", "run_dir"),
    ("annotate", "--no-browser"),
    ("audit", "--out"),
    ("record", "--out-dir"),
])
def test_each_tool_has_its_own_help(command, expected, monkeypatch, capsys):
    original = ["main.py", command, "--help"]
    monkeypatch.setattr(sys, "argv", original)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert f"main.py {command}" in help_text and expected in help_text
    assert sys.argv is original


def test_top_level_help_lists_every_command(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "--help"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert all(command in help_text for command in ("cv", "tune", "train-final", *cli.TOOL_COMMANDS))


@pytest.mark.parametrize("arguments", [["--help"], ["report", "--help"]])
def test_help_does_not_import_the_training_pipeline(arguments):
    script = """
import runpy
import sys
sys.argv = ["main.py", *sys.argv[1:]]
try:
    runpy.run_path("main.py", run_name="__main__")
except SystemExit as error:
    assert error.code == 0, error.code
assert "training.pipelines" not in sys.modules
if sys.argv == ["main.py", "--help"]:
    assert "torch" not in sys.modules and "datasets" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script, *arguments], cwd=cli.REPO,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("command,arguments", [
    ("report", []),
    ("report", ["one", "two"]),
    ("annotate", ["--port", "not-an-integer"]),
    ("audit", ["--port", "not-an-integer"]),
    ("record", []),
])
def test_tool_argument_errors_keep_the_child_exit_code(command, arguments, monkeypatch, capsys):
    original = ["main.py", command, *arguments]
    monkeypatch.setattr(sys, "argv", original)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert f"main.py {command}: error:" in capsys.readouterr().err
    assert sys.argv is original


@pytest.mark.parametrize("command", ["annotate", "audit"])
def test_annotation_commands_reject_training_overrides_before_loading_data(command, monkeypatch):
    module = import_module(cli.TOOL_COMMANDS[command])
    monkeypatch.setattr(module, "load_from_disk", lambda *a: pytest.fail("invalid data config must fail first"))
    monkeypatch.setattr(sys, "argv", ["main.py", command, "training.epochs=1"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert str(error.value.code).startswith("Error:")


def test_report_command_rebuilds_a_saved_run(tmp_path, monkeypatch, capsys):
    results = RunResults(None, [FoldResult(1, ["Bob"])], Pooled(3, {"accuracy": 0.5}, {}))
    (tmp_path / "results.json").write_text(json.dumps(results.to_dict()), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["main.py", "report", str(tmp_path)])
    cli.main()
    assert (tmp_path / "report.md").is_file()
    assert "## Pooled results" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert str(tmp_path / "report.md") in capsys.readouterr().out


@pytest.mark.parametrize("contents", [None, "not json", "{}"])
def test_report_command_reports_missing_or_invalid_results(tmp_path, contents, monkeypatch, capsys):
    if contents is not None:
        (tmp_path / "results.json").write_text(contents, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["main.py", "report", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "main.py report: error:" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()


def test_report_explains_an_unsupported_legacy_result_schema(tmp_path, monkeypatch, capsys):
    saved = {"status": "complete", "config": {}, "folds": [], "pooled": {}, "predictions": []}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    original = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["main.py", "report", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "expected the current run-results schema" in capsys.readouterr().err
    assert path.read_bytes() == original
