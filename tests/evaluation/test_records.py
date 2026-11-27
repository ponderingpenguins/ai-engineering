import json

import pytest

from training.fitting import History
from evaluation.records import BestSetting, Candidate, FoldResult, InnerFold, Pooled, RunResults, Tuning
from evaluation.report import write_report


def full_results() -> RunResults:
    history = History(steps_per_epoch=2, train_steps=[1, 2, 3, 4], train_loss=[1.0, 0.9, 0.8, 0.7],
                      val_steps=[2, 4], val_loss=[0.9, 0.8], best_step=4)
    candidate = Candidate({"hidden": 8}, 0.5, 2, [InnerFold("Bob", 2, 0.5, 0.4)], 0.4, {"lying": 1})
    tuning = Tuning({"hidden": 8}, 2, 0.5, [candidate])
    fold = FoldResult(1, ["Ian"], ["Bob"], n_test=5, metrics={"accuracy": 0.8}, history=history, tuning=tuning,
                      per_class={"lying": {"precision": None, "recall": 0.5, "f1": None, "support": 2}})
    pooled = Pooled(5, {"accuracy": 0.8}, {"lying": {"precision": None, "recall": 0.5, "f1": None, "support": 2}})
    return RunResults({"model": {"name": "baseline"}}, [fold], pooled, {"Ian": {"n": 5}},
                      BestSetting({"hidden": 8}, 2, 1, 1, 0.5))


def test_results_round_trip_through_plain_values():
    results = full_results()
    stored = json.loads(json.dumps(results.to_dict()))  # what results.json holds
    assert RunResults.from_dict(stored) == results


def test_nested_records_come_back_as_records():
    loaded = RunResults.from_dict(full_results().to_dict())
    fold = loaded.folds[0]
    assert isinstance(fold.history, History) and fold.history.best_step == 4
    assert fold.tuning.candidates[0].folds[0].val_group == "Bob"
    assert loaded.tuned == BestSetting({"hidden": 8}, 2, 1, 1, 0.5)


def test_a_stored_run_without_its_keys_is_refused():
    with pytest.raises(KeyError):
        RunResults.from_dict({"config": None})


def test_the_report_can_be_rebuilt_from_a_stored_run(tmp_path):
    stored = json.loads(json.dumps(full_results().to_dict()))
    report = write_report(RunResults.from_dict(stored), tmp_path).read_text()
    assert "## Tuning" in report and "**Chosen for the final model:**" in report and "Fold 1" in report


def test_report_shows_saved_frame_and_window_coverage(tmp_path):
    results = full_results()
    results.config["window_coverage"] = {
        "frames_total": 100, "frames_unlabelled": 10, "frames_bad_heatmap": 3,
        "frames_wrong_shape": 2, "frames_kept": 85, "windows_gap_skipped": 4, "windows_kept": 5,
    }
    text = write_report(results, tmp_path).read_text(encoding="utf-8")
    assert "## Window coverage" in text and "counted once, in order" in text
    for label, count in [("Selected frames", 100), ("Dropped: missing target label", 10),
                         ("Dropped: bad heatmap", 3), ("Dropped: wrong sensor shape", 2),
                         ("Frames kept", 85), ("Windows skipped across frame gaps", 4),
                         ("Windows kept after any per-person cap", 5)]:
        assert f"| {label} | {count} |" in text


def test_report_omits_coverage_when_an_older_run_did_not_record_it(tmp_path):
    text = write_report(full_results(), tmp_path).read_text(encoding="utf-8")
    assert "## Window coverage" not in text


@pytest.mark.parametrize("legacy", [False, True])
def test_report_retains_current_and_legacy_config_layouts(tmp_path, legacy):
    results = full_results()
    results.tuned = None
    training = {"epochs": 3, "batch_size": 8, "lr": 0.001, "validation": False, "seed": 5}
    windows = {"window_len": 16, "stride": 8, "label_rule": "any_lying", "target": "label_pose", "folds": 2}
    results.config = (
        {"model": {"name": "rnn", "rnn": training}, "train": {"validation": False}, "eval": windows}
        if legacy else
        {"model": {"name": "rnn", "arch": {"hidden": 8}}, "training": training,
         "windowing": windows, "cv": {"folds": 2}, "seed": 5}
    )
    text = write_report(results, tmp_path).read_text(encoding="utf-8")
    assert "exactly 3 epochs, batch 8, lr 0.001, seed 5" in text
    assert "Windows: 16 frames" in text and "Every fold trains on all its training people" in text


def test_report_module_entry_point_still_rebuilds_saved_results(tmp_path, monkeypatch, capsys):
    import sys
    from evaluation.report import main

    (tmp_path / "results.json").write_text(json.dumps(full_results().to_dict()), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["python -m evaluation.report", str(tmp_path)])
    main()
    assert (tmp_path / "report.md").is_file()
    assert str(tmp_path / "report.md") in capsys.readouterr().out
