import json

import pytest

from models.model import History
from utils.results import BestSetting, Candidate, FoldResult, InnerFold, Pooled, RunResults, Tuning
from utils.run_report import write_report


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


def test_old_files_without_newer_keys_still_load():
    old = {"config": None, "folds": [{"fold": 1, "test_groups": ["Ian"], "n_test": 3, "metrics": {}, "old_key": 1}],
           "pooled": {"n": 3, "metrics": {}, "per_class": {}}}
    loaded = RunResults.from_dict(old)
    assert loaded.folds[0].history is None and loaded.folds[0].tuning is None and loaded.tuned is None
    assert loaded.per_person == {} and loaded.folds[0].val_groups == []


def test_a_file_without_folds_or_pooled_is_refused():
    with pytest.raises(KeyError):
        RunResults.from_dict({"config": None})


def test_the_report_can_be_rebuilt_from_a_stored_run(tmp_path):
    stored = json.loads(json.dumps(full_results().to_dict()))
    report = write_report(RunResults.from_dict(stored), tmp_path).read_text()
    assert "## Tuning" in report and "**Chosen for the final model:**" in report and "Fold 1" in report
