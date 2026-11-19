import pytest

from utils.config import EvalConfig
from utils.task import Task


def test_task_is_built_from_the_eval_config():
    task = Task.from_eval(EvalConfig(window_len=8, max_points=64))
    assert task.window_len == 8 and task.max_points == 64 and task.label_rule == EvalConfig().label_rule


def test_task_round_trips_through_plain_values():
    task = Task("label_pose", True, 16, "any_lying", 12)
    assert Task.from_dict(task.to_dict()) == task


@pytest.mark.parametrize("values", [
    None, [], {"target": "label_pose"},
    {"target": "label_pose", "collapse_lying": True, "window_len": 16, "label_rule": "any_lying"},  # a key is missing
    {"target": "label_pose", "collapse_lying": True, "window_len": 16, "label_rule": "any_lying", "max_points": 8,
     "extra": 1},
])
def test_stored_tasks_must_have_exactly_the_task_keys(values):
    with pytest.raises(ValueError, match="task settings"):
        Task.from_dict(values)
