import pytest

from utils.config import EvalConfig
from utils.task import DEFAULT_MAX_POINTS, Task


def test_task_is_built_from_the_eval_config():
    task = Task.from_eval(EvalConfig(window_len=8, max_points=64))
    assert task.window_len == 8 and task.max_points == 64 and task.label_rule == EvalConfig().label_rule


def test_old_stored_tasks_get_the_old_defaults():
    task = Task.from_dict({"target": "label_pose", "collapse_lying": True, "window_len": 16})
    assert task.label_rule == "last" and task.max_points == DEFAULT_MAX_POINTS


def test_extra_keys_are_ignored_and_the_dict_round_trips():
    task = Task("label_pose", True, 16, "any_lying", 12)
    assert Task.from_dict({**task.to_dict(), "unknown": 1}) == task


@pytest.mark.parametrize("values", [None, [], {"target": "label_pose"}, {"collapse_lying": True, "window_len": 16}])
def test_stored_tasks_without_the_required_keys_are_refused(values):
    with pytest.raises(ValueError, match="task settings"):
        Task.from_dict(values)


def test_the_default_number_of_points_is_defined_once():
    assert EvalConfig().max_points == DEFAULT_MAX_POINTS
