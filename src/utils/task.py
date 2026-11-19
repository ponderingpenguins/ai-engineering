"""The evaluation task: what a window is and how it is labelled.

A model is only valid for the task it was trained on. The task is therefore stored in every checkpoint and in
`tuned.yaml`, and it is compared when a tuned setting or a checkpoint is reused. This module is the one place
that knows the keys, the defaults and the rules for old files.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any

DEFAULT_MAX_POINTS = 128  # point-cloud windows keep at most this many points
_REQUIRED = {"target", "collapse_lying", "window_len"}  # settings that every stored task has


@dataclass(frozen=True)
class Task:
    """The settings that decide what the inputs and labels of a model are.

    The defaults of `label_rule` and `max_points` are the behaviour from before those settings existed, so
    that old checkpoints and old `tuned.yaml` files still load: `label_rule` was always "last" and the
    point-cloud windows did not exist.
    """

    target: str
    collapse_lying: bool
    window_len: int
    label_rule: str = "last"
    max_points: int = DEFAULT_MAX_POINTS

    @classmethod
    def from_eval(cls, evaluation: Any) -> "Task":
        """The task of a run, from its `EvalConfig`."""
        return cls(**{f.name: getattr(evaluation, f.name) for f in fields(cls)})

    @classmethod
    def from_dict(cls, values: Any) -> "Task":
        """A task from stored values. Missing optional keys get their defaults, extra keys are ignored.

        Raises:
            ValueError: If `values` is not a mapping or lacks `target`, `collapse_lying` or `window_len`.
        """
        if not isinstance(values, dict) or not _REQUIRED <= set(values):
            raise ValueError(f"the task settings must be a mapping with the keys {sorted(_REQUIRED)}")
        return cls(**{f.name: values[f.name] for f in fields(cls) if f.name in values})

    def to_dict(self) -> dict[str, Any]:
        """Plain values, for checkpoints and YAML."""
        return asdict(self)

