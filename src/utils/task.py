"""The evaluation task: what a window is and how it is labelled.

A model is only valid for the task it was trained on. The task is stored in every checkpoint and in
`tuned.yaml`, and it is compared when a checkpoint or a tuned setting is reused.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class Task:
    target: str
    collapse_lying: bool
    window_len: int
    label_rule: str
    max_points: int  # point-cloud windows keep at most this many points

    @classmethod
    def from_eval(cls, evaluation: Any) -> "Task":
        """The task of a run, from its `EvalConfig`."""
        return cls(**{f.name: getattr(evaluation, f.name) for f in fields(cls)})

    @classmethod
    def from_dict(cls, values: Any) -> "Task":
        """A task from stored values.

        Raises:
            ValueError: If `values` is not a mapping with exactly the task keys.
        """
        names = [f.name for f in fields(cls)]
        if not isinstance(values, dict) or set(values) != set(names):
            raise ValueError(f"the task settings must be a mapping with the keys {names}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Plain values, for checkpoints and YAML."""
        return asdict(self)
