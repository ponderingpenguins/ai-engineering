"""Structured results for cross-validation, tuning and final training.

Reports and plots read these records; to_dict and from_dict handle JSON conversion.
Cross-validation uses results.json; final training uses final.json.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from training.fitting import History

Setting = dict[str, Any]  # One candidate's dotted configuration overrides.


@dataclass
class InnerFold:
    """Validation loss, epoch and macro F1 for one candidate on one held-out person."""

    val_group: str
    best_epoch: int | None
    val_loss: float
    macro_f1: float | None


@dataclass
class Candidate:
    """One grid setting and its scores across inner validation folds."""

    setting: Setting
    score: float  # Mean weighted inner validation loss; lower is better.
    epochs: int  # Median best inner epoch, rounded up.
    folds: list[InnerFold] = field(default_factory=list)
    macro_f1: float | None = None
    predicted_classes: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Candidate":
        """Restore a candidate and its inner-fold records from plain values."""
        return cls(**{**values, "folds": [InnerFold(**f) for f in values["folds"]]})


@dataclass
class Tuning:
    """Winning setting, epoch count and candidate results for one outer fold."""

    setting: Setting  # Winning overrides for this outer fold.
    epochs: int  # Winner's median best inner epoch, rounded up.
    score: float  # Winner's mean weighted inner validation loss.
    candidates: list[Candidate] = field(default_factory=list)  # Every candidate and its inner results.

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Tuning":
        """Restore the winning setting and all candidate records from plain values."""
        return cls(**{**values, "candidates": [Candidate.from_dict(c) for c in values["candidates"]]})


@dataclass
class BestSetting:
    """Final training choice, selected by wins across the outer folds."""

    setting: Setting
    epochs: int
    wins: int  # Number of outer folds won by this setting.
    of_folds: int  # Total number of outer folds.
    mean_val_loss: float  # Mean inner validation loss for the folds it won.


@dataclass
class FoldResult:
    """People, sample counts, metrics, history and checkpoint for one outer fold."""

    fold: int
    test_groups: list[str]
    val_groups: list[str] = field(default_factory=list)
    n_train: int = 0
    n_train_after_resampling: int = 0
    train_classes: dict[str, int] = field(default_factory=dict)
    n_val: int = 0
    n_test: int = 0
    test_classes: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float | None] = field(default_factory=dict)
    per_class: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: History | None = None  # None for models that do not train iteratively
    tuning: Tuning | None = None  # None when tuning was not requested.
    checkpoint: str | None = None  # Path relative to the run directory.
    train_seconds: float = 0.0
    predict_seconds: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FoldResult":
        """Restore a fold result, including its optional training and tuning histories."""
        history, tuning = values["history"], values["tuning"]
        return cls(**{**values,
                      "history": History(**history) if history else None,
                      "tuning": Tuning.from_dict(tuning) if tuning else None})


@dataclass
class Pooled:
    """Metrics computed once over all outer folds' held-out predictions."""

    n: int
    metrics: dict[str, float | None]
    per_class: dict[str, dict[str, Any]]
    predicted_classes: dict[str, int] = field(default_factory=dict)


@dataclass
class RunResults:
    """A complete cross-validation run, including optional nested tuning results."""

    config: dict[str, Any] | None  # Run settings and data metadata as plain values.
    folds: list[FoldResult]
    pooled: Pooled
    per_person: dict[str, dict[str, Any]] = field(default_factory=dict)  # Counts and flattened scores by person.
    tuned: BestSetting | None = None  # Overall winner when nested tuning was used.

    def to_dict(self) -> dict[str, Any]:
        """Return nested plain values ready for JSON serialisation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RunResults":
        """Restore nested run records from the results.json schema."""
        tuned = values["tuned"]
        return cls(config=values["config"],
                   folds=[FoldResult.from_dict(f) for f in values["folds"]],
                   pooled=Pooled(**values["pooled"]),
                   per_person=values["per_person"],
                   tuned=BestSetting(**tuned) if tuned else None)


@dataclass
class FinalResult:
    """Training metadata for a model fitted on all selected people, without a test score."""

    config: dict[str, Any] | None
    n_windows: int
    n_train_after_resampling: int
    train_classes: dict[str, int]
    train_seconds: float
    history: History | None
    checkpoint: str | None  # Filename inside the run directory.

    def to_dict(self) -> dict[str, Any]:
        """Return nested plain values ready for JSON serialisation."""
        return asdict(self)
