"""The results of a run, as records.

`Trainer` builds them, the report and the loss curves read them, and `results.json` stores them. Only the
conversion to and from plain values (`to_dict`, `from_dict`) knows about the stored file, so a key is spelled
in one place.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from models.model import History

Setting = dict[str, Any]  # one hyperparameter setting of the tuning grid


@dataclass
class InnerFold:
    """One inner fold of tuning: how a setting did on one validation person."""

    val_group: str
    best_epoch: int | None
    val_loss: float
    macro_f1: float | None


@dataclass
class Candidate:
    """One setting of the grid with its inner results."""

    setting: Setting
    score: float  # mean class-balanced validation loss over the inner folds (lower is better)
    epochs: int  # median best epoch of the inner folds
    folds: list[InnerFold] = field(default_factory=list)
    macro_f1: float | None = None
    predicted_classes: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Candidate":
        return cls(**{**values, "folds": [InnerFold(**f) for f in values["folds"]]})


@dataclass
class Tuning:
    """The outcome of tuning one outer fold."""

    setting: Setting  # the winning setting
    epochs: int  # median best epoch of the winner's inner folds
    score: float  # the winner's mean class-balanced validation loss over its inner folds (lower is better)
    candidates: list[Candidate] = field(default_factory=list)  # every setting with its inner results

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Tuning":
        return cls(**{**values, "candidates": [Candidate.from_dict(c) for c in values["candidates"]]})


@dataclass
class BestSetting:
    """The setting to use for the final model, chosen over all outer folds."""

    setting: Setting
    epochs: int
    wins: int  # outer folds that this setting won
    of_folds: int  # outer folds in all
    mean_val_loss: float  # mean validation loss of its wins


@dataclass
class FoldResult:
    """One outer fold: who was held out, what was trained on, how it scored."""

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
    tuning: Tuning | None = None  # None without tuning
    checkpoint: str | None = None  # path relative to the run folder
    train_seconds: float = 0.0
    predict_seconds: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FoldResult":
        history, tuning = values["history"], values["tuning"]
        return cls(**{**values,
                      "history": History(**history) if history else None,
                      "tuning": Tuning.from_dict(tuning) if tuning else None})


@dataclass
class Pooled:
    """The score over the held-out predictions of all folds together: the headline number."""

    n: int
    metrics: dict[str, float | None]
    per_class: dict[str, dict[str, Any]]
    predicted_classes: dict[str, int] = field(default_factory=dict)


@dataclass
class RunResults:
    """A cross-validation run."""

    config: dict[str, Any] | None  # the run configuration as plain values, saved with the results
    folds: list[FoldResult]
    pooled: Pooled
    per_person: dict[str, dict[str, Any]] = field(default_factory=dict)  # n, then recall and support per class
    tuned: BestSetting | None = None  # set when the run tuned

    def to_dict(self) -> dict[str, Any]:
        """Plain values, ready for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RunResults":
        tuned = values["tuned"]
        return cls(config=values["config"],
                   folds=[FoldResult.from_dict(f) for f in values["folds"]],
                   pooled=Pooled(**values["pooled"]),
                   per_person=values["per_person"],
                   tuned=BestSetting(**tuned) if tuned else None)


@dataclass
class FinalResult:
    """The final model of a full run: trained on every person, so it has no held-out score."""

    config: dict[str, Any] | None
    n_windows: int
    n_train_after_resampling: int
    train_classes: dict[str, int]
    train_seconds: float
    history: History | None
    checkpoint: str | None  # file name inside the run folder

    def to_dict(self) -> dict[str, Any]:
        """Plain values, ready for JSON."""
        return asdict(self)
