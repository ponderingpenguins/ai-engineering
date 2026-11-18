"""Compare registered models and write a Markdown report with PNG figures.

    python -m utils.compare 'checkpoints=["results/training/<timestamp>/rnn.pt"]'
"""

import json
import logging
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from data.preprocess import build_pipeline
from data.windows import HEATMAP_SHAPE, collapse_lying, make_windows, window_timing
from models.baseline import MajorityClass
from models.model import BaseModel
from utils.config import Config
from utils.metrics import scores

REPO = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


@dataclass
class CompareConfig(Config):
    checkpoints: list[str] = field(default_factory=list)
    failure_timelines: int = 3
    partitions: str = "data/processed/floor_posture_person_split"


def preflight(ds, cfg, path):
    """Validate metadata and describe exclusions before loading radar arrays."""
    required = {
        "recording", "person", "frame_idx", "timestamp", "heatmap_ok",
        "num_range_bins", "num_doppler_bins", cfg.eval.target,
    }
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}. Rebuild it with python -m data.extract.")
    meta = ds.select_columns(sorted(required)).to_pandas()
    if meta[["recording", "person", "frame_idx", "timestamp"]].isna().any().any():
        raise ValueError("Missing recording, participant, frame ID or timestamp.")
    if meta.duplicated(["recording", "frame_idx"]).any():
        raise ValueError("Duplicate frame IDs within a recording.")
    if cfg.eval.window_len < 1 or cfg.eval.stride < 1:
        raise ValueError("Window length and stride must be positive.")
    correct_shape = (
        (meta.num_range_bins == HEATMAP_SHAPE[0])
        & (meta.num_doppler_bins == HEATMAP_SHAPE[1])
    )
    valid_heatmap = meta.heatmap_ok.fillna(False)
    has_label = meta[cfg.eval.target].notna()
    eligible = valid_heatmap & has_label & correct_shape
    notes = [
        f"Frames: {len(meta)}; eligible: {int(eligible.sum())}; excluded: {int((~eligible).sum())}.",
        f"Missing labels: {int((~has_label).sum())}; "
        f"invalid heatmaps: {int((~valid_heatmap).sum())}; "
        f"other geometry: {int((~correct_shape).sum())}. Counts overlap.",
    ]
    if not eligible.any():
        raise ValueError("No usable frames for the selected target and radar geometry.")
    sources = [
        REPO / "src/data/extract.py",
        REPO / "data/processed/person_map.csv",
        REPO / "data/processed/presence_events.csv",
    ]
    state = path / "state.json"
    if state.exists() and any(
        source.exists() and source.stat().st_mtime > state.stat().st_mtime
        for source in sources
    ):
        notes.append("Prepared dataset predates extraction code or annotations; rebuild before interpreting scores.")
    if "person_present" not in ds.column_names:
        notes.append("Prepared dataset has no manual person_present column.")
    raw = REPO / cfg.data.root / "data"
    if raw.exists():
        present = {p.name for p in raw.glob("*/*") if p.is_dir()}
        stale = set(meta.recording) - present
        if stale:
            notes.append(f"Prepared dataset contains {len(stale)} recordings absent from raw data; rebuild it.")
    return notes


def save_figure(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"![{name.removesuffix('.png').replace('_', ' ')}](figures/{name})"


def display(value):
    return "N/A" if value is None else f"{value:.3f}"


def plot_scores(results, metrics, figures):
    """Compare overall scores side by side for every model."""
    fig, ax = plt.subplots(figsize=(9, 4))
    locations = np.arange(len(metrics))
    width = 0.8 / len(results)
    for i, (name, entry) in enumerate(results.items()):
        values = [entry["scores"][m] for m in metrics]
        positions = locations - 0.4 + width / 2 + i * width
        ax.bar(positions,
               [np.nan if v is None else v for v in values], width, label=name)
        for position, value in zip(positions, values):
            ax.text(position, 0.02 if value is None else value + 0.015,
                    display(value), ha="center", va="bottom", fontsize=8)
    ax.set(xticks=locations, xticklabels=metrics, ylim=(0, 1.05), ylabel="Score")
    ax.legend()
    return save_figure(fig, figures, "scores.png")


def plot_confusion_matrices(results, predictions, classes, figures):
    """Show the counts of actual versus predicted labels for each model."""
    lines = []
    for name in results:
        rows = predictions[predictions.model == name]
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for actual_index, actual_label in enumerate(classes):
            for predicted_index, predicted_label in enumerate(classes):
                matching = (
                    (rows.true_label == actual_label)
                    & (rows.predicted_label == predicted_label)
                )
                matrix[actual_index, predicted_index] = int(matching.sum())
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black")
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
               xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="Actual", title=name)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        lines += [save_figure(fig, figures, f"confusion_{name}.png"), ""]
    return lines


def plot_participant_scores(results, predictions, classes, figures):
    """Show whether a model performs differently across held-out participants."""
    fig, ax = plt.subplots(figsize=(10, 4))
    groups = sorted(predictions.group.unique())
    for name in results:
        values = []
        for group in groups:
            rows = predictions[(predictions.model == name) & (predictions.group == group)]
            values.append(scores(rows.true_label, rows.predicted_label, classes)["macro_f1"])
        ax.plot(range(len(groups)), values, "o-", label=name)
    ax.set(xticks=range(len(groups)), xticklabels=groups, ylabel="Macro F1", ylim=(0, 1.05))
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    ax.legend()
    return save_figure(fig, figures, "participants.png")


def plot_failure_timelines(results, predictions, classes, figures, limit):
    """Plot actual and predicted labels for the recordings with most mistakes."""
    lines = []
    error_counts = predictions.groupby("recording").is_error.sum().sort_values(ascending=False)
    worst_recordings = error_counts[error_counts > 0].head(limit).index
    class_positions = {label: index for index, label in enumerate(classes)}
    for number, recording in enumerate(worst_recordings, 1):
        rows = predictions[predictions.recording == recording].sort_values("end_timestamp")
        fig, axes = plt.subplots(
            len(results), 1, figsize=(10, 2.5 * len(results)), squeeze=False,
        )
        for ax, name in zip(axes[:, 0], results):
            model_predictions = rows[rows.model == name]
            timestamps = pd.to_datetime(model_predictions.end_timestamp)
            seconds = (timestamps - timestamps.min()).dt.total_seconds()
            actual = model_predictions.true_label.map(class_positions)
            predicted = model_predictions.predicted_label.map(class_positions)
            errors = model_predictions.is_error.to_numpy()

            ax.step(seconds, actual, where="post", label="Actual")
            ax.step(seconds, predicted, where="post", label="Predicted", linestyle="--")
            ax.scatter(seconds[errors], predicted[errors], color="red", s=18, label="Error")
            ax.set(
                yticks=range(len(classes)), yticklabels=classes,
                title=f"{name}: {recording}", xlabel="Seconds from first evaluated window",
            )
            ax.legend(loc="upper right")
        lines += ["", save_figure(fig, figures, f"failure_{number}.png")]
    return lines


def report(out, results, predictions, classes, checks, cfg, provenance, timing):
    """Generate fixed Markdown tables and plots"""
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    metrics = ["macro_f1", "accuracy"]
    if "lying" in classes:
        metrics += ["lying_precision", "lying_recall"]
    lines = ["# Model comparison", "", "| Model | " + " | ".join(metrics) + " | Predict s |",
             "|---|" + "---|" * (len(metrics) + 1)]
    for name, entry in results.items():
        lines.append(f"| {name} | " + " | ".join(display(entry["scores"][m]) for m in metrics)
                     + f" | {entry['predict_seconds']:.2f} |")
    lines += ["", "Scores apply to test windows; N/A means undefined.", ""]
    score_graph = plot_scores(results, metrics, figures)
    lines += [score_graph, "", "## Confusion matrices", ""]
    lines += plot_confusion_matrices(results, predictions, classes, figures)
    lines += ["## Participant scores", ""]
    participant_graph = plot_participant_scores(results, predictions, classes, figures)
    lines += [participant_graph, "", "## Failures", "",
              "| Model | Recording | Errors | Windows | Error rate |", "|---|---|---|---|---|"]
    ranked = predictions.groupby(["model", "recording"]).agg(errors=("is_error", "sum"), windows=("is_error", "size"))
    ranked["rate"] = ranked.errors / ranked.windows
    ranked = ranked.sort_values(["errors", "rate"], ascending=False)
    for (name, rec), row in ranked.head(10).iterrows():
        lines.append(f"| {name} | {rec} | {int(row.errors)} | {int(row.windows)} | {row.rate:.1%} |")
    lines += ["", "| Model | Recording | End timestamp | Actual | Predicted | Sample ID |",
              "|---|---|---|---|---|---|"]
    for row in predictions[predictions.is_error].head(10).itertuples():
        lines.append(f"| {row.model} | {row.recording} | {row.end_timestamp} | {row.true_label} | {row.predicted_label} | {row.sample_id} |")
    lines += plot_failure_timelines(results, predictions, classes, figures, cfg.failure_timelines)
    lines += ["", "[All predictions](predictions.csv) · [Failures](failures.csv) · [Participant metrics](participants.csv)",
              "", "## Data checks", ""] + [f"- {note}" for note in checks]
    lines += ["", "## Run", "", f"- Task: `{cfg.eval.target}`; collapse lying: `{cfg.eval.collapse_lying}`.",
              f"- Windows: {cfg.eval.window_len} frames = {timing['window_s']:.1f} s; "
              f"stride: {cfg.eval.stride} frames = {timing['stride_s']:.1f} s "
              f"({timing['frame_period_s']:.2f} s per frame).",
              f"- Window label rule: `{cfg.eval.label_rule}`"
              + (" (the classifier predicts whether lying occurred within the window; a positive window need not be lying throughout)." if cfg.eval.label_rule == "any_lying"
                 else " (the label of its last frame)."),
              f"- Prepared partitions: `{cfg.partitions}`. Train: baseline label; test: scores. No model training.",
              f"- Classes: `{dict(Counter(predictions[predictions.model == next(iter(results))].true_label))}`.",
              f"- Commit: `{provenance['commit']}`; uncommitted changes: `{provenance['dirty']}`.",
              "- [Settings and dataset identity](config.json) · [Metrics](metrics.json)", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def error_type(actual_label, predicted_label):
    """Name the error so failures can be filtered in the CSV."""
    if actual_label == predicted_label:
        return "correct"
    if actual_label == "lying":
        return "missed_lying"
    if predicted_label == "lying":
        return "false_lying"
    return "misclassified"


def evaluate_model(name, model, windows, classes):
    """Predict test windows with an already prepared model; never call fit."""
    x_test, y_test, _, metadata = windows["test"]
    started = perf_counter()
    predicted_labels = model.predict(x_test)
    predict_seconds = perf_counter() - started
    if len(predicted_labels) != len(y_test):
        raise ValueError(f"{name}: expected one prediction per test window.")

    rows = []
    for sample_id, (actual, predicted) in enumerate(zip(y_test, predicted_labels)):
        row = {
            "model": name, "sample_id": sample_id, "split": "test",
            "true_label": str(actual), "predicted_label": str(predicted),
            **metadata[sample_id],
        }
        row["is_error"] = row["true_label"] != row["predicted_label"]
        row["error_type"] = error_type(row["true_label"], row["predicted_label"])
        row["frame_ids"] = json.dumps(row["frame_ids"])
        rows.append(row)

    predictions = pd.DataFrame(rows)
    result = {
        "scores": scores(y_test, predicted_labels, classes),
        "n_test": len(y_test),
        "predict_seconds": predict_seconds,
    }

    participant_rows = []
    for group, group_predictions in predictions.groupby("group"):
        participant_rows.append({
            "model": name,
            "group": group,
            "windows": len(group_predictions),
            **scores(group_predictions.true_label, group_predictions.predicted_label, classes),
        })
    return result, rows, participant_rows


def run_git(*arguments):
    """Read repository information without modifying it."""
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


def run(cfg):
    """Load saved models and compare their predictions on the supplied test set."""
    if cfg.failure_timelines < 0:
        raise ValueError("failure_timelines cannot be negative.")
    models = {"baseline": MajorityClass()}
    checkpoint_info = {}
    saved_task = None
    for filename in cfg.checkpoints:
        checkpoint_path = REPO / filename
        name = checkpoint_path.stem
        if name in models:
            raise ValueError(f"Duplicate model name {name!r}; checkpoint filenames must be unique and not baseline.")
        model, task = BaseModel.load(checkpoint_path)
        if saved_task is not None and task != saved_task:
            raise ValueError("Compared checkpoints must use the same target, label conversion and window length.")
        saved_task = task
        models[name] = model
        checkpoint_info[name] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "evaluation": task,
        }
    # Task settings come from training, not from guesses at comparison time.
    if saved_task is not None:
        for key, value in saved_task.items():
            setattr(cfg.eval, key, value)

    path = REPO / cfg.partitions
    partitions = load_from_disk(str(path))
    if not isinstance(partitions, DatasetDict) or not {"train", "test"} <= set(partitions):
        raise ValueError("partitions must point to a prepared DatasetDict with train and test splits.")

    # The supplied partitions define membership; never create folds or move rows.
    checks = []
    recordings_seen = set()
    participants_seen = set()
    heatmaps = any(model.needs_heatmaps for model in models.values())
    windows = {}
    pipeline = build_pipeline(cfg.data.preprocess)
    for split in ["train", "test"]:
        if split not in partitions:
            continue
        raw_size = len(partitions[split])
        dataset = pipeline(partitions[split])  # same filters as training: Config 4, known participants
        checks.append(f"{split}: preprocessing kept {len(dataset)} of {raw_size} frames.")
        if len(dataset) == 0:
            raise ValueError(f"{split}: the supplied partition is empty after preprocessing.")
        checks.extend(f"{split}: {note}" for note in preflight(dataset, cfg, path / split))
        recordings = set(dataset["recording"])
        participants = set(dataset["person"])
        if recordings & recordings_seen or participants & participants_seen:
            raise ValueError(f"{split}: recording or participant overlaps another partition.")
        recordings_seen.update(recordings)
        participants_seen.update(participants)

        windows[split] = make_windows(
            dataset,
            cfg.eval.target,
            window_len=cfg.eval.window_len,
            stride=cfg.eval.stride,
            heatmaps=heatmaps and split == "test",
            relabel=collapse_lying if cfg.eval.collapse_lying else None,
            return_metadata=True,
            label_rule=cfg.eval.label_rule,
        )
        _, labels, _, _ = windows[split]
        if split == "test":
            timing = window_timing(dataset, cfg.eval.window_len, cfg.eval.stride)
        if not len(labels):
            raise ValueError(f"{split}: no usable windows for the selected task.")
        checks.append(f"{split}: {len(labels)} windows; classes: {dict(Counter(labels.tolist()))}.")

    # The baseline needs only the majority training label; no radar is fitted.
    models["baseline"].label_ = Counter(windows["train"][1]).most_common(1)[0][0]
    # Include saved model classes even when the supplied test set lacks one.
    classes = set(windows["train"][1]) | set(windows["test"][1])
    for name, model in models.items():
        if name != "baseline":
            classes.update(getattr(model, "classes_", []))
    classes = sorted(classes)
    for split, (_, labels, _, _) in windows.items():
        missing = set(classes) - set(labels)
        if missing:
            checks.append(f"{split}: missing classes: {', '.join(sorted(missing))}.")
    for note in checks:
        logger.info(note)

    out = REPO / cfg.eval.out_dir / "comparisons" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out.mkdir(parents=True)
    provenance = {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(run_git("status", "--porcelain")),
        "dataset": str(path.resolve()),
        "checkpoints": checkpoint_info,
        "partitions": {
            split: {
                "fingerprint": dataset._fingerprint,
                "state": json.loads((path / split / "state.json").read_text()),
            }
            for split, dataset in partitions.items()
        },
    }
    (out / "config.json").write_text(json.dumps({"settings": asdict(cfg), **provenance}, indent=2))
    results = {}
    all_predictions = []
    participant_rows = []
    for name, model in models.items():
        logger.info("Evaluating %s", name)
        result, rows, model_participants = evaluate_model(
            name, model, windows, classes,
        )
        all_predictions.extend(rows)
        results[name] = result
        participant_rows.extend(model_participants)
        # Preserve completed models even if a subsequent model fails.
        pd.DataFrame(all_predictions).to_csv(out / "predictions.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(results, indent=2))
    predictions = pd.DataFrame(all_predictions)
    predictions[predictions.is_error].to_csv(out / "failures.csv", index=False)
    pd.DataFrame(participant_rows).to_csv(out / "participants.csv", index=False)
    report(out, results, predictions, classes, checks, cfg, provenance, timing)
    logger.info("Report: %s", out / "comparison.md")
    return out


def main():
    """Combine defaults, YAML settings and command-line overrides."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        merged_config = OmegaConf.merge(
            OmegaConf.structured(CompareConfig),
            OmegaConf.load(REPO / "config/config.yaml"),
            OmegaConf.from_cli(),
        )
        cfg = OmegaConf.to_object(merged_config)
        run(cfg)
    except (ValueError, OSError, KeyError, OmegaConfBaseException) as exc:
        raise SystemExit(f"Comparison failed: {exc}") from exc


if __name__ == "__main__":
    main()
