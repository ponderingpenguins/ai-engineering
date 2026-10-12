"""Compare registered models and write a Markdown report with PNG figures.

    python compare.py models=baseline,rnn model.rnn.epochs=20
"""

import json
import logging
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from data.extract import NO_PERSON
from data.windows import HEATMAP_SHAPE, collapse_lying, make_windows
from models.factory import MODELS, build_model
from utils.config import Config, ModelConfig
from utils.trainer import Trainer, group_folds

REPO = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


@dataclass
class CompareConfig(Config):
    models: str = "baseline,rnn"
    failure_timelines: int = 3


def scores(true, pred, classes):
    """Use a fixed class set; missing denominators are reported as undefined."""
    actual_labels = np.asarray(true)
    predicted_labels = np.asarray(pred)
    result = {"accuracy": float(np.mean(actual_labels == predicted_labels))}
    class_f1_scores = []
    for label in classes:
        true_positives = int(np.sum((actual_labels == label) & (predicted_labels == label)))
        actual_count = int(np.sum(actual_labels == label))
        predicted_count = int(np.sum(predicted_labels == label))
        precision = true_positives / predicted_count if predicted_count else None
        recall = true_positives / actual_count if actual_count else None
        total_count = actual_count + predicted_count
        f1 = 2 * true_positives / total_count if total_count else None
        result[f"{label}_precision"] = precision
        result[f"{label}_recall"] = recall
        result[f"{label}_support"] = actual_count
        if f1 is not None:
            class_f1_scores.append(f1)
    result["macro_f1"] = float(np.mean(class_f1_scores)) if class_f1_scores else None
    return result


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
    unknown = meta.loc[eligible & (meta.person == NO_PERSON), "recording"].nunique()
    if unknown:
        notes.append(f"Unknown participants: {unknown} recordings grouped by recording; participant independence is unverified.")
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


def report(out, results, predictions, classes, checks, cfg, provenance):
    """Generate fixed Markdown tables and plots"""
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    metrics = ["macro_f1", "accuracy"]
    if "lying" in classes:
        metrics += ["lying_precision", "lying_recall"]
    lines = ["# Model comparison", "", "| Model | " + " | ".join(metrics) + " | Train s | Predict s |",
             "|---|" + "---|" * (len(metrics) + 2)]
    for name, entry in results.items():
        lines.append(f"| {name} | " + " | ".join(display(entry["scores"][m]) for m in metrics)
                     + f" | {entry['train_seconds']:.2f} | {entry['predict_seconds']:.2f} |")
    lines += ["", "Scores apply to windows; N/A means undefined. Times total all folds.", ""]
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
              f"- Windows: {cfg.eval.window_len} frames; stride: {cfg.eval.stride}; folds: {cfg.eval.folds}.",
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


def evaluate_model(name, cfg, x, y, groups, metadata, classes):
    """Evaluate one model and attach recording references to its predictions."""
    model_cfg = ModelConfig(name=name, rnn=cfg.model.rnn)

    def make_model():
        # Each fold must start with a fresh model.
        return build_model(model_cfg)

    trainer = Trainer(make_model)
    result = trainer.cross_validate(
        x, y, groups, cfg.eval.folds, collect_predictions=True,
    )

    rows = []
    for prediction in result.pop("predictions"):
        sample_metadata = metadata[prediction["sample_id"]]
        row = {"model": name, **prediction, **sample_metadata}
        row["is_error"] = row["true_label"] != row["predicted_label"]
        row["error_type"] = error_type(row["true_label"], row["predicted_label"])
        row["frame_ids"] = json.dumps(row["frame_ids"])
        rows.append(row)

    predictions = pd.DataFrame(rows)
    result["scores"] = scores(predictions.true_label, predictions.predicted_label, classes)
    for fold in result["folds"]:
        fold_predictions = predictions[predictions.fold == fold["fold"]]
        fold["metrics"] = scores(
            fold_predictions.true_label, fold_predictions.predicted_label, classes,
        )
    result["pooled"]["metrics"] = result["scores"]
    result["train_seconds"] = sum(fold["train_seconds"] for fold in result["folds"])
    result["predict_seconds"] = sum(fold["predict_seconds"] for fold in result["folds"])

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
    """Check the data, evaluate selected models, then save the comparison."""
    names = [n.strip() for n in cfg.models.split(",") if n.strip()]
    if not names or len(set(names)) != len(names) or set(names) - set(MODELS):
        raise ValueError(f"Choose unique models from {MODELS}, separated by commas.")
    if cfg.failure_timelines < 0:
        raise ValueError("failure_timelines cannot be negative.")
    path = REPO / cfg.data.baked
    ds = load_from_disk(str(path))
    checks = preflight(ds, cfg, path)
    for note in checks:
        logger.info(note)
    # Load radar arrays only if at least one selected model uses them.
    heatmaps = any(
        build_model(ModelConfig(name=name, rnn=cfg.model.rnn)).needs_heatmaps
        for name in names
    )
    x, y, groups, metadata = make_windows(
        ds,
        cfg.eval.target,
        window_len=cfg.eval.window_len,
        stride=cfg.eval.stride,
        heatmaps=heatmaps,
        relabel=collapse_lying if cfg.eval.collapse_lying else None,
        return_metadata=True,
    )
    classes = sorted(set(y))
    if len(classes) < 2:
        raise ValueError("Evaluation requires at least two classes after windowing.")
    for i, held_out in enumerate(group_folds(groups, cfg.eval.folds), 1):
        test = np.isin(groups, held_out)
        for label, selected in [("training", ~test), ("test", test)]:
            absent = set(classes) - set(y[selected])
            if absent:
                checks.append(f"Fold {i} {label} missing classes: {', '.join(sorted(absent))}.")
    out = REPO / cfg.eval.out_dir / "comparisons" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out.mkdir(parents=True)
    provenance = {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(run_git("status", "--porcelain")),
        "dataset": str(path.resolve()),
        "dataset_fingerprint": ds._fingerprint,
        "dataset_state": json.loads((path / "state.json").read_text()),
    }
    (out / "config.json").write_text(json.dumps({"settings": asdict(cfg), **provenance}, indent=2))
    results = {}
    all_predictions = []
    participant_rows = []
    for name in names:
        logger.info("Evaluating %s", name)
        result, rows, model_participants = evaluate_model(
            name, cfg, x, y, groups, metadata, classes,
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
    report(out, results, predictions, classes, checks, cfg, provenance)
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
