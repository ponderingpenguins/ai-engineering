"""A short Markdown report of a cross-validation run: settings, result tables and loss curves.

Written automatically at the end of a run. To rebuild it from a saved run:

    python -m utils.run_report results/<timestamp>_<model>
"""

import json
import sys
from pathlib import Path
from typing import Any

from utils.curves import plot_fold_curves, plot_mean_curves


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def _settings(results: dict[str, Any]) -> list[str]:
    config = results.get("config") or {}
    model, evaluation = config.get("model", {}), config.get("eval", {})
    timing = config.get("window_timing")
    validation = (config.get("train") or {}).get("validation", True)
    tuned = "tuned" in results
    lines = [f"- Model: `{model.get('name', 'unknown')}`."]
    rnn = model.get(model.get("name"))  # the chosen model's own hyperparameters; the baseline has none
    if isinstance(rnn, dict):
        if config.get("input_type") == "pointcloud":
            lines.append(f"- Input: the radar point cloud of each window, up to {(config.get('eval') or {}).get('max_points')} "
                         "points (x, y, z, doppler, snr, time in window), not the heatmaps.")
        if tuned:
            lines.append(f"- Training: each fold is refit on all its training people for the epoch count chosen by tuning "
                         f"(see Tuning), batch {rnn.get('batch_size')}, seed {rnn.get('seed')}; no validation data and no early "
                         "stopping in the refit. The other settings are the tuned setting on top of the defaults below.")
        elif validation:
            lines.append(f"- Training: up to {rnn.get('epochs')} epochs, batch {rnn.get('batch_size')}, lr {rnn.get('lr')}; "
                         f"early stopping on validation loss with patience {rnn.get('patience')} "
                         f"(validation loss evaluated {rnn.get('evals_per_epoch')}x per epoch).")
        else:
            lines.append(f"- Training: exactly {rnn.get('epochs')} epochs, batch {rnn.get('batch_size')}, lr {rnn.get('lr')}, "
                         "seed " + str(rnn.get('seed')) + ". No validation data and no early stopping: the epoch count was fixed "
                         "beforehand, and the weights after the last epoch are kept.")
    folds = evaluation.get("folds")
    lines.append(f"- Cross-validation: {len(results['folds'])} folds, "
                 f"{'one fold per person' if folds is None else 'stratified, people kept together'}; seed {evaluation.get('seed')}. "
                 + ("Each fold tunes on its training people with an inner leave-one-person-out loop, then refits on all of them."
                    if tuned else "Each fold also holds out one training person for validation." if validation
                    else "Every fold trains on all its training people."))
    if evaluation:
        window = f"{evaluation.get('window_len')} frames"
        if timing:
            window += (f" = {timing['window_s']:.1f} s (stride {evaluation.get('stride')} frames = "
                       f"{timing['stride_s']:.1f} s)")
        lines.append(f"- Windows: {window}; label rule `{evaluation.get('label_rule')}`. "
                     "With `any_lying` a positive window means lying occurred somewhere within it.")
    return lines


def _tuning(results: dict[str, Any]) -> list[str]:
    """The Tuning section: the chosen setting, then every setting's inner scores for each outer fold."""
    best = results["tuned"]
    lines = ["## Tuning", "",
             "Settings are compared by the mean class-balanced validation loss over the inner folds (lower is better), "
             "measured at the weights early stopping kept. Validation macro F1 is shown for information only. "
             "The test person of the outer fold is never used.", "",
             f"**Chosen for the final model:** `{best['setting']}` for {best['epochs']} epochs. It won {best['wins']} of "
             f"{best['of_folds']} outer folds with mean validation loss {_fmt(best['mean_val_loss'])}. "
             "Saved to `tuned.yaml`; a full run (`train.full_run=true`) reads it.", ""]
    for fold in results["folds"]:
        tuning = fold.get("tuning")
        if not tuning:
            continue
        lines += [f"### Fold {fold['fold']}: test {'+'.join(fold['test_groups'])}", ""]
        rows = []
        for candidate in tuning["candidates"]:
            won = candidate["setting"] == tuning["setting"]
            inner = ", ".join(f"{f['val_group']}: {_fmt(f['val_loss'])}" for f in candidate["folds"])
            epochs = ", ".join(str(f["best_epoch"]) for f in candidate["folds"])
            rows.append([("**" + str(candidate["setting"]) + "**") if won else str(candidate["setting"]),
                         _fmt(candidate["score"]), inner, _fmt(candidate["macro_f1"]), epochs, str(candidate["epochs"]),
                         "yes" if won else ""])
        lines += _table(["Setting", "Val loss", "Loss per validation person", "Val macro F1", "Best epochs", "Epochs used", "Chosen"],
                        rows)
    return lines


def write_report(results: dict[str, Any], out_dir: Path) -> Path:
    """Writes `report.md` and the loss-curve figures into `out_dir` and returns the report path."""
    folds = results["folds"]
    classes = list(results["pooled"]["per_class"])
    figures = [("Loss per fold", plot_fold_curves(folds, out_dir / "loss_per_fold.png")),
               ("Loss across folds", plot_mean_curves(folds, out_dir / "loss_mean.png"))]

    lines = ["# Training report", "", *_settings(results), "", "## Pooled results", "",
             "Scored once over the held-out predictions of all folds together. This is the headline number.", ""]
    pooled = results["pooled"]
    lines += _table(["Accuracy", "Macro F1", "Windows"],
                    [[_fmt(pooled["metrics"].get("accuracy")), _fmt(pooled["metrics"].get("macro_f1")), str(pooled["n"])]])
    lines += _table(["Class", "Precision", "Recall", "F1", "Support"],
                    [[c, _fmt(r["precision"]), _fmt(r["recall"]), _fmt(r["f1"]), str(r["support"])]
                     for c, r in pooled["per_class"].items()])

    if "tuned" in results:
        lines += _tuning(results)

    lines += ["## Per fold", "", "N/A means undefined, e.g. the class has no windows in that fold.", ""]
    header = ["Fold", "Test", "Val", "Test windows", "Accuracy", "Macro F1", *[f"Recall {c}" for c in classes],
              "Best step", "Stopped"]
    rows = []
    for fold in folds:
        history = fold.get("history") or {}
        per_class = fold.get("per_class", {})
        stopped = "early" if history.get("stopped_early") else ("max epochs" if history else "-")
        rows.append([str(fold["fold"]), "+".join(fold["test_groups"]), "+".join(fold.get("val_groups", [])) or "-",
                     str(fold["n_test"]), _fmt(fold["metrics"].get("accuracy")), _fmt(fold["metrics"].get("macro_f1")),
                     *[_fmt(per_class.get(c, {}).get("recall")) + f" (n={per_class.get(c, {}).get('support', 0)})"
                       for c in classes],
                     str(history.get("best_step", "-")), stopped])
    lines += _table(header, rows)

    lines += ["## Per person", "", "How each held-out person does, with the number of windows behind each recall.", ""]
    lines += _table(["Person", "Windows", *[f"Recall {c} (n)" for c in classes]],
                    [[person, str(row["n"]), *[f"{_fmt(row.get(f'{c}_recall'))} ({row.get(f'{c}_support', 0)})" for c in classes]]
                     for person, row in results.get("per_person", {}).items()])

    if any(path for _, path in figures):
        lines += ["## Loss curves", ""]
        for title, path in figures:
            if path:
                lines += [f"![{title}]({path.name})", ""]
    else:
        lines += ["## Loss curves", "", "None: this model does not train iteratively.", ""]

    checkpoints = [(fold["fold"], fold["checkpoint"]) for fold in folds if fold.get("checkpoint")]
    if checkpoints:
        lines += ["## Checkpoints", "", "Best weights of each fold; load with `BaseModel.load`.", ""]
        lines += [f"- Fold {number}: `{path}`" for number, path in checkpoints] + [""]

    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    """Rebuilds the report of a saved run from its `results.json`."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m utils.run_report <run folder>")
    run = Path(sys.argv[1])
    print(write_report(json.loads((run / "results.json").read_text()), run))


if __name__ == "__main__":
    main()
