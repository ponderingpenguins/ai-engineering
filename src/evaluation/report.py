"""Rebuild a cross-validation report and loss plots from results.json.

    python main.py report results/<timestamp>_<model>
Cross-validation and tuning also write these outputs automatically.
"""

import argparse
import json
from pathlib import Path

from evaluation.plots import plot_fold_curves, plot_mean_curves
from evaluation.metrics import fmt
from evaluation.records import RunResults


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Build Markdown table lines, followed by a blank line."""
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def _settings(results: RunResults) -> list[str]:
    """Describe saved model, training and window settings, including legacy config keys."""
    config = results.config or {}
    # Accept legacy config keys when rebuilding reports from older result files.
    model, evaluation = config.get("model", {}), config.get("eval", {})
    windowing = config.get("windowing", evaluation)
    evaluation = config.get("cv", evaluation)
    timing = config.get("window_timing")
    validation = (config.get("training", config.get("train")) or {}).get("validation", True)
    tuned = results.tuned is not None
    lines = [f"- Model: `{model.get('name', 'unknown')}`."]
    used = config.get("recordings_used")
    if used is not None:
        short = lambda names: ", ".join(name.split("__")[-1] for name in names) or "none"  # noqa: E731
        lines.append(f"- Target: `{windowing.get('target')}`. {len(used)} recordings used: {short(used)}.")
        if config.get("recordings_unlabelled"):
            lines.append(f"- Not used, because they have no labels for this target: {short(config['recordings_unlabelled'])}.")
    # Read shared training settings for current models, or model-specific legacy settings.
    rnn = config.get("training") if "arch" in model and model.get("name") != "baseline" else model.get(model.get("name"))
    if isinstance(rnn, dict):
        if config.get("input_type") == "pointcloud":
            lines.append(f"- Input: the radar point cloud of each window, up to {windowing.get('max_points')} "
                         "points (x, y, z, doppler, snr, time in window), not the heatmaps.")
        if tuned:
            lines.append(f"- Training: each fold is refit on all its training people for the epoch count chosen by tuning "
                         f"(see Tuning), batch {rnn.get('batch_size')}, seed {config.get('seed', rnn.get('seed'))}; no validation data and no early "
                         "stopping in the refit. The other settings are the tuned setting on top of the defaults below.")
        elif validation:
            lines.append(f"- Training: up to {rnn.get('epochs')} epochs, batch {rnn.get('batch_size')}, lr {rnn.get('lr')}; "
                         f"early stopping on validation loss with patience {rnn.get('patience')} "
                         f"(validation loss evaluated {rnn.get('evals_per_epoch')}x per epoch).")
        else:
            lines.append(f"- Training: exactly {rnn.get('epochs')} epochs, batch {rnn.get('batch_size')}, lr {rnn.get('lr')}, "
                         "seed " + str(config.get('seed', rnn.get('seed'))) + ". No validation data and no early stopping: the epoch count was fixed "
                         "beforehand, and the weights after the last epoch are kept.")
    folds = evaluation.get("folds")
    lines.append(f"- Cross-validation: {len(results.folds)} folds, "
                 f"{'one fold per person' if folds is None else 'stratified, people kept together'}; seed {config.get('seed', evaluation.get('seed'))}. "
                 + ("Each fold tunes on its training people with an inner leave-one-person-out loop, then refits on all of them."
                    if tuned else "Each fold also holds out one training person for validation." if validation
                    else "Every fold trains on all its training people."))
    if windowing:
        window = f"{windowing.get('window_len')} frames"
        if timing:
            window += (f" = {timing['window_s']:.1f} s (stride {windowing.get('stride')} frames = "
                       f"{timing['stride_s']:.1f} s)")
        lines.append(f"- Windows: {window}; label rule `{windowing.get('label_rule')}`. "
                     "With `any_lying` a positive window means lying occurred somewhere within it.")
    return lines


def _coverage(results: RunResults) -> list[str]:
    """Describe saved frame and window counts; return no lines for older runs without them."""
    coverage = (results.config or {}).get("window_coverage")
    if not isinstance(coverage, dict):
        return []
    rows = [
        ("Selected frames", "frames_total"),
        ("Dropped: missing target label", "frames_unlabelled"),
        ("Dropped: bad heatmap", "frames_bad_heatmap"),
        ("Dropped: wrong sensor shape", "frames_wrong_shape"),
        ("Frames kept", "frames_kept"),
        ("Windows skipped across frame gaps", "windows_gap_skipped"),
        ("Windows kept after any per-person cap", "windows_kept"),
    ]
    return ["## Window coverage", "",
            "Counts start after recording selection. Frame drops are counted once, in order: "
            "missing target label, bad heatmap, then wrong sensor shape. "
            "The windows kept include any per-person cap.", "",
            *_table(["Stage", "Count"], [[label, str(coverage.get(key, "N/A"))] for label, key in rows])]


def _tuning(results: RunResults) -> list[str]:
    """Build the tuning section with the overall winner and each outer fold's candidates."""
    best = results.tuned
    assert best is not None
    lines = ["## Tuning", "",
             "Settings are compared by the mean class-balanced validation loss over the inner folds (lower is better), "
             "measured at the weights early stopping kept. Validation macro F1 is shown for information only. "
             "The test person of the outer fold is never used.", "",
             f"**Chosen for the final model:** `{best.setting}` for {best.epochs} epochs. It won {best.wins} of "
             f"{best.of_folds} outer folds with mean validation loss {fmt(best.mean_val_loss)}. "
             "Saved to `tuned.yaml`; a full run (`python main.py train-final`) reads it.", ""]
    for fold in results.folds:
        tuning = fold.tuning
        if not tuning:
            continue
        lines += [f"### Fold {fold.fold}: test {'+'.join(fold.test_groups)}", ""]
        rows = []
        for candidate in tuning.candidates:
            won = candidate.setting == tuning.setting
            inner = ", ".join(f"{f.val_group}: {fmt(f.val_loss)}" for f in candidate.folds)
            epochs = ", ".join(str(f.best_epoch) for f in candidate.folds)
            rows.append([("**" + str(candidate.setting) + "**") if won else str(candidate.setting),
                         fmt(candidate.score), inner, fmt(candidate.macro_f1), epochs, str(candidate.epochs),
                         "yes" if won else ""])
        lines += _table(["Setting", "Val loss", "Loss per validation person", "Val macro F1", "Best epochs", "Epochs used", "Chosen"],
                        rows)
    return lines


def write_report(results: RunResults, out_dir: Path) -> Path:
    """Write report.md and available loss plots into an existing output directory.

    Args:
        results: Completed cross-validation results, including any tuning records.
        out_dir: Directory for the report and figures.

    Returns:
        Path to the written report.
    """
    folds = results.folds
    classes = list(results.pooled.per_class)
    figures = [("Loss per fold", plot_fold_curves(folds, out_dir / "loss_per_fold.png")),
               ("Loss across folds", plot_mean_curves(folds, out_dir / "loss_mean.png"))]

    lines = ["# Training report", "", *_settings(results), "", *_coverage(results), "## Pooled results", "",
             "Scored once over the held-out predictions of all folds together. This is the headline number.", ""]
    pooled = results.pooled
    lines += _table(["Accuracy", "Macro F1", "Windows"],
                    [[fmt(pooled.metrics.get("accuracy")), fmt(pooled.metrics.get("macro_f1")), str(pooled.n)]])
    lines += _table(["Class", "Precision", "Recall", "F1", "Support"],
                    [[c, fmt(r["precision"]), fmt(r["recall"]), fmt(r["f1"]), str(r["support"])]
                     for c, r in pooled.per_class.items()])

    if results.tuned is not None:
        lines += _tuning(results)

    lines += ["## Per fold", "", "N/A means undefined, e.g. the class has no windows in that fold.", ""]
    header = ["Fold", "Test", "Val", "Test windows", "Accuracy", "Macro F1", *[f"Recall {c}" for c in classes],
              "Best step", "Stopped"]
    rows = []
    for fold in folds:
        history, per_class = fold.history, fold.per_class
        stopped = "early" if history and history.stopped_early else ("max epochs" if history else "-")
        rows.append([str(fold.fold), "+".join(fold.test_groups), "+".join(fold.val_groups) or "-",
                     str(fold.n_test), fmt(fold.metrics.get("accuracy")), fmt(fold.metrics.get("macro_f1")),
                     *[fmt(per_class.get(c, {}).get("recall")) + f" (n={per_class.get(c, {}).get('support', 0)})"
                       for c in classes],
                     "-" if history is None or history.best_step is None else str(history.best_step), stopped])
    lines += _table(header, rows)

    lines += ["## Per person", "", "How each held-out person does, with the number of windows behind each recall.", ""]
    lines += _table(["Person", "Windows", *[f"Recall {c} (n)" for c in classes]],
                    [[person, str(row["n"]), *[f"{fmt(row.get(f'{c}_recall'))} ({row.get(f'{c}_support', 0)})" for c in classes]]
                     for person, row in results.per_person.items()])

    if any(path for _, path in figures):
        lines += ["## Loss curves", ""]
        for title, path in figures:
            if path:
                lines += [f"![{title}]({path.name})", ""]
    else:
        lines += ["## Loss curves", "", "None: this model does not train iteratively.", ""]

    checkpoints = [(fold.fold, fold.checkpoint) for fold in folds if fold.checkpoint]
    if checkpoints:
        lines += ["## Checkpoints", "", "Best weights of each fold; load with `BaseModel.load`.", ""]
        lines += [f"- Fold {number}: `{path}`" for number, path in checkpoints] + [""]

    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    """Load a saved run's results.json and rebuild its report and loss plots."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="saved run folder containing results.json")
    args = parser.parse_args()
    try:
        results = RunResults.from_dict(json.loads((args.run_dir / "results.json").read_text(encoding="utf-8")))
        report = write_report(results, args.run_dir)
    except KeyError as error:
        parser.error(f"Unsupported or incomplete results.json: missing field {error}; "
                     "expected the current run-results schema.")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(report)


if __name__ == "__main__":
    main()
