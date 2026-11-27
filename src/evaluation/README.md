# Evaluation and Reporting (`src/evaluation/`)

Computes classification metrics against fixed class sets, defines the typed schema for `results.json` and `final.json`, generates loss-curve figures and Markdown run reports, and audits manual lying annotations against the raw `label_pose` track.



## Modules

| Module                  | Responsibility                                                                                                                                                                               |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `metrics.py`          | Classification metrics (`accuracy`, `per_class`, `macro_f1`, `scores`) and number formatting (`fmt`).                                                                              |
| `records.py`          | Typed dataclasses (`RunResults`, `FoldResult`, `Pooled`, `Tuning`, `Candidate`, `InnerFold`, `BestSetting`, `FinalResult`) and JSON conversion (`to_dict`, `from_dict`). |
| `plots.py`            | Headless Matplotlib (`Agg`) plotting for `loss_per_fold.png` and `loss_mean.png`.                                                                                                      |
| `report.py`           | Markdown report generator (`write_report`) and CLI entry point `python main.py report <run_dir>`.                                                                                        |
| `annotation_audit.py` | Agreement comparison between `lying_intervals.csv` and `label_pose`; CLI entry point `python main.py audit`.                                                                           |



## Walkthrough

### 1. Metrics (`metrics.py`)

Every metric accepts an explicit, dataset-wide `classes` sequence so folds that lack a rare class are scored on the same class set as every other fold. For class $c$ with true positives $\text{TP}_c$, true support $n_c = \sum \mathbb{1}[y = c]$, and predicted count $\hat{n}_c = \sum \mathbb{1}[\hat{y} = c]$:

$$
\text{Precision}_c = \frac{\text{TP}_c}{\hat{n}_c}\;(\hat{n}_c > 0), \qquad \text{Recall}_c = \frac{\text{TP}_c}{n_c}\;(n_c > 0), \qquad F_{1,c} = \frac{2\,\text{TP}_c}{n_c + \hat{n}_c}\;(n_c + \hat{n}_c > 0)
$$

When a denominator is $0$, the function returns `None` (`fmt()` formats `None` as `"N/A"`). If a class has $n_c = 0$ and $\hat{n}_c > 0$, then $\text{TP}_c = 0$ and $F_{1,c} = 0$.

- **`macro_f1`:** Mean of $F_{1,c}$ over all classes where $F_{1,c}$ is not `None`.
- **`accuracy`:** Fraction of exact label matches.
- **Aggregation levels:** `CrossValidator` scores each fold individually (`FoldResult`), concatenates all held-out predictions across folds for the headline score (`Pooled`), and breaks down per-participant accuracy, macro-$F_1$, precision, recall, and support (`per_person`).



### 2. Result Schema (`records.py`)

`records.py` is the single definition of the JSON structure written by `CrossValidator` and `run_train_final`, and read back by `report.py` and `plots.py`:

- `RunResults` (`results.json`): `config`, `folds: list[FoldResult]` (each holding optional `History` and `Tuning -> list[Candidate -> list[InnerFold]]`), `pooled: Pooled`, `per_person`, and optional `tuned: BestSetting`.
- `FinalResult` (`final.json`): `config`, `n_windows`, `n_train_after_resampling`, `train_classes`, `train_seconds`, `history`, and `checkpoint`.



### 3. Loss Curves and Markdown Reports (`plots.py`, `report.py`)

`write_report(results, out_dir)` runs automatically at the end of every `cv` or `tune` run and produces:

- `loss_per_fold.png`: one subplot per fold showing per-step train loss, validation loss points, a dashed vertical line at `best_step`, and a dotted red line where early stopping fired.
- `loss_mean.png`: epoch-aligned mean and $\pm 1\,\text{SD}$ band across folds (1-epoch rolling train mean interpolated onto 10 points per epoch; validation loss at exact evaluation epochs). Points are plotted only where at least two folds remain active.
- `report.md`: run settings, window coverage funnel, pooled headline and per-class tables, nested tuning tables (when tuned), per-fold table, per-person table, embedded curves, and checkpoint paths.

Non-iterative models (such as `baseline`) have `history=None`, so no curve PNGs are written. `report.py` also supports legacy `eval`/`train` config keys so archived `results.json` runs can still be rebuilt.



### 4. Annotation Audit (`annotation_audit.py`)

Compares reviewed recordings in `data/annotations/lying_intervals.csv` against raw `label_pose` (ignoring unreviewed recordings and recordings without pose labels) in capture order at three levels:

1. **Intervals:** Consecutive runs of lying frames; a run counts as matched if the other track overlaps it by at least one frame.
2. **Frames:** Confusion counts (`both`, `only_pose`, `only_annotation`, `neither`).
3. **Windows:** Confusion counts after applying the `any_lying` window rule with `windowing.window_len` and `windowing.stride`.

Every unmatched run is listed with a clickable deep link (`http://127.0.0.1:<port>/#rec=<recording>&f=<position>`) that opens directly at the start of the disagreement in [src/annotation_tool/README.md](../annotation_tool/README.md).



## Usage

```bash
# Rebuild report.md and loss curves from a saved run
python main.py report results/<timestamp>_<model>

# Generate an agreement report between lying_intervals.csv and label_pose
python main.py audit --out results/annotation_agreement.md --port 8765
```



## Extending

### Adding a Classification Metric

1. Implement a function with signature `(y_true, y_pred, classes) -> float | None` in `src/evaluation/metrics.py`.
2. Pass it in `CrossValidator.metrics` (`src/training/cross_validation.py`).
3. Add its column to the pooled and per-fold tables in `src/evaluation/report.py` and add unit tests in `tests/evaluation/test_metrics.py`.

### Extending the `results.json` Schema

1. Add the field with a default value to the relevant dataclass in `src/evaluation/records.py`.
2. Update `from_dict()` if the field contains nested dataclasses, preserving backward compatibility with saved runs that omit the new key.
3. Render the field in `src/evaluation/report.py` and update `tests/evaluation/test_records.py`.
