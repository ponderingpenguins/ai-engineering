"""Propose quality thresholds for config/quality.yaml from annotated Config_4 recordings.

    uv run python scripts/analysis/tune_quality_policy.py
    uv run python scripts/analysis/tune_quality_policy.py --recordings 2311__20231124_152926 2311__20231124_153817

Recordings in selection.exclude_recordings (config/data.yaml) are skipped unless --include-excluded.
Per recording, the empty-room reference is fitted on the first reference_frames empty frames (capture
order, from presence_events.csv), as the live viewer does. The remaining empty frames are held out.
Windows are window_len consecutive frames (stride 8) with one occupancy state and no frame gap.

Proposals:
    visibility_min, gradient_min   95th percentile over held-out empty windows
    z_change                       99th percentile of |Z| over held-out empty ROI cells
    roi_points_min, snr_min_db     10th percentile over occupied windows
    empty_frame_fraction_max       90th percentile over occupied windows (at least 1/window_len)

The script prints a report and a YAML block. It does not write config/quality.yaml.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from data.annotations import attach_presence, capture_positions, lying_labels, read_annotations
from datasets import Dataset
from vt_radar.config.ti_config import ConfigFile
from vt_radar.data.dataset import read_directory
from vt_radar.parsing_utils.parseFrame import parseStandardFrame
from vt_radar.quality import (
    context_from_config,
    fit_empty_reference,
    heatmap_problem,
    load_policy,
    measure_frame,
    points_from_frame,
    window_measurements,
)

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/floor_posture_dataset/data"
STRIDE = 8


def find_recordings(names, include_excluded):
    dirs = sorted(RAW.glob("*/*/Config_4.cfg"))
    found = {p.parent.name: p.parent for p in dirs}
    if names:
        missing = sorted(set(names) - set(found))
        if missing:
            raise SystemExit(f"no Config_4 recording named {missing}")
        return [found[n] for n in names]
    events = pd.read_csv(REPO / "data/annotations/presence_events.csv")
    selected = set(events.recording) & set(found)
    if not include_excluded:
        selected -= set(OmegaConf.load(REPO / "config/data.yaml").selection.exclude_recordings)
    return [found[n] for n in sorted(selected)]


def load_recording(rec_dir, cfg):
    """Parsed frames in capture order, with presence, lying label and a segment id that changes at gaps."""
    frames = list(read_directory(str(rec_dir), config=cfg))
    meta = pd.DataFrame({"recording": rec_dir.name, "frame_idx": np.arange(len(frames)),
                         "timestamp": pd.to_datetime([f.timestamp for f in frames])})
    meta["position"] = capture_positions(meta)
    present = attach_presence(Dataset.from_dict({"recording": meta.recording.tolist(),
                                                 "frame_idx": meta.frame_idx.tolist()}))["person_present"]
    meta["present"] = pd.Series(present, dtype=object)  # True, False or None
    meta["lying"] = lying_labels(meta, read_annotations())
    rows = []
    for i in meta.sort_values("position").index:
        try:
            data = parseStandardFrame(frames[i].frame, cfg)
        except Exception:
            data = None
        rows.append((i, data))
    return meta, rows


def recording_windows(rec_dir, policy):
    cfg = ConfigFile()
    cfg.parseCfg(str(rec_dir / "Config_4.cfg"))
    context = context_from_config(cfg, policy)
    meta, rows = load_recording(rec_dir, cfg)

    frames, segment, last_num = [], 0, None
    for i, data in rows:
        if data is None or heatmap_problem(data.range_doppler_heatmap, context):
            segment, last_num = segment + 1, None
            continue
        if last_num is not None and data.frame_num != last_num + 1:
            segment += 1
        last_num = data.frame_num
        frames.append({"i": i, "data": data, "segment": segment,
                       "present": meta.present[i], "lying": meta.lying[i]})

    empty = [f for f in frames if f["present"] is False]
    reference, held_out_z = None, []
    if len(empty) >= policy.reference_frames + policy.window_len:
        fit = empty[: policy.reference_frames]
        reference = fit_empty_reference([f["data"].range_doppler_heatmap for f in fit], context, policy)
        fit_ids = {f["i"] for f in fit}
        for f in empty[policy.reference_frames:]:
            held_out_z.append(np.abs(reference.standardise(f["data"].range_doppler_heatmap)[context.range_roi]).ravel())
    else:
        fit_ids = set()

    for f in frames:
        points, declared, snr_ok = points_from_frame(f["data"])
        f["metrics"] = measure_frame(f["data"].range_doppler_heatmap, points, context, reference, policy,
                                     f["data"].frame_num, "present" if f["present"] else "empty", declared, snr_ok)

    out = []
    for start in range(0, len(frames) - policy.window_len + 1, STRIDE):
        win = frames[start : start + policy.window_len]
        if len({f["segment"] for f in win}) > 1 or len({f["present"] for f in win}) > 1:
            continue
        if win[0]["present"] is None or any(f["i"] in fit_ids for f in win):
            continue
        if win[0]["present"]:
            labels = {f["lying"] for f in win}
            kind = "unreviewed" if None in labels else ("lying" if "lying" in labels else "not_lying")
        else:
            kind = "empty"
        m = window_measurements([f["metrics"] for f in win], policy)
        out.append({"recording": rec_dir.name, "config": context.config_hash, "kind": kind, **m})
    return out, held_out_z, len(empty), reference is not None


def pct(series, q):
    s = series.dropna()
    return float(np.percentile(s, q)) if len(s) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", nargs="*", help="recording names (default: all Config_4 with presence events)")
    parser.add_argument("--include-excluded", action="store_true",
                        help="also use selection.exclude_recordings from config/data.yaml")
    args = parser.parse_args()
    logging.getLogger("data.annotations").setLevel(logging.ERROR)  # warns about every other recording
    policy = load_policy()

    rows, z_cells = [], []
    print(f"{'recording':32} {'config':16} {'empty frames':>12} {'reference':>9} {'windows':>7}")
    for rec_dir in find_recordings(args.recordings, args.include_excluded):
        windows, z, n_empty, has_ref = recording_windows(rec_dir, policy)
        rows += windows
        z_cells += z
        config = windows[0]["config"] if windows else "-"
        print(f"{rec_dir.name:32} {config:16} {n_empty:12d} {'yes' if has_ref else 'no':>9} {len(windows):7d}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no windows")
    if df.config.nunique() > 1:
        print(f"\nnote: {df.config.nunique()} different cfg contents are pooled")

    cols = ["visibility_median", "gradient_detail", "changed_fraction_median", "gini_median", "roi_points",
            "empty_roi_frame_fraction", "median_point_snr_db", "point_overflow_fraction"]
    print("\nmedian per window kind (n = windows):")
    summary = df.groupby("kind")[cols].median()
    summary.insert(0, "n", df.groupby("kind").size())
    print(summary.round(3).to_string())
    print("\nmedian visibility per recording with a reference (empty is held out):")
    per_rec = df[df.visibility_median.notna()].pivot_table(index="recording", columns="kind",
                                                           values="visibility_median", aggfunc="median")
    print(per_rec.round(1).to_string())

    empty = df[(df.kind == "empty") & df.visibility_median.notna()]
    occupied = df[df.kind.isin(["lying", "not_lying", "unreviewed"])]
    t = dict(policy.thresholds)
    proposals = {}
    if len(empty):
        proposals["visibility_min"] = pct(empty.visibility_median, 95)
        proposals["gradient_min"] = pct(empty.gradient_detail, 95)
    proposals["roi_points_min"] = pct(occupied.roi_points, 10)
    proposals["empty_frame_fraction_max"] = max(pct(occupied.empty_roi_frame_fraction, 90) or 0, 1 / policy.window_len)
    proposals["snr_min_db"] = pct(occupied.median_point_snr_db, 10)
    z_change = float(np.percentile(np.concatenate(z_cells), 99)) if z_cells else None

    print(f"\nheld-out empty windows: {len(empty)} from {empty.recording.nunique()} recordings")
    if len(empty) < 20:
        print("warning: fewer than 20 held-out empty windows; heatmap proposals are unreliable")
    for name in ("visibility_min", "gradient_min"):
        v = proposals.get(name)
        if v is None:
            continue
        col = "visibility_median" if name == "visibility_min" else "gradient_detail"
        print(f"\n{name}: current {t[name]:g}, proposed {v:.3g}")
        for kind, group in df.groupby("kind"):
            vals = group[col].dropna()
            if len(vals):
                print(f"  {kind:11} at or above: current {np.mean(vals >= t[name]):6.1%}, "
                      f"proposed {np.mean(vals >= v):6.1%}  (n={len(vals)})")
    for name in ("roi_points_min", "empty_frame_fraction_max", "snr_min_db"):
        v = proposals[name]
        print(f"{name}: current {t[name]:g}, proposed {'n/a' if v is None else f'{v:.3g}'}")
    if z_change is not None:
        print(f"z_change: current {policy.z_change:g}, proposed {z_change:.3g}")
    empty_pts = df[df.kind == "empty"].roi_points / policy.window_len
    if len(empty_pts):
        print(f"empty-scene ROI points per frame: median {empty_pts.median():.2f}, max {empty_pts.max():.2f}")

    print("\nproposed values for config/quality.yaml (review before you copy):")
    if z_change is not None:
        print(f"z_change: {z_change:.2f}")
    print("thresholds:")
    for name in t:
        v = proposals.get(name)
        print(f"  {name}: {t[name] if v is None or v <= 0 else round(v, 3)}")


if __name__ == "__main__":
    main()
