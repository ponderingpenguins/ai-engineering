"""Compare reviewed lying annotations with the original label_pose track.

    python main.py audit [--csv PATH] [--out PATH] [--port 8765] [key=value ...]
Write interval, frame and window agreement, with links into the annotation tool.
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from omegaconf.errors import OmegaConfBaseException

from data.annotations import LYING_INTERVALS, REPO, Annotations, Interval, capture_positions, read_annotations
from data.preprocess import select_recordings
from config.loader import load_data_config

logger = logging.getLogger(__name__)


def runs(flags: np.ndarray) -> list[Interval]:
    """Return inclusive (first, last) indices for each consecutive run of True values."""
    edges = np.diff(np.r_[0, flags.astype(int), 0])
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def overlaps(a: Interval, b: Interval) -> bool:
    """Return whether two inclusive intervals share at least one frame."""
    return a[0] <= b[1] and b[0] <= a[1]


def window_flags(flags: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    """Return whether each complete window contains any lying frame."""
    starts = range(0, len(flags) - window_len + 1, stride)
    return np.array([flags[s : s + window_len].any() for s in starts], dtype=bool)


@dataclass
class Agreement:
    """Interval, frame and window agreement between two tracks for one recording."""

    recording: str
    frames: int
    pose_runs: list[Interval]
    annotation_runs: list[Interval]
    frame_counts: dict[str, int] = field(default_factory=dict)  # both, only_pose, only_annotation, neither.
    window_counts: dict[str, int] = field(default_factory=dict)  # Same categories, counted over windows.
    pose_runs_found: list[bool] = field(default_factory=list)  # Whether each pose run overlaps an annotation.
    annotation_runs_found: list[bool] = field(default_factory=list)  # Whether each annotation overlaps a pose run.


def counts(pose: np.ndarray, annotation: np.ndarray) -> dict[str, int]:
    """Count positions marked lying by both tracks, either track alone, or neither."""
    return {"both": int((pose & annotation).sum()), "only_pose": int((pose & ~annotation).sum()),
            "only_annotation": int((~pose & annotation).sum()), "neither": int((~pose & ~annotation).sum())}


def compare(recording: str, pose: np.ndarray, annotation: np.ndarray, window_len: int, stride: int) -> Agreement:
    """Compare two aligned boolean lying tracks at interval, frame and window levels.

    Both arrays must have the same length and follow the recording's capture order.
    Intervals match when they overlap by at least one frame.
    """
    pose_runs, annotation_runs = runs(pose), runs(annotation)
    return Agreement(
        recording=recording, frames=len(pose), pose_runs=pose_runs, annotation_runs=annotation_runs,
        frame_counts=counts(pose, annotation),
        window_counts=counts(window_flags(pose, window_len, stride), window_flags(annotation, window_len, stride)),
        pose_runs_found=[any(overlaps(p, a) for a in annotation_runs) for p in pose_runs],
        annotation_runs_found=[any(overlaps(a, p) for p in pose_runs) for a in annotation_runs],
    )


def tracks(meta: pd.DataFrame, annotations: Annotations) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build aligned pose and manual lying flags for reviewed recordings.

    Args:
        meta: Frame table with recording, timestamp, frame_idx and label_pose
            columns and a unique index.
        annotations: Recording names mapped to intervals and review status.

    Returns:
        Pose and manual boolean arrays in capture order, keyed by recording.
        Skip recordings without review or any pose labels. Missing pose values
        within a compared recording count as not lying.
    """
    meta = meta.assign(position=capture_positions(meta))
    out = {}
    for recording, frames in meta.groupby("recording"):
        marks = annotations.get(str(recording))
        if marks is None or not marks.reviewed or frames.label_pose.isna().all():
            continue
        ordered = frames.sort_values("position")
        pose = ordered.label_pose.fillna("").str.startswith("lying").to_numpy()
        annotation = np.zeros(len(ordered), dtype=bool)
        for first, last in marks.intervals:
            annotation[first : last + 1] = True
        out[str(recording)] = (pose, annotation)
    return out


def short(recording: str) -> str:
    """Return the timestamp suffix of a recording name for display."""
    return recording.split("__")[-1]


def report(meta: pd.DataFrame, annotations: Annotations, window_len: int, stride: int, tool: str) -> str:
    """Build a Markdown agreement report with links to unmatched lying intervals.

    Args:
        meta: Frame metadata required by tracks.
        annotations: Recording names mapped to intervals and review status.
        window_len: Frames per comparison window.
        stride: Frames between window starts.
        tool: Annotation tool base URL used for frame links.

    Returns:
        Markdown text with interval, frame and window counts and review coverage.
    """
    results = [compare(name, pose, annotation, window_len, stride)
               for name, (pose, annotation) in tracks(meta, annotations).items()]
    reviewed = {name for name, marks in annotations.items() if marks.reviewed}
    skipped = sorted((reviewed & set(meta.recording)) - {r.recording for r in results})
    waiting = sorted(set(meta.recording) - reviewed)
    lines = ["# Agreement of the lying annotation with `label_pose`", "",
             f"Compared: {len(results)} reviewed recordings with pose labels. "
             f"Windows are {window_len} frames with stride {stride}, labelled `any_lying`.", ""]
    if skipped:
        lines += [f"Reviewed but without pose labels, so not compared: {', '.join(map(short, skipped))}.", ""]
    if waiting:
        lines += [f"Not reviewed yet, so not compared: {', '.join(map(short, waiting))}.", ""]
    if not results:
        return "\n".join(lines + ["Nothing to compare yet."]) + "\n"

    def total(values: list[int]) -> int:
        """Sum counts across recordings and return a plain Python integer."""
        return int(sum(values))

    lines += ["## Intervals (headline)", "",
              "A lying run is *found* if the other track has a lying run that overlaps it by at least one frame.", "",
              "| Recording | Pose-label runs | found by the annotator | Annotation runs | found in the pose label |",
              "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {short(r.recording)} | {len(r.pose_runs)} | {sum(r.pose_runs_found)} | "
                     f"{len(r.annotation_runs)} | {sum(r.annotation_runs_found)} |")
    lines.append(f"| **all** | **{total([len(r.pose_runs) for r in results])}** | "
                 f"**{total([sum(r.pose_runs_found) for r in results])}** | "
                 f"**{total([len(r.annotation_runs) for r in results])}** | "
                 f"**{total([sum(r.annotation_runs_found) for r in results])}** |")
    for title, key, unit in (("Frames", "frame_counts", "frames"), ("Windows", "window_counts", "windows")):
        lines += ["", f"## {title}", "", f"Number of {unit} that are lying in both tracks, only in one, or in neither.", "",
                  "| Recording | both | only pose label | only annotation | neither |", "|---|---|---|---|---|"]
        for r in results:
            c = getattr(r, key)
            lines.append(f"| {short(r.recording)} | {c['both']} | {c['only_pose']} | {c['only_annotation']} | {c['neither']} |")
        c = {k: total([getattr(r, key)[k] for r in results]) for k in ("both", "only_pose", "only_annotation", "neither")}
        lines.append(f"| **all** | **{c['both']}** | **{c['only_pose']}** | **{c['only_annotation']}** | **{c['neither']}** |")

    def link(recording: str, position: int) -> str:
        """Build an annotation-tool link to the requested capture position."""
        return f"[{short(recording)} @ {position}]({tool}#rec={recording}&f={position})"

    lines += ["", "## Disagreements", "", "Each link opens the annotation tool at the first frame of the run.", ""]
    missed = [(r.recording, run) for r in results for run, found in zip(r.pose_runs, r.pose_runs_found) if not found]
    extra = [(r.recording, run) for r in results for run, found in zip(r.annotation_runs, r.annotation_runs_found)
             if not found]
    lines += [f"**The pose label has lying, the annotator marked none ({len(missed)}):**", ""]
    lines += [f"- {link(rec, first)}, {last - first + 1} frames" for rec, (first, last) in missed] or ["- none"]
    lines += ["", f"**The annotator marked lying, the pose label has none ({len(extra)}):**", ""]
    lines += [f"- {link(rec, first)}, {last - first + 1} frames" for rec, (first, last) in extra] or ["- none"]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Load selected recordings and annotations, then write the agreement report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default="results/annotation_agreement.md")
    parser.add_argument("--port", type=int, default=8765, help="port of the annotation tool, for the links")
    args, overrides = parser.parse_known_args()
    try:
        cfg = load_data_config(REPO, overrides)
    except (ValueError, TypeError, OmegaConfBaseException) as error:
        sys.exit(f"Error: {error}")
    ds = select_recordings(cast(Dataset, load_from_disk(str(REPO / cfg.paths.baked_dir))), cfg.selection)
    csv_path = REPO / args.csv if args.csv is not None else REPO / cfg.paths.annotations_dir / LYING_INTERVALS.name
    meta = cast(pd.DataFrame, ds.select_columns(["recording", "timestamp", "frame_idx", "label_pose"]).to_pandas())
    text = report(meta, read_annotations(csv_path), cfg.windowing.window_len, cfg.windowing.stride,
                  f"http://127.0.0.1:{args.port}/")
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
