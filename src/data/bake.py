"""Bake the raw floor-posture recordings into one Hugging Face dataset.

Besides the raw labels, hand annotations are joined in: `person`, `person_present`
and `lying_annotation`, from the CSVs in data/annotations/.

Run it:

    python main.py bake [paths.raw_root=...] [--num-proc 4] [--overwrite] [--recording a --recording b]
    python main.py bake --refresh-annotations [paths.annotations_dir=...]

The module command `python -m data.bake` also remains available.

Usage:
    ds = load_from_disk("data/baked/floor_posture")
    ds = ds.select_columns(["recording", "heatmap", "label_pose"])
"""

import argparse
import gc
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Image, Sequence, Value, load_from_disk
from omegaconf.errors import OmegaConfBaseException

from data.annotations import attach_annotations
from config.data import DataConfig
from config.loader import load_data_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
ALIGN_TOLERANCE = pd.Timedelta("2us")  # CSV rounding


# --- labels ---------------------------------------------------------------
# Label CSVs are messy: each recording ships a different set of tracks, the
# manifest.json only lists some of them, and timestamps come in three flavours
# (ISO strings, epoch-ns of local wall-clock, epoch-seconds shifted by the
# timezone). Align by timestamp because frame and label row orders can differ.

def _label_files(rec_dir: Path) -> list[Path]:
    """Lists the label CSV files of a recording.

    Args:
        rec_dir: Directory of one recording.

    Returns:
        The paths of `labels/*.csv`, sorted by name.
    """
    return sorted((rec_dir / "labels").glob("*.csv"))


def _kind(labels: pd.Series) -> str | None:
    """Classifies a label track as boolean, numeric or string.

    Args:
        labels: The values of one label track, possibly containing missing values.

    Returns:
        "bool", "float" or "str", or None if every value is missing.
    """
    values = labels.dropna()
    if values.empty:
        return None
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return "bool"
    if pd.api.types.is_numeric_dtype(values):
        return "float"
    return "str"


def _scan_label_kinds(recs: list[dict[str, Any]]) -> dict[str, str]:
    """Track name -> 'bool' | 'float' | 'str', resolved across all recordings."""
    seen: dict[str, set[str]] = {}
    for rec in recs:
        for path in _label_files(Path(rec["path"])):
            kinds = seen.setdefault(path.stem, set())
            kind = _kind(pd.read_csv(path)["label"])
            if kind:
                kinds.add(kind)
    return {
        track: kinds.pop() if len(kinds) == 1 and next(iter(kinds)) in ("bool", "float") else "str"
        for track, kinds in seen.items()
    }


def _local_time(timestamps: pd.Series) -> pd.Series | None:
    """Timestamps on the frames' wall-clock, or None if they can't be trusted."""
    if pd.api.types.is_numeric_dtype(timestamps):
        if timestamps.iloc[0] > 1e17:  # epoch-ns of local wall-clock
            return pd.to_datetime(timestamps, unit="ns").astype("datetime64[ns]")
        return None  # epoch-seconds, shifted by the (DST-dependent) UTC offset
    return pd.to_datetime(timestamps).astype("datetime64[ns]")


def _align_labels(rec_dir: Path, frame_ts: pd.Series) -> dict[str, pd.DataFrame]:
    """Each track as a DataFrame(label, confidence) with one row per frame."""
    aligned: dict[str, pd.DataFrame] = {}
    frame_ts = pd.to_datetime(frame_ts).astype("datetime64[ns]")
    left = pd.DataFrame({"t": frame_ts.values, "i": np.arange(len(frame_ts))})
    for path in _label_files(rec_dir):
        df = pd.read_csv(path)
        label_ts = _local_time(df["timestamp"]) if "timestamp" in df else None
        if label_ts is None:
            logger.warning("%s/%s: no usable timestamps, skipped", rec_dir.name, path.stem)
            continue
        if label_ts.isna().any() or label_ts.duplicated().any():
            raise ValueError(f"{rec_dir.name}/{path.stem}: missing or duplicate label timestamps")
        right = pd.DataFrame({
            "t": label_ts.values,
            "label": df["label"].values,
            "confidence": df["confidence"].values if "confidence" in df else np.nan,
            "matched": True,
        })
        merged = pd.merge_asof(
            left.sort_values("t"), right.sort_values("t"), on="t",
            direction="nearest", tolerance=ALIGN_TOLERANCE,
        ).sort_values("i").reset_index(drop=True)
        unmatched = int(merged["matched"].isna().sum())
        if unmatched:
            logger.warning("%s/%s: %d/%d frames have no matching label timestamp",
                           rec_dir.name, path.stem, unmatched, len(frame_ts))
        aligned[path.stem] = merged[["label", "confidence"]]
    return aligned


def _cast(value: Any, kind: str) -> Any:
    """Converts a label value to the Python type of its track.

    Args:
        value: A single label value, possibly missing.
        kind: One of "bool", "float" or "str", as returned by `_kind`.

    Returns:
        The converted value, or None if `value` is missing.
    """
    if pd.isna(value):
        return None
    return {"bool": bool, "float": float, "str": str}[kind](value)


# --- frames ---------------------------------------------------------------

def _features(kinds: dict[str, str]) -> Features:
    """Builds the Hugging Face schema of the baked dataset.

    Args:
        kinds: Maps each label track name to its kind ("bool", "float" or "str").

    Returns:
        The features, with a label column and a confidence column per track.
    """
    columns: dict[str, Any] = {
        "recording": Value("string"),
        "room": Value("string"),
        "frame_idx": Value("int32"),  # position in the recording; radar_frame_num wraps and repeats
        "timestamp": Value("timestamp[us]"),
        "radar_frame_num": Value("int32"),
        "image": Image(),  # camera JPEG, decoded lazily
        "radar_payload": Value("binary"),  # raw TLV bytes, enough to re-parse anything
        "heatmap": Sequence(Sequence(Value("uint16"))),  # (range_bins, doppler_bins), varies per recording
        "heatmap_ok": Value("bool"),  # False for the few frames whose heatmap came back empty
        "pointcloud": Sequence(Sequence(Value("float32"))),  # (n, 7): x y z doppler snr noise track
        "n_points": Value("int32"),
        "n_tracks": Value("int32"),
        "n_heights": Value("int32"),
        "cfg_file": Value("string"),
        "range_resolution_m": Value("float32"),
        "range_max_m": Value("float32"),
        "doppler_resolution_m_s": Value("float32"),
        "max_velocity_m_s": Value("float32"),
        "num_range_bins": Value("int32"),
        "num_doppler_bins": Value("int32"),
        "num_tx": Value("int32"),
        "num_rx": Value("int32"),
    }
    dtypes = {"bool": "bool", "float": "float32", "str": "string"}
    for track, kind in kinds.items():
        columns[f"label_{track}"] = Value(dtypes[kind])
        columns[f"label_{track}_confidence"] = Value("float32")
    return Features(columns)


def _rows(recordings: list[dict[str, Any]], kinds: dict[str, str]) -> Iterator[dict[str, Any]]:
    """Generator for Dataset.from_generator; `recordings` is sharded across processes."""
    from loguru import logger as loguru_logger
    from vt_radar.config.ti_config import ConfigFile
    from vt_radar.data.dataset import read_directory

    loguru_logger.remove()  # the loader is chatty

    for rec in recordings:
        rec_dir = Path(rec["path"])
        cfg_path = sorted(rec_dir.glob("*.cfg"))[0]
        cfg = ConfigFile()
        cfg.parseCfg(str(cfg_path))
        s = cfg.sensor()

        # read_directory swallows unreadable files, so check we got everything
        frames = list(read_directory(str(rec_dir), config=cfg))
        if len(frames) != rec["frames"]:
            raise RuntimeError(f"{rec['recording']}: read {len(frames)} frames, manifest says {rec['frames']}")

        frame_ts = pd.Series(pd.to_datetime([f.timestamp for f in frames])).astype("datetime64[ns]")
        labels = _align_labels(rec_dir, frame_ts)
        logger.info("%s: %d frames, label tracks %s", rec["recording"], len(frames), sorted(labels))

        for i, f in enumerate(frames):
            out = f.output
            heatmap = np.asarray(out.range_doppler_heatmap)
            row: dict[str, Any] = {
                "recording": rec["recording"],
                "room": rec["room"],
                "frame_idx": i,
                "timestamp": frame_ts.iloc[i].to_pydatetime(),
                "radar_frame_num": out.frame_num,
                "image": {"bytes": f.raw_image, "path": None} if f.raw_image else None,
                "radar_payload": f.frame,
                "heatmap": heatmap,
                "heatmap_ok": heatmap.size > 0,
                "pointcloud": np.asarray(out.pointcloud, dtype=np.float32).reshape(-1, 7),
                "n_points": out.num_detected_points,
                "n_tracks": out.num_detected_tracks,
                "n_heights": out.num_detected_heights,
                "cfg_file": cfg_path.name,
                "range_resolution_m": s.range_resolution,
                "range_max_m": s.range_max,
                "doppler_resolution_m_s": s.doppler_resolution,
                "max_velocity_m_s": s.max_velocity,
                "num_range_bins": s.num_range_bins,
                "num_doppler_bins": s.num_doppler_bins,
                "num_tx": s.num_tx_antennas,
                "num_rx": s.num_rx_antennas,
            }
            for track, kind in kinds.items():
                df = labels.get(track)
                if df is None:  # this recording has no such track
                    row[f"label_{track}"] = row[f"label_{track}_confidence"] = None
                    continue
                row[f"label_{track}"] = _cast(df["label"].iloc[i], kind)
                row[f"label_{track}_confidence"] = _cast(df["confidence"].iloc[i], "float") if "confidence" in df else None
            yield row


def _recordings(root: Path, only: list[str]) -> list[dict[str, Any]]:
    """Lists the recordings that are present on disk.

    Args:
        root: Root of the raw dataset, containing MANIFEST.csv and `data/`.
        only: Recording names to keep. Empty means every recording.

    Returns:
        One dict per recording with its name, room, path and frame count. Recordings
        whose folder is missing (for example in another zip part) are skipped.

    Raises:
        ValueError: If a name in `only` is not in MANIFEST.csv.
    """
    manifest = pd.read_csv(root / "MANIFEST.csv")
    if only:
        unknown = set(only) - set(manifest.dataset)
        if unknown:
            raise ValueError(f"not in MANIFEST.csv: {sorted(unknown)}")
        manifest = manifest[manifest.dataset.isin(only)]
    recs = []
    for r in manifest.itertuples():
        path = root / "data" / r.room / r.dataset
        if not path.is_dir():
            logger.warning("%s is not in this folder (another zip part?), skipped", r.dataset)
            continue
        recs.append({"recording": r.dataset, "room": r.room, "path": str(path), "frames": int(r.frames)})
    return recs


def bake_dataset(cfg: DataConfig, *, num_proc: int = 4, overwrite: bool = False, recordings: list[str] | None = None) -> None:
    """Bakes the raw recordings into a Hugging Face dataset and saves it to disk.

    Args:
        cfg: Dataset paths and annotation configuration.

    Raises:
        SystemExit: If the output exists and `overwrite` is False, or no
            recordings are found.
    """
    if num_proc < 1:
        raise ValueError("--num-proc must be positive")
    root = REPO / cfg.paths.raw_root
    out = REPO / cfg.paths.baked_dir
    if out.exists():
        if not overwrite:
            logger.error("%s already exists. Pass --overwrite to rebuild it.", out)
            sys.exit(1)
        shutil.rmtree(out)

    recs = _recordings(root, recordings or [])
    if not recs:
        logger.error("no recordings found under %s", root)
        sys.exit(1)
    kinds = _scan_label_kinds(recs)
    logger.info("%d recordings, label tracks: %s", len(recs), kinds)

    cache = out.parent / f".{out.name}_cache"  # keep HF's build cache next to the output, not in ~/.cache
    ds = Dataset.from_generator(
        _rows,
        features=_features(kinds),
        gen_kwargs={"recordings": recs, "kinds": kinds},
        num_proc=min(num_proc, len(recs)) if num_proc > 1 else None,
        cache_dir=str(cache),
    )
    ds = attach_annotations(ds, REPO / cfg.paths.annotations_dir)
    ds.save_to_disk(str(out))
    shutil.rmtree(cache, ignore_errors=True)
    logger.info("saved %d rows to %s", len(ds), out)


def refresh_annotations(cfg: DataConfig) -> None:
    """Refresh hand annotations in an existing bake without reading raw recordings.

    Write a sibling copy and release Arrow memory maps before renaming directories
    (required on Windows). Existing recovery folders are never overwritten.
    """
    out = REPO / cfg.paths.baked_dir
    temporary = out.with_name(f".{out.name}.refresh_tmp")
    backup = out.with_name(f".{out.name}.old")
    if not out.is_dir():
        raise FileNotFoundError(f"No baked dataset at {out}; run bake first.")
    for path in (temporary, backup):
        if path.exists():
            raise FileExistsError(f"Refresh recovery folder already exists: {path}; inspect it before retrying.")

    try:
        ds = cast(Dataset, load_from_disk(str(out)))
        try:
            ds = attach_annotations(ds, REPO / cfg.paths.annotations_dir)
            n_rows = len(ds)
            ds.save_to_disk(str(temporary))
        finally:
            del ds
            gc.collect()  # close Arrow memory maps before touching the source folder
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    out.rename(backup)
    try:
        temporary.rename(out)
    except OSError:
        backup.rename(out)
        raise
    shutil.rmtree(backup)
    logger.info("refreshed annotations for %d rows in %s", n_rows, out)


def main() -> None:
    """main function"""
    parser = argparse.ArgumentParser(description="Bake raw recordings and join source annotations.")
    parser.add_argument("--num-proc", type=int, default=4, help="parallel workers (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing baked dataset")
    parser.add_argument("--recording", action="append", default=[], help="recording to bake; repeat to select several")
    parser.add_argument("--refresh-annotations", action="store_true",
                        help="refresh annotations in the existing bake without re-reading raw recordings")
    args, overrides = parser.parse_known_args()
    if args.refresh_annotations and (args.recording or args.overwrite):
        parser.error("--refresh-annotations cannot be combined with --recording or --overwrite")
    try:
        cfg = load_data_config(REPO, overrides)
        if args.refresh_annotations:
            refresh_annotations(cfg)
        else:
            bake_dataset(cfg, num_proc=args.num_proc, overwrite=args.overwrite, recordings=args.recording)
    except (TypeError, ValueError, OSError, OmegaConfBaseException) as e:
        parser.error(str(e))



if __name__ == "__main__":
    main()
