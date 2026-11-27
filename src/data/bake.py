"""Build a Hugging Face dataset from raw radar recordings and annotation CSVs.

    python main.py bake [--num-proc 4] [--overwrite] [--recording NAME] ...
    python main.py bake --refresh-annotations [key=value ...]
The refresh command updates manual annotations without parsing raw recordings.
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
ALIGN_TOLERANCE = pd.Timedelta("2us")  # Allow for timestamp rounding in CSV exports.


# Label loading and alignment.
# Discover tracks from CSV files because recording manifests may omit some.
# Match by timestamp because frame and label row orders can differ.

def _label_files(rec_dir: Path) -> list[Path]:
    """Return the recording's labels/*.csv paths, sorted by filename."""
    return sorted((rec_dir / "labels").glob("*.csv"))


def _kind(labels: pd.Series) -> str | None:
    """Return bool, float or str for a label track, or None if all values are missing."""
    values = labels.dropna()
    if values.empty:
        return None
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return "bool"
    if pd.api.types.is_numeric_dtype(values):
        return "float"
    return "str"


def _scan_label_kinds(recs: list[dict[str, Any]]) -> dict[str, str]:
    """Choose a storage type for each label track across all recordings.

    Tracks with mixed types or no non-missing values use string storage.
    """
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
    """Convert reliable timestamps to the frames' local wall-clock time.

    Accept timestamp strings and local epoch nanoseconds. Return None for
    numeric epoch seconds, whose timezone offset cannot be trusted.
    """
    if pd.api.types.is_numeric_dtype(timestamps):
        if timestamps.iloc[0] > 1e17:  # Epoch nanoseconds representing local wall-clock time.
            return pd.to_datetime(timestamps, unit="ns").astype("datetime64[ns]")
        return None  # Numeric seconds have an unreliable timezone offset.
    return pd.to_datetime(timestamps).astype("datetime64[ns]")


def _align_labels(rec_dir: Path, frame_ts: pd.Series) -> dict[str, pd.DataFrame]:
    """Match label timestamps to frames within ALIGN_TOLERANCE.

    Args:
        rec_dir: Recording directory containing label CSVs.
        frame_ts: Frame timestamps in their stored row order.

    Returns:
        A mapping from track names to label/confidence tables in frame row order.
        Tracks without reliable timestamps are skipped; unmatched values are missing.

    Raises:
        ValueError: If a label track has missing or duplicate timestamps.
    """
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
    """Convert one label to its track type, returning None for a missing value."""
    if pd.isna(value):
        return None
    return {"bool": bool, "float": float, "str": str}[kind](value)


# Frame parsing and dataset storage.

def _features(kinds: dict[str, str]) -> Features:
    """Build the dataset schema, including label and confidence columns for each track."""
    columns: dict[str, Any] = {
        "recording": Value("string"),
        "room": Value("string"),
        "frame_idx": Value("int32"),  # Stored row order; radar_frame_num can wrap and repeat.
        "timestamp": Value("timestamp[us]"),
        "radar_frame_num": Value("int32"),
        "image": Image(),  # Camera JPEG, decoded on access.
        "radar_payload": Value("binary"),  # Raw radar bytes retained for later parsing.
        "heatmap": Sequence(Sequence(Value("uint16"))),  # (range_bins, doppler_bins).
        "heatmap_ok": Value("bool"),  # False when the parsed heatmap is empty.
        "pointcloud": Sequence(Sequence(Value("float32"))),  # (N, 7): x, y, z, Doppler, SNR, noise, track.
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
    """Yield parsed frames with timestamp-aligned labels for Dataset.from_generator.

    The generator splits the recordings list between workers when multiprocessing
    is enabled. Camera JPEGs and radar payloads are stored alongside parsed features.

    Raises:
        RuntimeError: If the parsed frame count differs from the manifest.
    """
    from loguru import logger as loguru_logger
    from vt_radar.config.ti_config import ConfigFile
    from vt_radar.data.dataset import read_directory

    loguru_logger.remove()  # Suppress per-frame loader output; this module logs per recording.

    for rec in recordings:
        rec_dir = Path(rec["path"])
        cfg_path = sorted(rec_dir.glob("*.cfg"))[0]
        cfg = ConfigFile()
        cfg.parseCfg(str(cfg_path))
        s = cfg.sensor()

        # The reader skips unreadable files; check against the expected frame count.
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
                if df is None:  # This recording has no such label track.
                    row[f"label_{track}"] = row[f"label_{track}_confidence"] = None
                    continue
                row[f"label_{track}"] = _cast(df["label"].iloc[i], kind)
                row[f"label_{track}_confidence"] = _cast(df["confidence"].iloc[i], "float") if "confidence" in df else None
            yield row


def _recordings(root: Path, only: list[str]) -> list[dict[str, Any]]:
    """List manifest recordings whose directories are present on disk.

    Args:
        root: Raw dataset root containing MANIFEST.csv and data/.
        only: Recording names to keep; an empty list selects all available recordings.

    Returns:
        Recording name, room, path and frame count for each available recording.

    Raises:
        ValueError: If a requested name is absent from MANIFEST.csv.
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
    """Parse raw recordings, join manual annotations and save the baked dataset.

    Args:
        cfg: Raw, annotation and output paths, resolved from the repository root.
        num_proc: Maximum number of recording-parsing workers.
        overwrite: Whether to remove an existing output before rebuilding it.
        recordings: Recording names to bake; None or an empty list selects all.

    Raises:
        ValueError: If num_proc is not positive or a recording name is unknown.
        SystemExit: If the output exists without overwrite, or no recordings are found.
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

    cache = out.parent / f".{out.name}_cache"  # Keep the build cache beside the output.
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
    """Update manual labels in an existing bake without parsing raw recordings.

    Save a sibling copy and release Arrow memory maps before renaming directories,
    as required on Windows. Restore the original if installing the copy fails.

    Args:
        cfg: Baked dataset and annotation paths, resolved from the repository root.

    Raises:
        FileNotFoundError: If there is no baked dataset.
        FileExistsError: If a temporary or backup recovery directory already exists.
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
            gc.collect()  # Release Arrow memory maps before renaming folders on Windows.
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
    """Parse bake flags and data overrides, then build or refresh the dataset."""
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
