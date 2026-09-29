"""Bake the raw floor-posture recordings into one Hugging Face dataset.

Run it:

    python -m data.extract [data.root=...] [num_proc=4] [overwrite=true] [test_room=room2] ['test_persons=[Dave]'] ['val_persons=[Igor]'] ['recordings=["a","b"]']

Usage:
    ds = load_from_disk("data/processed/floor_posture")
    ds = ds.select_columns(["recording", "heatmap", "label_pose"])
"""

import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from utils.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
ALIGN_TOLERANCE = pd.Timedelta("100ms")  # ~2 frames at 18 fps
PERSON_MAP = REPO / "data" / "processed" / "person_map.csv"  # recording -> pseudonym, hand-labelled
NO_PERSON = "NA"


@dataclass
class ExtractConfig:
    data: DataConfig = field(default_factory=DataConfig)
    num_proc: int = 4
    overwrite: bool = False
    test_room: str = "room2"  # room held out for the room-wise split
    # persons held out for the person-wise split; Bob has ~46% of the frames, so these two give ~10% each
    test_persons: list[str] = field(default_factory=lambda: ["Dave"])
    val_persons: list[str] = field(default_factory=lambda: ["Igor"])
    recordings: list[str] = field(default_factory=list)  # empty = every recording present

    def __post_init__(self) -> None:
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)


# --- labels ---------------------------------------------------------------
# Label CSVs are messy: each recording ships a different set of tracks, the
# manifest.json only lists some of them, and timestamps come in three flavours
# (ISO strings, epoch-ns of local wall-clock, epoch-seconds shifted by the
# timezone). Rows are one-per-frame in order, so we align by position and only
# fall back to timestamps for partial tracks.

def _label_files(rec_dir: Path) -> list[Path]:
    return sorted((rec_dir / "labels").glob("*.csv"))


def _kind(labels: pd.Series) -> str | None:
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
    for path in _label_files(rec_dir):
        df = pd.read_csv(path)
        label_ts = _local_time(df["timestamp"]) if "timestamp" in df else None
        if len(df) == len(frame_ts):
            if label_ts is not None and np.abs(label_ts.values - frame_ts.values).max() > np.timedelta64(1, "s"):
                logger.warning("%s/%s: labels drift from frame timestamps by >1s", rec_dir.name, path.stem)
            aligned[path.stem] = df
        elif label_ts is None:
            logger.warning("%s/%s: %d rows for %d frames and no usable timestamps, skipped",
                           rec_dir.name, path.stem, len(df), len(frame_ts))
        else:
            left = pd.DataFrame({"t": frame_ts.values, "i": np.arange(len(frame_ts))}).sort_values("t")
            right = pd.DataFrame({
                "t": label_ts.values,
                "label": df["label"].values,
                "confidence": df["confidence"].values if "confidence" in df else np.nan,
            }).sort_values("t")
            merged = pd.merge_asof(left, right, on="t", direction="nearest", tolerance=ALIGN_TOLERANCE)
            aligned[path.stem] = merged.sort_values("i").reset_index(drop=True)
    return aligned


def _cast(value: Any, kind: str) -> Any:
    if pd.isna(value):
        return None
    return {"bool": bool, "float": float, "str": str}[kind](value)


# --- frames ---------------------------------------------------------------

def _features(kinds: dict[str, str]) -> Features:
    columns: dict[str, Any] = {
        "recording": Value("string"),
        "room": Value("string"),
        "frame_idx": Value("int32"),  # position in the recording; radar_frame_num wraps and repeats
        "timestamp": Value("timestamp[us]"),
        "radar_frame_num": Value("int32"),
        "image": Image(),  # camera JPEG, decoded lazily
        "radar_payload": Value("binary"),  # raw TLV bytes, enough to re-parse anything
        "heatmap": Sequence(Sequence(Value("uint16"))),  # (range_bins, doppler_bins), varies per recording
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
            row: dict[str, Any] = {
                "recording": rec["recording"],
                "room": rec["room"],
                "frame_idx": i,
                "timestamp": frame_ts.iloc[i].to_pydatetime(),
                "radar_frame_num": out.frame_num,
                "image": {"bytes": f.raw_image, "path": None} if f.raw_image else None,
                "radar_payload": f.frame,
                "heatmap": np.asarray(out.range_doppler_heatmap),
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


def _add_person(ds: Dataset) -> Dataset:
    """Add a `person` column from person_map.csv; unlabelled recordings get "NA"."""
    df = pd.read_csv(PERSON_MAP, keep_default_na=False)  # keep_default_na: don't turn a name into NaN
    mapping = {r: p.strip() or NO_PERSON for r, p in zip(df.recording, df.person)}
    recs = ds.select_columns(["recording"]).to_pandas().recording
    unlisted = sorted(set(recs) - set(mapping))
    if unlisted:
        logger.warning("not in %s, person set to %s: %s", PERSON_MAP.name, NO_PERSON, unlisted)
    return ds.add_column("person", recs.map(lambda r: mapping.get(r, NO_PERSON)).tolist())


def _split(ds: Dataset, is_test: np.ndarray, is_val: np.ndarray) -> DatasetDict:
    train = ~(is_test | is_val)
    return DatasetDict({
        "train": ds.select(np.flatnonzero(train)),
        "val": ds.select(np.flatnonzero(is_val)),
        "test": ds.select(np.flatnonzero(is_test)),
    })


def _split_by_room(ds: Dataset, test_room: str) -> DatasetDict:
    """Train/test only: with two rooms there is no room left over for val."""
    rooms = np.asarray(ds["room"])
    if test_room not in rooms:
        raise ValueError(f"test_room {test_room!r} not in dataset rooms {sorted(set(rooms))}")
    is_test = rooms == test_room
    return DatasetDict({
        "train": ds.select(np.flatnonzero(~is_test)),
        "test": ds.select(np.flatnonzero(is_test)),
    })


def _split_by_person(ds: Dataset, test_persons: list[str], val_persons: list[str]) -> DatasetDict:
    """Hold out whole persons. Unlabelled ("NA") recordings always stay in train."""
    persons = np.asarray(ds["person"])
    named = set(persons) - {NO_PERSON}
    unknown = (set(test_persons) | set(val_persons)) - named
    if unknown or set(test_persons) & set(val_persons):
        raise ValueError(f"bad test/val persons {test_persons}/{val_persons}; have {sorted(named)}")
    return _split(ds, np.isin(persons, test_persons), np.isin(persons, val_persons))


def split_paths(out: Path) -> dict[str, Path]:
    return {"room": out.with_name(out.name + "_room_split"), "person": out.with_name(out.name + "_person_split")}


def _recordings(root: Path, only: list[str]) -> list[dict[str, Any]]:
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


def extract(cfg: ExtractConfig) -> None:
    root = REPO / cfg.data.root
    out = REPO / cfg.data.baked
    if out.exists():
        if not cfg.overwrite:
            logger.error("%s already exists. Pass overwrite=true to rebuild it.", out)
            sys.exit(1)
        shutil.rmtree(out)
        for path in split_paths(out).values():
            shutil.rmtree(path, ignore_errors=True)

    recs = _recordings(root, cfg.recordings)
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
        num_proc=min(cfg.num_proc, len(recs)) if cfg.num_proc > 1 else None,
        cache_dir=str(cache),
    )
    ds = _add_person(ds)
    ds.save_to_disk(str(out))
    splits = {"room": _split_by_room(ds, cfg.test_room), "person": _split_by_person(ds, cfg.test_persons, cfg.val_persons)}
    for name, path in split_paths(out).items():
        splits[name].save_to_disk(str(path))
        logger.info("%s split -> %s: %s", name, path, {k: len(v) for k, v in splits[name].items()})
    shutil.rmtree(cache, ignore_errors=True)
    logger.info("saved %d rows to %s", len(ds), out)


def main() -> None:
    """main function"""
    schema = OmegaConf.structured(ExtractConfig)
    file_cfg = OmegaConf.load(REPO / "config" / "config.yaml")
    cli_cfg = OmegaConf.from_cli()
    try:
        merged = OmegaConf.merge(schema, file_cfg, cli_cfg)
        raw = cast(dict[str, Any], OmegaConf.to_container(merged, resolve=True))
        cfg = ExtractConfig(**raw)
    except (TypeError, OmegaConfBaseException) as e:  # pylint: disable=broad-exception-raised
        logger.error("Error: %s\n\nUsage: python -m data.extract data.root=<path> num_proc=4", e)
        sys.exit(1)

    extract(cfg)


if __name__ == "__main__":
    main()
