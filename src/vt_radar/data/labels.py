"""Unified label infrastructure for multi-track, multi-modal, timestamp-based labels.

Supports multiple label tracks per dataset (presence, activity, pose, etc.),
each with its own schema and source. Labels are aligned to radar frames via
nearest-timestamp matching rather than index-based 1:1 mapping.

Storage layout::

    dataset_dir/
      labels/
        manifest.json          # describes all label tracks
        presence.csv           # track: presence
        activity.csv           # track: activity

Example usage::

    from vt_radar.data.labels import Labels

    # Load from dataset directory
    labels = Labels.load("path/to/dataset")
    print(labels.track_names)

    # Align to radar timestamps
    aligned = labels.align_to_timestamps(frame_timestamps)

    # Single track access
    labels["presence"].at_timestamp(some_ts)
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class LabelSchema:
    """Defines validation rules for a label track.

    Attributes:
        name: Schema identifier (e.g. "presence", "activity", "custom").
        label_dtype: Expected dtype for the label column ("bool", "int", "str", "float").
        valid_values: Allowed label values. None means any value of the dtype is OK.
        extra_columns: Additional columns and their dtypes (e.g. {"confidence": "float"}).
    """

    name: str
    label_dtype: str
    valid_values: Optional[List[Any]] = None
    extra_columns: Dict[str, str] = field(default_factory=dict)
    multi_row: bool = False


# Built-in schemas
SCHEMAS: Dict[str, LabelSchema] = {
    "presence": LabelSchema(
        "presence", "bool", [True, False], {"confidence": "float"}
    ),
    "people_count": LabelSchema("people_count", "int"),
    "activity": LabelSchema(
        "activity",
        "str",
        ["walking", "sitting", "lying_down", "fall", "empty_scene"],
        {"confidence": "float"},
    ),
    "fall": LabelSchema("fall", "bool", [True, False], {"confidence": "float"}),
    "pose": LabelSchema(
        "pose",
        "str",
        [
            "standing",
            "sitting",
            "lying_left",
            "lying_right",
            "lying_back",
            "lying_front",
            "moving",
        ],
        {"confidence": "float"},
    ),
    "room_state": LabelSchema(
        "room_state",
        "str",
        ["empty", "occupied", "in_bed", "on_bed_edge", "in_chair", "fallen"],
        {"confidence": "float", "person_count": "int"},
    ),
    "object_interaction": LabelSchema(
        "object_interaction",
        "str",
        [
            "none",
            "bed_lying",
            "bed_sitting",
            "bed_edge",
            "chair_sitting",
            "floor_lying",
        ],
        {"confidence": "float", "zone_id": "str"},
    ),
    "transition": LabelSchema(
        "transition",
        "str",
        [
            "stable",
            "room_enter",
            "room_leave",
            "bed_enter",
            "bed_leave",
            "chair_sit",
            "chair_stand",
            "fall",
            "fall_recover",
        ],
        {"confidence": "float", "is_event": "bool"},
    ),
    "localization": LabelSchema(
        "localization",
        "int",
        valid_values=None,
        extra_columns={"x": "float", "y": "float", "confidence": "float"},
        multi_row=True,
    ),
    "tracking": LabelSchema(
        "tracking",
        "int",
        valid_values=None,
        extra_columns={"x": "float", "y": "float", "confidence": "float"},
        multi_row=True,
    ),
    "person_identity": LabelSchema(
        "person_identity",
        "str",
        valid_values=None,
        extra_columns={
            "track_id": "int",
            "x": "float",
            "y": "float",
            "confidence": "float",
        },
        multi_row=True,
    ),
    "orientation": LabelSchema(
        name="orientation",
        label_dtype="float",
        valid_values=None,
        extra_columns={"confidence": "float", "source": "str"},
    ),
}

MANIFEST_FILENAME = "manifest.json"
LABELS_DIR = "labels"

# Mapping from dtype strings to pandas/Python types for validation
_DTYPE_MAP: Dict[str, type] = {
    "bool": bool,
    "int": int,
    "str": str,
    "float": float,
}


def _coerce_label_column(series: pd.Series, dtype_str: str) -> pd.Series:
    """Coerce label column to the expected dtype."""
    if dtype_str == "bool":
        # Handle string representations of booleans
        if series.dtype == object:
            mapping = {"true": True, "false": False, "1": True, "0": False}
            mapped = series.astype(str).str.lower().map(mapping)
            # Unmapped values become NaN; a blind .astype(bool) turns NaN into
            # True, silently corrupting invalid labels (they'd then pass
            # validation). Reject them instead, per from_csv's ValueError contract.
            if mapped.isna().any():
                bad = series[mapped.isna()].unique().tolist()
                raise ValueError(
                    f"Invalid boolean label values: {bad}. "
                    "Expected one of: true, false, 1, 0."
                )
            return mapped.astype(bool)
        return series.astype(bool)
    elif dtype_str == "int":
        return series.astype(int)
    elif dtype_str == "float":
        return series.astype(float)
    else:
        return series.astype(str)


def _validate_labels(data: pd.DataFrame, schema: LabelSchema) -> None:
    """Validate a label DataFrame against a schema.

    Raises:
        ValueError: If validation fails.
    """
    has_timestamp = "timestamp" in data.columns
    has_frame_num = "frame_num" in data.columns
    if not has_timestamp and not has_frame_num:
        raise ValueError(
            "Label data must have at least one of 'timestamp' or 'frame_num' columns"
        )
    if "label" not in data.columns:
        raise ValueError("Label data must have a 'label' column")

    if schema.valid_values is not None:
        invalid = ~data["label"].isin(schema.valid_values)
        if invalid.any():
            bad = data.loc[invalid, "label"].unique().tolist()
            raise ValueError(
                f"Invalid label values for schema '{schema.name}': {bad}. "
                f"Valid values: {schema.valid_values}"
            )


def _infer_schema_from_columns(df: pd.DataFrame) -> str:
    """Try to infer schema name from CSV column names and values."""
    if "label" not in df.columns:
        return "custom"

    values = set(df["label"].dropna().unique())

    # Check each built-in schema
    for name, schema in SCHEMAS.items():
        if schema.valid_values is not None:
            valid_set = set(schema.valid_values)
            if values and values.issubset(valid_set):
                # {0, 1} == {False, True} in Python (0 == False, 1 == True), so an
                # integer people_count that only reaches 0/1 (single-occupancy
                # room) would spuriously match the bool 'presence' schema. Require
                # genuine bool values before matching a bool-dtype schema.
                if schema.label_dtype == "bool" and not all(
                    isinstance(v, (bool, np.bool_)) for v in values
                ):
                    continue
                return name

    # Fallback: infer from dtype
    sample = df["label"].dropna()
    if len(sample) == 0:
        return "custom"

    # Check if all values are integers. astype(int) TRUNCATES floats rather than
    # raising, so a genuinely float-valued label column (e.g. [0.5, 1.7]) would
    # otherwise pass here and be mis-inferred as the int 'people_count' schema —
    # then silently truncated on load by from_csv's int coercion. Only classify
    # as people_count when the int round-trip is lossless.
    try:
        as_int = sample.astype(int)
        if (as_int == sample).all():
            return "people_count"
    except (ValueError, TypeError):
        pass

    return "custom"


@dataclass
class LabelTrack:
    """One label track -- a timestamped series of labels of a single type.

    Attributes:
        name: Track name (e.g. "presence", "pose_video").
        schema: Validation schema for this track.
        source: Origin of the labels ("radar", "video", "manual", "algorithm").
        data: DataFrame with columns: timestamp, label, [confidence], [extras].
        description: Human-readable description of this track.
    """

    name: str
    schema: LabelSchema
    source: str
    data: pd.DataFrame
    description: str = ""

    def at_timestamp(
        self, ts: Any, tolerance_ms: float = 100
    ) -> Optional[Any]:
        """Get the label closest to a timestamp within tolerance.

        Args:
            ts: Target timestamp (datetime-like, parsed by pd.Timestamp).
            tolerance_ms: Maximum allowed time difference in milliseconds.

        Returns:
            The label value, or None if no label is within tolerance.
        """
        # Empty track has no label within tolerance; guard like
        # rows_at_timestamp/at_timestamps do (idxmin() raises on empty).
        # Frame-only tracks lack a "timestamp" column — degrade gracefully like
        # at_frame() does for a missing "frame_num" column (else KeyError).
        if self.data.empty or "timestamp" not in self.data.columns:
            return None

        target = pd.Timestamp(ts)
        timestamps = self.data["timestamp"]
        diffs = (timestamps - target).abs()
        idx = diffs.idxmin()
        if diffs.loc[idx] <= pd.Timedelta(milliseconds=tolerance_ms):
            return self.data.loc[idx, "label"]
        return None

    def rows_at_timestamp(
        self, ts: Any, tolerance_ms: float = 100
    ) -> pd.DataFrame:
        """Get ALL rows closest to a timestamp within tolerance.

        Unlike ``at_timestamp`` (which returns a single scalar label), this
        returns a DataFrame slice — useful for multi-row tracks like
        ``localization`` and ``tracking`` where multiple rows share the same
        timestamp.

        Args:
            ts: Target timestamp (datetime-like, parsed by pd.Timestamp).
            tolerance_ms: Maximum allowed time difference in milliseconds.

        Returns:
            DataFrame of matching rows (may be empty).
        """
        # Empty or frame-only track (no "timestamp" column) -> no matches, mirror
        # rows_at_frame()'s graceful empty return rather than KeyError.
        if self.data.empty or "timestamp" not in self.data.columns:
            return self.data.iloc[0:0]

        target = pd.Timestamp(ts)
        timestamps = self.data["timestamp"]
        diffs = (timestamps - target).abs()
        best_diff = diffs.min()

        if best_diff > pd.Timedelta(milliseconds=tolerance_ms):
            return self.data.iloc[0:0]

        # Return all rows at the nearest timestamp
        nearest_ts = timestamps.loc[diffs.idxmin()]
        return self.data[timestamps == nearest_ts].reset_index(drop=True)

    def at_frame(self, frame_num: int) -> Optional[Any]:
        """Get the label at an exact frame number.

        Args:
            frame_num: Target frame number.

        Returns:
            The label value, or None if no label exists for this frame.
        """
        if "frame_num" not in self.data.columns:
            return None
        match = self.data.loc[self.data["frame_num"] == frame_num]
        if match.empty:
            return None
        return match.iloc[0]["label"]

    def at_frames(self, frame_nums: pd.Series) -> pd.Series:
        """Vectorized exact frame-number lookup.

        Args:
            frame_nums: Series of target frame numbers.

        Returns:
            Series of label values, NaN where no label exists for that frame.
        """
        if "frame_num" not in self.data.columns or frame_nums.empty:
            return pd.Series(
                [np.nan] * len(frame_nums), index=frame_nums.index, dtype=object
            )

        # drop_duplicates keeps the first row per frame, matching at_frame()'s .iloc[0].
        # Without it, multi-row tracks (several targets per frame) give a non-unique index
        # and Series.map raises InvalidIndexError.
        lookup = self.data.drop_duplicates("frame_num").set_index("frame_num")["label"]
        return frame_nums.map(lookup)

    def rows_at_frame(self, frame_num: int) -> pd.DataFrame:
        """Get ALL rows at an exact frame number.

        Useful for multi-row tracks (localization, tracking).

        Args:
            frame_num: Target frame number.

        Returns:
            DataFrame of matching rows (may be empty).
        """
        if "frame_num" not in self.data.columns:
            return self.data.iloc[0:0]
        return self.data[self.data["frame_num"] == frame_num].reset_index(drop=True)

    def at_timestamps(
        self, timestamps: pd.Series, tolerance_ms: float = 100
    ) -> pd.Series:
        """Vectorized nearest-timestamp lookup for a batch of timestamps.

        Args:
            timestamps: Series of target timestamps.
            tolerance_ms: Maximum allowed time difference in milliseconds.

        Returns:
            Series of label values, NaN where no label is within tolerance.
        """
        # Frame-only track (no "timestamp" column) yields all-NaN, mirroring
        # at_frames()'s guard for a missing "frame_num" column (else KeyError).
        # This keeps align_to_timestamps() from crashing when one track among
        # several is frame-based.
        if self.data.empty or timestamps.empty or "timestamp" not in self.data.columns:
            return pd.Series([np.nan] * len(timestamps), index=timestamps.index, dtype=object)

        track_ts = self.data["timestamp"].values.astype("datetime64[ns]")
        target_ts = pd.to_datetime(timestamps).values.astype("datetime64[ns]")

        # For each target, find nearest track timestamp via searchsorted
        indices = np.searchsorted(track_ts, target_ts, side="left")
        indices = np.clip(indices, 0, len(track_ts) - 1)

        # Check both the found index and the one before it
        results = pd.Series([np.nan] * len(timestamps), index=timestamps.index, dtype=object)
        tolerance = np.timedelta64(int(tolerance_ms * 1e6), "ns")

        for i, idx in enumerate(indices):
            best_diff = abs(track_ts[idx] - target_ts[i])
            best_idx = idx

            if idx > 0:
                prev_diff = abs(track_ts[idx - 1] - target_ts[i])
                if prev_diff < best_diff:
                    best_diff = prev_diff
                    best_idx = idx - 1

            if best_diff <= tolerance:
                results.iloc[i] = self.data.iloc[best_idx]["label"]

        return results

    def to_csv(self, path: str) -> None:
        """Write track data to CSV."""
        self.data.to_csv(path, index=False)

    @classmethod
    def from_csv(
        cls,
        path: str,
        name: str,
        schema: LabelSchema,
        source: str = "unknown",
        description: str = "",
    ) -> "LabelTrack":
        """Load from CSV file.

        The CSV must have at least 'timestamp' and 'label' columns.

        Args:
            path: Path to the CSV file.
            name: Track name.
            schema: Schema to validate against.
            source: Label source identifier.
            description: Human-readable description.

        Returns:
            LabelTrack instance.

        Raises:
            ValueError: If the CSV is invalid or fails schema validation.
        """
        df = pd.read_csv(path)

        has_timestamp = "timestamp" in df.columns
        has_frame_num = "frame_num" in df.columns

        if not has_timestamp and not has_frame_num:
            raise ValueError(
                f"CSV {path} missing required columns: need at least "
                "'timestamp' or 'frame_num'"
            )
        if "label" not in df.columns:
            raise ValueError(f"CSV {path} missing required 'label' column")

        if has_timestamp:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if has_frame_num:
            df["frame_num"] = df["frame_num"].astype(int)

        # Sort by frame_num if available, otherwise by timestamp
        if has_frame_num:
            df = df.sort_values("frame_num").reset_index(drop=True)
        else:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Coerce label dtype
        if schema.name != "custom":
            df["label"] = _coerce_label_column(df["label"], schema.label_dtype)

        _validate_labels(df, schema)

        return cls(
            name=name,
            schema=schema,
            source=source,
            data=df,
            description=description,
        )


class Labels:
    """Multi-track label container. The main entry point for label access.

    Example::

        labels = Labels.load("/path/to/dataset")
        print(labels.track_names)
        aligned = labels.align_to_timestamps(frame_timestamps)
    """

    def __init__(self, tracks: Optional[Dict[str, LabelTrack]] = None) -> None:
        self._tracks: Dict[str, LabelTrack] = tracks or {}

    # --- Loading ---

    @classmethod
    def load(cls, path: str) -> "Labels":
        """Load labels from a dataset directory or labels/ subdirectory.

        Tries in order:
        1. path/labels/manifest.json -> load all tracks from manifest
        2. path/labels/\\*.csv -> auto-detect tracks from filenames
        3. path/\\*.csv (legacy) -> single track, schema inferred

        Args:
            path: Dataset directory path.

        Returns:
            Labels instance (may be empty if no labels found).
        """
        labels_dir = os.path.join(path, LABELS_DIR)

        # 1. Manifest-based loading
        manifest_path = os.path.join(labels_dir, MANIFEST_FILENAME)
        if os.path.isfile(manifest_path):
            return cls.from_manifest(manifest_path)

        # 2. CSV files in labels/ directory (auto-detect)
        if os.path.isdir(labels_dir):
            import glob as globmod

            csv_files = sorted(globmod.glob(os.path.join(labels_dir, "*.csv")))
            if csv_files:
                return cls._load_csvs_auto(csv_files)

        # 3. Legacy: CSV files in dataset root
        import glob as globmod

        csv_files = sorted(globmod.glob(os.path.join(path, "*.csv")))
        if csv_files:
            return cls._load_csvs_auto(csv_files)

        return cls()

    @classmethod
    def from_manifest(cls, manifest_path: str) -> "Labels":
        """Load from an explicit manifest.json.

        Args:
            manifest_path: Path to manifest.json file.

        Returns:
            Labels instance with all tracks defined in the manifest.
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        labels_dir = os.path.dirname(manifest_path)
        tracks: Dict[str, LabelTrack] = {}

        for track_name, track_info in manifest.get("tracks", {}).items():
            csv_file = track_info.get("file")
            if csv_file is None:
                logger.warning(f"Track '{track_name}' has no 'file' field, skipping")
                continue

            csv_path = os.path.join(labels_dir, csv_file)
            if not os.path.isfile(csv_path):
                logger.warning(f"Track '{track_name}' file not found: {csv_path}")
                continue

            schema_name = track_info.get("schema", "custom")
            source = track_info.get("source", "unknown")
            description = track_info.get("description", "")

            if schema_name in SCHEMAS:
                schema = SCHEMAS[schema_name]
            else:
                # Custom schema from manifest
                columns = track_info.get("columns", {})
                schema = LabelSchema(
                    name="custom",
                    label_dtype=columns.get("label", "str"),
                    extra_columns={
                        k: v for k, v in columns.items() if k != "label"
                    },
                )

            try:
                track = LabelTrack.from_csv(
                    csv_path, track_name, schema, source, description
                )
                tracks[track_name] = track
            except Exception as e:
                logger.warning(f"Failed to load track '{track_name}': {e}")

        return cls(tracks)

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        track_name: str = "default",
        schema: str = "custom",
        source: str = "unknown",
    ) -> "Labels":
        """Load a single CSV as a single-track Labels object.

        Args:
            csv_path: Path to the CSV file.
            track_name: Name for the track.
            schema: Schema name (built-in) or "custom".
            source: Label source identifier.

        Returns:
            Labels instance with a single track.
        """
        if schema in SCHEMAS:
            schema_obj = SCHEMAS[schema]
        else:
            schema_obj = LabelSchema("custom", "str")

        track = LabelTrack.from_csv(csv_path, track_name, schema_obj, source)
        return cls({track_name: track})

    @classmethod
    def _load_csvs_auto(cls, csv_files: List[str]) -> "Labels":
        """Auto-detect tracks from a list of CSV files."""
        tracks: Dict[str, LabelTrack] = {}

        for csv_path in csv_files:
            basename = os.path.splitext(os.path.basename(csv_path))[0]

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                logger.warning(f"Could not read {csv_path}: {e}")
                continue

            has_ts = "timestamp" in df.columns
            has_fn = "frame_num" in df.columns
            if (not has_ts and not has_fn) or "label" not in df.columns:
                logger.debug(f"Skipping {csv_path}: missing required columns")
                continue

            # Try to match filename to a built-in schema
            schema_name = basename if basename in SCHEMAS else _infer_schema_from_columns(df)

            if schema_name in SCHEMAS:
                schema = SCHEMAS[schema_name]
            else:
                schema = LabelSchema("custom", "str")

            try:
                track = LabelTrack.from_csv(csv_path, basename, schema, "unknown")
                tracks[basename] = track
            except Exception as e:
                logger.warning(f"Failed to auto-load {csv_path}: {e}")

        return cls(tracks)

    # --- Access ---

    @property
    def track_names(self) -> List[str]:
        """List available track names."""
        return list(self._tracks.keys())

    def __getitem__(self, track_name: str) -> LabelTrack:
        """Get a track by name. Raises KeyError if not found."""
        return self._tracks[track_name]

    def __contains__(self, track_name: str) -> bool:
        return track_name in self._tracks

    def __len__(self) -> int:
        return len(self._tracks)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._tracks)

    # --- Alignment ---

    def align_to_timestamps(
        self,
        timestamps: pd.Series,
        tracks: Optional[List[str]] = None,
        tolerance_ms: float = 100,
    ) -> pd.DataFrame:
        """Align one or more tracks to a series of timestamps.

        Returns a DataFrame with one row per timestamp and one column per track.
        Values are NaN where no label falls within tolerance.

        Args:
            timestamps: Series of target timestamps.
            tracks: Track names to align. None means all tracks.
            tolerance_ms: Maximum allowed time difference in milliseconds.

        Returns:
            DataFrame with aligned labels.
        """
        # Only None means "all tracks"; an explicit empty list must select no tracks
        # (`tracks or self.track_names` wrongly treated [] as all).
        track_names = self.track_names if tracks is None else tracks
        result: Dict[str, pd.Series] = {}

        for name in track_names:
            if name not in self._tracks:
                logger.warning(f"Track '{name}' not found, skipping")
                continue
            result[name] = self._tracks[name].at_timestamps(timestamps, tolerance_ms)

        return pd.DataFrame(result, index=timestamps.index)

    def align_to_frames(
        self,
        frame_nums: pd.Series,
        tracks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Align one or more tracks to a series of frame numbers (exact join).

        Returns a DataFrame with one row per frame number and one column per track.
        Values are NaN where no label exists for that frame.

        Args:
            frame_nums: Series of target frame numbers.
            tracks: Track names to align. None means all tracks.

        Returns:
            DataFrame with aligned labels.
        """
        # Only None means "all tracks"; an explicit empty list must select no tracks
        # (`tracks or self.track_names` wrongly treated [] as all).
        track_names = self.track_names if tracks is None else tracks
        result: Dict[str, pd.Series] = {}

        for name in track_names:
            if name not in self._tracks:
                logger.warning(f"Track '{name}' not found, skipping")
                continue
            result[name] = self._tracks[name].at_frames(frame_nums)

        return pd.DataFrame(result, index=frame_nums.index)

    # --- Saving ---

    def save(self, path: str) -> None:
        """Save all tracks to path/labels/ with manifest.json.

        Args:
            path: Dataset directory path. Labels are saved to path/labels/.
        """
        labels_dir = os.path.join(path, LABELS_DIR)
        os.makedirs(labels_dir, exist_ok=True)

        manifest: Dict[str, Any] = {"version": 1, "tracks": {}}

        for name, track in self._tracks.items():
            csv_filename = f"{name}.csv"
            csv_path = os.path.join(labels_dir, csv_filename)
            track.to_csv(csv_path)

            track_info: Dict[str, Any] = {
                "file": csv_filename,
                "schema": track.schema.name,
                "source": track.source,
            }
            if track.description:
                track_info["description"] = track.description
            if track.schema.name == "custom" and track.schema.extra_columns:
                track_info["columns"] = {
                    "label": track.schema.label_dtype,
                    **track.schema.extra_columns,
                }

            manifest["tracks"][name] = track_info

        manifest_path = os.path.join(labels_dir, MANIFEST_FILENAME)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    # --- Track management ---

    def add_track(self, track: LabelTrack) -> None:
        """Add or replace a label track."""
        self._tracks[track.name] = track

    def remove_track(self, name: str) -> None:
        """Remove a track by name.

        Raises:
            KeyError: If the track does not exist.
        """
        del self._tracks[name]


# --- Versioned run storage -------------------------------------------

import json as _json
import shutil as _shutil
from pathlib import Path


class LabelRunStore:
    """
    Writes auto-label outputs to versioned run directories.

    Layout:
        dataset_dir/labels/
            manifest.json          # {"task": "run_id", ...}  <- never deleted entries
            presence.csv           # canonical copy of best run
            orientation.csv
            ...
            runs/
                <run_id>/
                    presence.csv
                    confidence_map.parquet
                manifest.json      # run manifest, co-located with runs/
    """

    def __init__(self, dataset_dir: "Union[str, Path]", run_id: str) -> None:
        self._labels_dir = Path(dataset_dir) / "labels"
        self._run_dir = self._labels_dir / "runs" / run_id
        self._manifest_path = self._labels_dir / "runs" / "manifest.json"
        self._run_id = run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def write_track(self, task: str, df: "pd.DataFrame") -> None:
        """Write label CSV to the run directory and the canonical labels/ location."""
        run_csv = self._run_dir / f"{task}.csv"
        df.to_csv(run_csv, index=False)
        canonical = self._labels_dir / f"{task}.csv"
        _shutil.copy2(run_csv, canonical)
        self._update_manifest(task)

    def write_confidence_map(self, confidence_map: Any) -> None:
        """Write a ConfidenceMap to the run directory as a parquet file."""
        confidence_map.save(self._run_dir / "confidence_map.parquet")

    def _update_manifest(self, task: str) -> None:
        manifest: Dict[str, str] = {}
        if self._manifest_path.exists():
            manifest = _json.loads(self._manifest_path.read_text())
        manifest[task] = self._run_id
        self._manifest_path.write_text(_json.dumps(manifest, indent=2))


def add_timestamps_from_dataset(
    track: LabelTrack, dataset: Any
) -> LabelTrack:
    """Backfill timestamps onto a frame_num-based LabelTrack using a dataset.

    Maps ``frame_num → timestamp`` by iterating the dataset's frames. This
    bridges frame-based ground truth labels to timestamp-based evaluators.

    Args:
        track: A LabelTrack whose data has a ``frame_num`` column.
        dataset: An iterable of frame objects with ``.frame_num`` and
            ``.timestamp`` attributes (e.g. ``TLVDataset``).

    Returns:
        A new LabelTrack with the ``timestamp`` column filled in.
    """
    if "frame_num" not in track.data.columns:
        raise ValueError("Track data has no 'frame_num' column to map from")

    # Build frame_num -> timestamp mapping from dataset
    frame_to_ts: Dict[int, Any] = {}
    for frame in dataset:
        frame_to_ts[frame.frame_num] = frame.timestamp

    new_data = track.data.copy()
    new_data["timestamp"] = new_data["frame_num"].map(frame_to_ts)

    unmapped = new_data["timestamp"].isna().sum()
    if unmapped > 0:
        logger.warning(
            f"add_timestamps_from_dataset: {unmapped} frame_nums had no "
            f"matching timestamp in dataset"
        )

    return LabelTrack(
        name=track.name,
        schema=track.schema,
        source=track.source,
        data=new_data,
        description=track.description,
    )
