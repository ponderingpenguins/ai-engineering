"""Content-addressed cache for algorithm pipeline outputs.

Each algorithm in a DAG pipeline stores its output in a per-node LMDB.
Cache keys are Merkle hashes of the node's params + all upstream hashes,
so changing anything upstream automatically invalidates downstream caches.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import orjson

from loguru import logger

from vt_radar.data.cache import CacheManifest, LMDBStage


def compute_chain_hash(
    algorithm_name: str,
    params: Dict[str, Any],
    upstream_hashes: Sequence[str] = (),
    source_hash: str = "",
) -> str:
    """Merkle hash: algo name + params + sorted upstream hashes + source."""
    blob = {
        "algorithm": algorithm_name,
        "params": params,
        "upstreams": sorted(upstream_hashes),
        "source": source_hash,
    }
    return hashlib.sha256(
        json.dumps(blob, sort_keys=True, default=str).encode()
    ).hexdigest()


def _short_hash(full_hash: str) -> str:
    """First 12 hex chars of a full hash."""
    return full_hash[:12]


class ResultWriter:
    """Writes named numpy arrays per frame into an LMDB."""

    def __init__(self, lmdb: LMDBStage) -> None:
        self._lmdb = lmdb
        self._txn: Any = None
        self._keys: set[str] = set()
        self._n_frames = 0
        self._puts_since_commit = 0
        # Flush periodically so the LMDB map can grow between transactions
        # (maybe_grow only runs when no write txn is open).
        self._commit_interval = 500

    def put(self, frame_idx: int, **arrays: np.ndarray) -> None:
        """Store named arrays for a frame."""
        if self._txn is None:
            self._txn = self._lmdb.begin_write()
        for name, arr in arrays.items():
            self._keys.add(name)
            self._lmdb.put_array(self._txn, f"{name}:{frame_idx}", arr)
            self._lmdb.put_json(self._txn, f"{name}_meta:{frame_idx}", {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            })
        self._n_frames = max(self._n_frames, frame_idx + 1)
        self._puts_since_commit += 1
        if self._puts_since_commit >= self._commit_interval:
            self._txn.commit()
            self._lmdb.maybe_grow()
            self._txn = self._lmdb.begin_write()
            self._puts_since_commit = 0

    def close(self) -> None:
        if self._txn is not None:
            self._txn.commit()
            self._txn = None
        self._lmdb.close()

    @property
    def keys(self) -> List[str]:
        return sorted(self._keys)

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class ResultReader:
    """Reads named numpy arrays per frame from an LMDB."""

    def __init__(self, lmdb: LMDBStage, meta: Dict[str, Any]) -> None:
        self._lmdb = lmdb
        self._meta = meta

    def get(self, frame_idx: int, name: str) -> Optional[np.ndarray]:
        """Retrieve a named array for a frame."""
        meta_val = self._lmdb.get_json(f"{name}_meta:{frame_idx}")
        if meta_val is None:
            return None
        shape = tuple(meta_val["shape"])
        dtype = np.dtype(meta_val["dtype"])
        return self._lmdb.get_array(f"{name}:{frame_idx}", dtype, shape)

    def get_all(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Retrieve all named arrays for a frame."""
        result: Dict[str, np.ndarray] = {}
        for key in self._meta.get("keys", []):
            arr = self.get(frame_idx, key)
            if arr is not None:
                result[key] = arr
        return result

    @property
    def n_frames(self) -> int:
        return self._meta.get("n_frames", 0)

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta

    def close(self) -> None:
        self._lmdb.close()


class ResultCache:
    """Content-addressed cache for algorithm pipeline outputs.

    Each algorithm in a pipeline stores its output here. To optimize
    algorithm N, read algorithm N-1's cached output and iterate on N's
    hyperparameters without rerunning the upstream chain.

    Usage:
        cache = ResultCache(dataset_path)

        # First algorithm in pipeline (no upstream)
        with cache.open_write("dbscan", params={"eps_m": 0.35},
                              description="DBSCAN clustering") as writer:
            for idx in range(n_frames):
                writer.put(idx, clusters=clusters_arr, centroids=centroids_arr)

        # Read it back
        reader = cache.open_read("dbscan", params={"eps_m": 0.35})
        clusters = reader.get(0, "clusters")   # numpy array

        # Chained algorithm (PCA on top of DBSCAN)
        dbscan_hash = cache.resolve("dbscan", params={"eps_m": 0.35})
        with cache.open_write("pca", params={"n_components": 3},
                              upstreams=[dbscan_hash],
                              description="PCA on DBSCAN output") as writer:
            for idx in range(n_frames):
                writer.put(idx, features=pca_features)
    """

    def __init__(self, dataset_path: str) -> None:
        self._cache_dir = os.path.join(dataset_path, ".radar")
        self._results_dir = os.path.join(self._cache_dir, "results")
        self._manifest = CacheManifest(self._cache_dir)
        # Compute source hash from dataset for anchoring chains
        self._source_hash = self._compute_dataset_source_hash(dataset_path)

    def _compute_dataset_source_hash(self, dataset_path: str) -> str:
        """Hash the dataset path as a stable anchor. Real datasets use file hashes."""
        return hashlib.sha256(
            os.path.abspath(dataset_path).encode()
        ).hexdigest()

    def _chain_hash(
        self,
        algorithm: str,
        params: Dict[str, Any],
        upstreams: Sequence[str] = (),
    ) -> str:
        return compute_chain_hash(algorithm, params, upstreams, self._source_hash)

    def resolve(
        self,
        algorithm: str,
        params: Dict[str, Any],
        upstreams: Sequence[str] = (),
    ) -> Optional[str]:
        """Return short hash if valid cache exists, else None."""
        full_hash = self._chain_hash(algorithm, params, upstreams)
        short = _short_hash(full_hash)
        results = self._manifest.data.get("results", {})
        if short not in results:
            return None
        # Verify the LMDB directory exists
        lmdb_dir = os.path.join(self._results_dir, short)
        if not os.path.isdir(lmdb_dir):
            return None
        return short

    def open_write(
        self,
        algorithm: str,
        params: Dict[str, Any],
        description: str = "",
        upstreams: Sequence[str] = (),
    ) -> "ResultWriter":
        """Context manager that returns a ResultWriter for storing output."""
        full_hash = self._chain_hash(algorithm, params, upstreams)
        short = _short_hash(full_hash)
        lmdb_dir = os.path.join(self._results_dir, short)

        # Remove existing cache for this hash
        if os.path.isdir(lmdb_dir):
            shutil.rmtree(lmdb_dir, ignore_errors=True)

        os.makedirs(lmdb_dir, exist_ok=True)
        lmdb = LMDBStage(lmdb_dir, readonly=False)
        lmdb.open()

        writer = _ManagedResultWriter(
            lmdb=lmdb,
            manifest=self._manifest,
            algorithm=algorithm,
            params=params,
            description=description,
            upstreams=list(upstreams),
            chain_hash_full=full_hash,
            short_hash=short,
        )
        return writer

    def open_read(
        self,
        algorithm: str,
        params: Dict[str, Any],
        upstreams: Sequence[str] = (),
    ) -> ResultReader:
        """Open a cached result for reading. Raises if not cached."""
        short = self.resolve(algorithm, params, upstreams)
        if short is None:
            raise FileNotFoundError(
                f"No cached result for {algorithm} with params={params}"
            )
        # Touch last_accessed
        self._manifest.touch_accessed(short)

        lmdb_dir = os.path.join(self._results_dir, short)
        lmdb = LMDBStage(lmdb_dir, readonly=True)
        lmdb.open()

        meta = self._manifest.data.get("results", {})[short]
        return ResultReader(lmdb, meta)

    def gc_stale(self, max_age_days: int = 30) -> int:
        """Remove caches not accessed in max_age_days."""
        return self._manifest.gc_stale_results(
            self._results_dir, max_age_days
        )

    def write_predictions(
        self,
        algorithm: str,
        params: Dict[str, Any],
        predictions: "Labels",
        upstreams: Sequence[str] = (),
        description: str = "",
    ) -> str:
        """Write Labels predictions into LMDB. Returns result short hash.

        Each LabelTrack is serialized as per-frame numpy arrays:
        - ``{track_name}_label``: label values (encoded as float or int)
        - ``{track_name}_confidence``: confidence values (if present)
        """
        from vt_radar.data.labels import Labels

        with self.open_write(
            algorithm, params, description=description, upstreams=upstreams
        ) as writer:
            for track_name in predictions.track_names:
                track = predictions[track_name]
                df = track.data

                # Convert labels to numeric array
                labels = df["label"].values
                # An empty track has no rows to write; skip it (labels[0] below
                # would otherwise IndexError, and read_predictions reconstructs
                # tracks from stored keys, so a zero-row track contributes nothing).
                if len(labels) == 0:
                    continue
                if hasattr(labels[0], "__bool__") and isinstance(labels[0], (bool, np.bool_)):
                    label_arr = labels.astype(np.float32)
                elif np.issubdtype(type(labels[0]), np.integer):
                    label_arr = labels.astype(np.int32)
                elif isinstance(labels[0], str):
                    # Store strings directly as a numpy unicode array. Each
                    # frame's key records its own dtype (e.g. "<U7"), so
                    # read_predictions reconstructs the original string. The old
                    # int-encoding discarded its {label: index} mapping (the
                    # "store as metadata" comment was never implemented), so
                    # string labels (activity, pose) round-tripped as
                    # meaningless integers.
                    label_arr = np.asarray(labels, dtype=str)
                else:
                    label_arr = np.array(labels, dtype=np.float32)

                # Write one entry per frame
                for i in range(len(df)):
                    arrays: Dict[str, np.ndarray] = {
                        f"{track_name}_label": np.array([label_arr[i]]),
                    }
                    if "confidence" in df.columns:
                        conf = df["confidence"].iloc[i]
                        if conf is not None and not (isinstance(conf, float) and np.isnan(conf)):
                            arrays[f"{track_name}_confidence"] = np.array(
                                [float(conf)], dtype=np.float32
                            )
                    if "frame_num" in df.columns:
                        arrays[f"{track_name}_frame_num"] = np.array(
                            [int(df["frame_num"].iloc[i])], dtype=np.int32
                        )
                    writer.put(i, **arrays)

        short = self.resolve(algorithm, params, upstreams)
        if short is None:
            raise RuntimeError("write_predictions succeeded but resolve returned None")
        return short

    def read_predictions(self, result_hash: str) -> "Labels":
        """Read LMDB result back as a Labels object.

        Reconstructs LabelTrack objects from stored per-frame arrays.
        """
        import pandas as pd

        from vt_radar.data.labels import LabelSchema, LabelTrack, Labels

        results = self._manifest.data.get("results", {})
        if result_hash not in results:
            raise FileNotFoundError(f"Result not found: {result_hash}")

        meta = results[result_hash]
        self._manifest.touch_accessed(result_hash)

        lmdb_dir = os.path.join(self._results_dir, result_hash)
        lmdb = LMDBStage(lmdb_dir, readonly=True)
        lmdb.open()
        reader = ResultReader(lmdb, meta)

        # Discover track names from keys (e.g. "presence_label" -> "presence")
        track_names: set = set()
        for key in meta.get("keys", []):
            if key.endswith("_label"):
                track_names.add(key[:-6])

        labels = Labels()
        for track_name in sorted(track_names):
            rows = []
            for idx in range(reader.n_frames):
                label_arr = reader.get(idx, f"{track_name}_label")
                if label_arr is None:
                    continue
                row: Dict[str, Any] = {"label": label_arr.flat[0]}

                conf_arr = reader.get(idx, f"{track_name}_confidence")
                if conf_arr is not None:
                    row["confidence"] = float(conf_arr.flat[0])

                frame_arr = reader.get(idx, f"{track_name}_frame_num")
                if frame_arr is not None:
                    row["frame_num"] = int(frame_arr.flat[0])

                rows.append(row)

            if not rows:
                continue

            df = pd.DataFrame(rows)
            # Infer schema from algorithm name or use generic
            algorithm = meta.get("algorithm", "")
            schema_name = track_name if track_name in (
                "presence", "activity", "fall", "people_count"
            ) else "custom"
            schema = LabelSchema(name=schema_name, label_dtype="float")

            track = LabelTrack(
                name=track_name,
                schema=schema,
                source="algorithm",
                data=df,
                description=f"Predictions from {algorithm}",
            )
            labels.add_track(track)

        reader.close()
        return labels

    def info(self) -> List[Dict[str, Any]]:
        """Human-readable summary of all cached results."""
        results = self._manifest.data.get("results", {})
        entries = []
        for short, meta in results.items():
            entries.append({
                "short_hash": short,
                "algorithm": meta.get("algorithm", ""),
                "description": meta.get("description", ""),
                "params": meta.get("params", {}),
                "upstreams": meta.get("upstreams", []),
                "keys": meta.get("keys", []),
                "n_frames": meta.get("n_frames", 0),
                "created": meta.get("created", ""),
                "last_accessed": meta.get("last_accessed", ""),
            })
        return entries


class _ManagedResultWriter(ResultWriter):
    """ResultWriter that updates the manifest on close."""

    def __init__(
        self,
        lmdb: LMDBStage,
        manifest: CacheManifest,
        algorithm: str,
        params: Dict[str, Any],
        description: str,
        upstreams: List[str],
        chain_hash_full: str,
        short_hash: str,
    ) -> None:
        super().__init__(lmdb)
        self._manifest = manifest
        self._algorithm = algorithm
        self._params = params
        self._description = description
        self._upstreams = upstreams
        self._chain_hash_full = chain_hash_full
        self._short_hash = short_hash

    def close(self) -> None:
        super().close()
        now = datetime.now(timezone.utc).isoformat()
        self._manifest.update_result(self._short_hash, {
            "algorithm": self._algorithm,
            "description": self._description,
            "params": self._params,
            "upstreams": self._upstreams,
            "chain_hash_full": self._chain_hash_full,
            "keys": self.keys,
            "n_frames": self.n_frames,
            "created": now,
            "last_accessed": now,
        })
        logger.debug(
            "Cached {} ({} frames, keys={}) → {}",
            self._algorithm, self.n_frames, self.keys, self._short_hash,
        )
