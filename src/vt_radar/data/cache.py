"""Low-level LMDB cache I/O for preprocessed radar data.

Provides CacheManifest for invalidation tracking and LMDBStage for
reading/writing individual preprocessing stages.
"""

import hashlib
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import orjson

from loguru import logger


class CacheManifest:
    """Reads/writes manifest.json and checks invalidation keys."""

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir
        self._path = os.path.join(cache_dir, "manifest.json")
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.isfile(self._path):
            try:
                with open(self._path, "rb") as f:
                    return orjson.loads(f.read())
            except Exception:
                logger.warning("Corrupt manifest.json, rebuilding cache")
        return {"version": 2, "stages": {}, "results": {}}

    def _save(self) -> None:
        os.makedirs(self._cache_dir, exist_ok=True)
        with open(self._path, "wb") as f:
            f.write(orjson.dumps(self._data, option=orjson.OPT_INDENT_2))

    def is_stage_valid(self, stage_name: str, source_hash: str,
                       schema_version: int) -> bool:
        """Check if a cached stage is still valid."""
        stage = self._data.get("stages", {}).get(stage_name)
        if stage is None:
            return False
        if stage.get("source_hash") != source_hash:
            return False
        if stage.get("schema_version") != schema_version:
            return False
        # Check the LMDB directory actually exists
        stage_dir = os.path.join(self._cache_dir, stage_name)
        if not os.path.isdir(stage_dir):
            return False
        return True

    def update_stage(self, stage_name: str, metadata: Dict[str, Any]) -> None:
        """Record metadata for a completed stage build."""
        if "stages" not in self._data:
            self._data["stages"] = {}
        self._data["stages"][stage_name] = metadata
        self._save()

    def invalidate_stage(self, stage_name: str) -> None:
        """Remove a stage from the manifest and delete its LMDB directory."""
        stages = self._data.get("stages", {})
        if stage_name in stages:
            del stages[stage_name]
            self._save()
        stage_dir = os.path.join(self._cache_dir, stage_name)
        if os.path.isdir(stage_dir):
            shutil.rmtree(stage_dir, ignore_errors=True)

    # --- Result cache methods ---

    def update_result(self, short_hash: str, metadata: Dict[str, Any]) -> None:
        """Record metadata for a cached algorithm result."""
        if "results" not in self._data:
            self._data["results"] = {}
        self._data["results"][short_hash] = metadata
        self._save()

    def remove_result(self, short_hash: str) -> None:
        """Remove a result entry from the manifest."""
        results = self._data.get("results", {})
        if short_hash in results:
            del results[short_hash]
            self._save()

    def touch_accessed(self, short_hash: str) -> None:
        """Update last_accessed timestamp for an LRU result entry."""
        results = self._data.get("results", {})
        if short_hash in results:
            results[short_hash]["last_accessed"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._save()

    def gc_stale_results(self, results_dir: str, max_age_days: int = 30) -> int:
        """Remove result caches not accessed in max_age_days. Returns count removed."""
        results = self._data.get("results", {})
        now = datetime.now(timezone.utc)
        to_remove: List[str] = []
        for short_hash, meta in results.items():
            accessed = meta.get("last_accessed", "")
            if not accessed:
                to_remove.append(short_hash)
                continue
            try:
                ts = datetime.fromisoformat(accessed)
                age = (now - ts).days
                if age > max_age_days:
                    to_remove.append(short_hash)
            except (ValueError, TypeError):
                to_remove.append(short_hash)

        for short_hash in to_remove:
            lmdb_dir = os.path.join(results_dir, short_hash)
            if os.path.isdir(lmdb_dir):
                shutil.rmtree(lmdb_dir, ignore_errors=True)
            del results[short_hash]
            logger.debug("GC removed stale result cache: {}", short_hash)

        if to_remove:
            self._save()
        return len(to_remove)

    @property
    def data(self) -> Dict[str, Any]:
        return self._data


# Writers start at this size and double as needed. Kept small because on
# Windows LMDB reserves map_size as real (non-sparse) disk immediately — a
# large default silently consumes the whole drive across many caches.
_INITIAL_WRITE_MAP = 16 * 1024**2  # 16 MB
# Fallback for readers when no DB file exists yet.
_DEFAULT_READ_MAP = 16 * 1024**2


class LMDBStage:
    """Manages a single stage's LMDB environment.

    Sizing strategy (see test_lmdb_stage_sizing.py):
    - Readers adopt the on-disk ``data.mdb`` size, so they always map exactly
      what the writer produced — including legacy 10 GB caches.
    - Writers start at ``initial_map_size`` and grow via :meth:`maybe_grow`,
      so a fresh cache only reserves what its data needs. Rebuilding over an
      existing cache never shrinks below the current file size (LMDB forbids
      mapping smaller than live data).
    """

    def __init__(self, path: str, map_size: Optional[int] = None,
                 readonly: bool = True,
                 initial_map_size: int = _INITIAL_WRITE_MAP) -> None:
        self._path = path
        self._readonly = readonly
        if map_size is None:
            on_disk = self._on_disk_size()
            if readonly:
                map_size = on_disk or _DEFAULT_READ_MAP
            else:
                # Never map smaller than existing data, else lmdb.open fails.
                map_size = max(initial_map_size, on_disk)
        self._map_size = map_size
        self._env: Any = None

    def _on_disk_size(self) -> int:
        f = os.path.join(self._path, "data.mdb")
        try:
            return os.path.getsize(f)
        except OSError:
            return 0

    def open(self) -> Any:
        """Open the LMDB environment."""
        if self._env is None:
            import lmdb
            os.makedirs(self._path, exist_ok=True)
            self._env = lmdb.open(
                self._path,
                map_size=self._map_size,
                readonly=self._readonly,
                readahead=True,
                meminit=False,
                max_readers=126,
                lock=not self._readonly,
            )
        return self._env

    def maybe_grow(self, headroom_frac: float = 0.5) -> None:
        """Double the map size if usage exceeds ``1 - headroom_frac`` of it.

        Safe to call only between write transactions (no txn active). The
        builder loops call this at each commit boundary so a single in-flight
        500-frame batch never overflows.
        """
        if self._readonly or self._env is None:
            return
        info = self._env.info()
        stat = self._env.stat()
        used = (info["last_pgno"] + 1) * stat["psize"]
        if used > self._map_size * (1.0 - headroom_frac):
            new_size = self._map_size * 2
            self._env.set_mapsize(new_size)
            self._map_size = new_size

    def close(self) -> None:
        """Close the LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def get(self, key: str) -> Optional[bytes]:
        """Get a value by key."""
        env = self.open()
        with env.begin(write=False) as txn:
            val = txn.get(key.encode("utf-8"))
        return val

    def get_json(self, key: str) -> Optional[Any]:
        """Get and decode a JSON value by key."""
        raw = self.get(key)
        if raw is None:
            return None
        return orjson.loads(raw)

    def get_array(self, key: str, dtype: np.dtype,
                  shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """Get a numpy array by key, reshaping from raw bytes."""
        raw = self.get(key)
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def begin_write(self) -> Any:
        """Begin a write transaction."""
        env = self.open()
        return env.begin(write=True)

    def put(self, txn: Any, key: str, value: bytes) -> None:
        """Put a key-value pair in the given transaction."""
        txn.put(key.encode("utf-8"), value)

    def put_json(self, txn: Any, key: str, value: Any) -> None:
        """Put a JSON-serializable value."""
        txn.put(key.encode("utf-8"), orjson.dumps(value))

    def put_array(self, txn: Any, key: str, arr: np.ndarray) -> None:
        """Put a numpy array as raw bytes."""
        txn.put(key.encode("utf-8"), np.ascontiguousarray(arr).tobytes())

    @property
    def path(self) -> str:
        return self._path


def compute_source_hash(h5_files: List[str]) -> str:
    """SHA-256 of sorted (filename, size, mtime) tuples. Stat calls only."""
    entries = []
    for f in sorted(h5_files):
        st = os.stat(f)
        entries.append((os.path.basename(f), st.st_size, st.st_mtime))
    raw = orjson.dumps(entries)
    return hashlib.sha256(raw).hexdigest()
