"""Unified sliding window for temporal pointcloud aggregation.

A single reusable ring buffer that accepts (N, C) arrays, tracks frame
boundaries, and provides both per-frame and stacked views.  Domain-agnostic:
no activity labels, no classification logic.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np


class PointCloudWindow:
    """Sliding window over pointcloud frames.

    Parameters
    ----------
    window_size:
        Maximum number of frames to retain.
    step:
        Emit ``ready`` every *step* pushes after the buffer is full.
    columns:
        If set, pad (with zeros) or truncate every incoming array to exactly
        this many columns.  ``None`` keeps the input width as-is.
    """

    def __init__(
        self,
        window_size: int,
        step: int = 1,
        columns: Optional[int] = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if step < 1:
            raise ValueError("step must be >= 1")
        if columns is not None and columns < 1:
            raise ValueError("columns must be >= 1 or None")

        self._window_size = window_size
        self._step = step
        self._columns = columns

        self._buf: Deque[np.ndarray] = deque(maxlen=window_size)
        self._frame_counter: int = 0

        # Lazy cache — invalidated on every push.
        self._stacked_cache: Optional[np.ndarray] = None
        self._boundaries_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def push(self, points: np.ndarray) -> bool:
        """Append a single frame's pointcloud.

        Parameters
        ----------
        points:
            ``(N, C)`` array.  ``None`` or non-2-D inputs are stored as an
            empty ``(0, C)`` array (where *C* = ``self.columns`` or 0).

        Returns
        -------
        ``True`` when the window is full **and** the frame counter aligns
        with *step*.
        """
        pc = self._normalise(points)
        self._buf.append(pc)
        self._frame_counter += 1

        # Invalidate caches.
        self._stacked_cache = None
        self._boundaries_cache = None

        return self.ready

    def frames(self) -> List[np.ndarray]:
        """Return a list copy of the per-frame arrays currently in the buffer."""
        return list(self._buf)

    def stacked(self) -> np.ndarray:
        """All buffered points concatenated into a single ``(M, C)`` array.

        Lazily computed and cached until the next :meth:`push`.
        """
        if self._stacked_cache is None:
            self._build_cache()
        return self._stacked_cache  # type: ignore[return-value]

    def boundaries(self) -> np.ndarray:
        """Frame start indices into :meth:`stacked`.

        ``boundaries()[i]`` is the first row of frame *i* in the stacked
        array.  To recover frame *i*::

            s, e = window.boundaries()[i], window.boundaries()[i + 1]
            frame_i = window.stacked()[s:e]

        The returned array has length ``len(frames) + 1`` (the last element
        equals ``stacked().shape[0]``).
        """
        if self._boundaries_cache is None:
            self._build_cache()
        return self._boundaries_cache  # type: ignore[return-value]

    @property
    def full(self) -> bool:
        """``True`` when the buffer contains *window_size* frames."""
        return len(self._buf) == self._window_size

    @property
    def ready(self) -> bool:
        """``True`` when full **and** the frame counter is aligned with *step*."""
        return self.full and (self._frame_counter % self._step == 0)

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def step(self) -> int:
        return self._step

    @property
    def columns(self) -> Optional[int]:
        return self._columns

    def reset(self) -> None:
        """Clear the buffer and all counters."""
        self._buf.clear()
        self._frame_counter = 0
        self._stacked_cache = None
        self._boundaries_cache = None

    def __len__(self) -> int:
        return len(self._buf)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalise(self, points: np.ndarray) -> np.ndarray:
        """Coerce *points* to a well-shaped ``(N, C)`` array."""
        cols = self._columns

        if points is None:
            return np.empty((0, cols or 0), dtype=np.float64)

        pc = np.asarray(points)
        if pc.ndim != 2:
            return np.empty((0, cols or 0), dtype=pc.dtype)

        if cols is None:
            return pc

        # Pad or truncate columns.
        if pc.shape[1] < cols:
            pad = np.zeros((pc.shape[0], cols - pc.shape[1]), dtype=pc.dtype)
            return np.concatenate([pc, pad], axis=1)
        if pc.shape[1] > cols:
            return pc[:, :cols]
        return pc

    def _build_cache(self) -> None:
        frames = list(self._buf)
        lengths = [f.shape[0] for f in frames]
        self._boundaries_cache = np.zeros(len(frames) + 1, dtype=np.intp)
        np.cumsum(lengths, out=self._boundaries_cache[1:])

        if sum(lengths) == 0:
            cols = self._columns or (frames[0].shape[1] if frames else 0)
            self._stacked_cache = np.empty((0, cols), dtype=np.float64)
        else:
            self._stacked_cache = np.vstack(frames)
