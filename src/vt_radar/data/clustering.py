"""Clustering abstractions for radar point cloud data.

Provides ``DBSCANClusterer`` and ``HDBSCANClusterer`` behind a shared
``Clusterer`` ABC so consumers can swap algorithms without code changes.
Domain-specific post-processing stays with the consumer; this module only
owns the clustering step and structured result.

Example::

    from vt_radar.data.clustering import DBSCANClusterer, HDBSCANClusterer

    clusterer = DBSCANClusterer(eps=0.5, min_samples=10, cluster_columns=slice(0, 2))
    result = clusterer.fit(pointcloud)
    for c in result.clusters:
        print(f"label={c.label}  size={c.size}  centroid={c.centroid}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class Cluster:
    """A single cluster extracted from a clustering operation.

    Attributes:
        label: Cluster label assigned by DBSCAN.
        points: ``(N, C)`` — full rows from the input array for this cluster.
        indices: ``(N,)`` — row indices into the original input array.
    """

    label: int
    points: np.ndarray
    indices: np.ndarray

    @property
    def centroid(self) -> np.ndarray:
        """Mean position across all columns, shape ``(C,)``."""
        return self.points.mean(axis=0)

    @property
    def size(self) -> int:
        """Number of points in this cluster."""
        return len(self.points)

    def weighted_centroid(self, weights: np.ndarray) -> np.ndarray:
        """Weighted mean position, shape ``(C,)``.

        Args:
            weights: 1-D array of per-point weights, same length as
                ``self.points``.  If all weights are zero the unweighted
                :attr:`centroid` is returned as a fallback.
        """
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum()
        if total == 0:
            return self.centroid
        return (self.points * (w / total)[:, np.newaxis]).sum(axis=0)

    @property
    def spread(self) -> float:
        """Cluster spread: ``2 * norm(std)``.  Matches the compactness metric
        used by :class:`DBSCANPresence`."""
        return float(np.linalg.norm(self.points.std(axis=0)) * 2.0)


@dataclass(frozen=True)
class ClusteringResult:
    """Structured result from clustering an ``(N, C)`` array.

    Attributes:
        labels: ``(N,)`` per-point labels; ``-1`` means noise.
        source_points: ``(N, C)`` reference to the input array.
        clusters: Non-noise clusters sorted by size descending.
    """

    labels: np.ndarray
    source_points: np.ndarray
    clusters: tuple  # tuple[Cluster, ...]

    @property
    def noise_mask(self) -> np.ndarray:
        """Boolean mask, ``True`` where the point is noise."""
        return self.labels == -1

    @property
    def denoised(self) -> np.ndarray:
        """``(M, C)`` array of non-noise points."""
        return self.source_points[~self.noise_mask]

    @property
    def largest(self) -> Optional[Cluster]:
        """Biggest cluster by point count, or ``None`` if no clusters."""
        return self.clusters[0] if self.clusters else None

    @property
    def n_clusters(self) -> int:
        """Number of non-noise clusters."""
        return len(self.clusters)


class Clusterer(ABC):
    """Abstract base class for clustering algorithms.

    Subclasses implement :meth:`fit` to cluster an ``(N, C)`` array and
    return a :class:`ClusteringResult`.  This allows DBSCAN to be swapped
    for HDBSCAN, OPTICS, etc. without touching consumers.
    """

    @abstractmethod
    def fit(self, points: np.ndarray) -> ClusteringResult:
        """Cluster *points* and return a structured result."""
        ...


@dataclass
class DBSCANClusterer(Clusterer):
    """Configurable DBSCAN on arbitrary ``(N, C)`` arrays.

    Parameters:
        eps: DBSCAN neighbourhood radius.
        min_samples: Minimum points to form a core point.
        cluster_columns: Slice selecting which columns DBSCAN uses for
            distance computation.  ``None`` means all columns.
        min_points: If the input has fewer points than this, skip DBSCAN
            and use *fallback* instead.
        fallback: Strategy when ``len(points) < min_points``:

            - ``'empty'`` — return no clusters (all noise).
            - ``'individual'`` — treat each point as its own cluster.
    """

    eps: float = 0.5
    min_samples: int = 10
    cluster_columns: Optional[slice] = None
    min_points: int = 0
    fallback: str = "empty"

    def fit(self, points: np.ndarray) -> ClusteringResult:
        """Run DBSCAN and return a structured :class:`ClusteringResult`.

        Args:
            points: ``(N, C)`` input array.

        Returns:
            A :class:`ClusteringResult` with per-point labels and extracted
            :class:`Cluster` objects.
        """
        if points.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {points.shape}")

        n = len(points)

        if n == 0:
            return ClusteringResult(
                labels=np.empty(0, dtype=np.intp),
                source_points=points,
                clusters=(),
            )

        if n < self.min_points:
            return self._apply_fallback(points)

        features = points if self.cluster_columns is None else points[:, self.cluster_columns]
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(features)

        return self._build_result(points, labels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_fallback(self, points: np.ndarray) -> ClusteringResult:
        """Handle the case where we have fewer than *min_points*."""
        n = len(points)
        if self.fallback == "individual":
            labels = np.arange(n, dtype=np.intp)
            return self._build_result(points, labels)

        # 'empty' — everything is noise
        labels = np.full(n, -1, dtype=np.intp)
        return ClusteringResult(labels=labels, source_points=points, clusters=())

    @staticmethod
    def _build_result(points: np.ndarray, labels: np.ndarray) -> ClusteringResult:
        """Extract :class:`Cluster` objects from raw labels."""
        unique_labels = np.unique(labels)
        clusters = []
        for lbl in unique_labels:
            if lbl < 0:
                continue
            mask = labels == lbl
            clusters.append(Cluster(
                label=int(lbl),
                points=points[mask],
                indices=np.where(mask)[0],
            ))

        # Sort by size descending
        clusters.sort(key=lambda c: c.size, reverse=True)

        return ClusteringResult(
            labels=labels,
            source_points=points,
            clusters=tuple(clusters),
        )


@dataclass
class HDBSCANClusterer(Clusterer):
    """Configurable HDBSCAN on arbitrary ``(N, C)`` arrays.

    HDBSCAN adapts to varying point density automatically — no ``eps``
    parameter required.  Uses :func:`sklearn.cluster.HDBSCAN` (available
    since scikit-learn 1.3).

    Parameters:
        min_cluster_size: Smallest grouping to consider a cluster.
        min_samples: Core-point density threshold.  ``None`` defaults to
            *min_cluster_size* (sklearn behaviour).
        cluster_columns: Slice selecting which columns to use for distance
            computation.  ``None`` means all columns.
        min_points: If the input has fewer points than this, skip HDBSCAN
            and use *fallback* instead.
        fallback: Strategy when ``len(points) < min_points``:

            - ``'empty'`` — return no clusters (all noise).
            - ``'individual'`` — treat each point as its own cluster.
    """

    min_cluster_size: int = 10
    min_samples: Optional[int] = None
    cluster_columns: Optional[slice] = None
    min_points: int = 0
    fallback: str = "empty"

    def fit(self, points: np.ndarray) -> ClusteringResult:
        """Run HDBSCAN and return a structured :class:`ClusteringResult`."""
        if points.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {points.shape}")

        n = len(points)

        if n == 0:
            return ClusteringResult(
                labels=np.empty(0, dtype=np.intp),
                source_points=points,
                clusters=(),
            )

        if n < self.min_points:
            return self._apply_fallback(points)

        from sklearn.cluster import HDBSCAN as _HDBSCAN

        features = points if self.cluster_columns is None else points[:, self.cluster_columns]
        labels = _HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
        ).fit_predict(features)

        return DBSCANClusterer._build_result(points, labels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_fallback(self, points: np.ndarray) -> ClusteringResult:
        """Handle the case where we have fewer than *min_points*."""
        n = len(points)
        if self.fallback == "individual":
            labels = np.arange(n, dtype=np.intp)
            return DBSCANClusterer._build_result(points, labels)

        # 'empty' — everything is noise
        labels = np.full(n, -1, dtype=np.intp)
        return ClusteringResult(labels=labels, source_points=points, clusters=())
