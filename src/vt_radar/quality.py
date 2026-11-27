"""Energy functions for radar recording quality.

A penalty is 0 when there is no problem and grows to 1 as the problem gets worse. Heatmap values stay
in the firmware's raw units (log2 magnitude, Q9), so visibility is in units of the empty-room spread,
not dB. Thresholds and weights live in config/quality.yaml. Background: docs/energy_research_1.md.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
HEATMAP_CEILING = 65535
OCCUPANCY_STATES = ("unknown", "present", "empty")
HEATMAP_TERMS = ("visibility", "focus")
POINT_TERMS = ("point_count", "empty_frames", "point_snr", "point_cap")


# Policy and context


@dataclass(frozen=True)
class Policy:
    policy_id: str
    profile: str
    window_len: int
    max_points: int
    roi_range_m: tuple[float, float]
    reference_frames: int
    visibility_top_k: int
    z_change: float
    spread_floor: float
    min_excess: float
    delta_quantile: float
    thresholds: dict[str, float]
    weights: dict[str, dict[str, float]]  # keys: heatmap, pointcloud, combined


def load_policy(profile: str | None = None, root: Path = REPO) -> Policy:
    """Read config/quality.yaml, plus window_len and max_points from config/data.yaml."""
    q = OmegaConf.to_container(OmegaConf.load(root / "config/quality.yaml"))
    windowing = OmegaConf.to_container(OmegaConf.load(root / "config/data.yaml"))["windowing"]
    profile = profile or q["default_profile"]
    if profile not in q["profiles"]:
        raise ValueError(f"unknown profile {profile!r}, choose from {sorted(q['profiles'])}")
    not_positive = [k for k, v in q["thresholds"].items() if v <= 0]
    if not_positive:
        raise ValueError(f"thresholds must be positive: {not_positive}")
    weights = q["profiles"][profile]
    unknown = set(weights["heatmap"]) - set(HEATMAP_TERMS) | set(weights["pointcloud"]) - set(POINT_TERMS)
    if unknown:
        raise ValueError(f"unknown penalty terms in profile {profile}: {sorted(unknown)}")
    lo, hi = q["roi_range_m"]
    return Policy(
        policy_id=q["policy_id"], profile=profile,
        window_len=int(windowing["window_len"]), max_points=int(windowing["max_points"]),
        roi_range_m=(float(lo), float(hi)), reference_frames=int(q["reference_frames"]),
        visibility_top_k=int(q["visibility_top_k"]), z_change=float(q["z_change"]),
        spread_floor=float(q["spread_floor"]), min_excess=float(q["min_excess"]),
        delta_quantile=float(q["delta_quantile"]),
        thresholds={k: float(v) for k, v in q["thresholds"].items()}, weights=weights,
    )


def config_hash(cfg_lines: Sequence[str]) -> str:
    """Hash of the cfg commands, ignoring comments, blank lines and whitespace."""
    lines = [" ".join(line.split()) for line in cfg_lines]
    text = "\n".join(line for line in lines if line and not line.startswith("%"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class QualityContext:
    config_hash: str
    range_m: np.ndarray  # (R,) distance of each range bin
    num_doppler_bins: int
    roi_range_m: tuple[float, float]
    range_roi: np.ndarray  # (R,) bool; the heatmap ROI is these rows over all Doppler bins

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.range_m), self.num_doppler_bins


def build_context(cfg_lines: Sequence[str], range_m: np.ndarray, num_doppler_bins: int,
                  roi_range_m: tuple[float, float]) -> QualityContext:
    range_m = np.asarray(range_m, dtype=float)
    lo, hi = roi_range_m
    range_roi = (range_m >= lo) & (range_m <= hi)
    if not range_roi.any():
        raise ValueError(f"ROI {roi_range_m} m contains no range bins")
    return QualityContext(config_hash(cfg_lines), range_m, int(num_doppler_bins), (lo, hi), range_roi)


def context_from_config(cfg, policy: Policy) -> QualityContext:
    """Context from a parsed vt_radar ConfigFile."""
    sensor = cfg.sensor()
    return build_context(cfg.cfg, sensor.range_bins, len(sensor.doppler_bins), policy.roi_range_m)


# Empty-room reference


@dataclass
class EmptyReference:
    config_hash: str
    background: np.ndarray  # (R, D) per-cell median of the empty frames
    spread: np.ndarray  # (R, D) 1.4826 * MAD, at least the spread floor
    delta_range: float  # neighbour-difference thresholds for the gradient descriptor
    delta_doppler: float
    n_frames: int

    def standardise(self, heatmap: np.ndarray) -> np.ndarray:
        return (np.asarray(heatmap, dtype=np.float32) - self.background) / self.spread

    def save(self, path: str | Path) -> None:
        np.savez(path, config_hash=self.config_hash, background=self.background, spread=self.spread,
                 delta_range=self.delta_range, delta_doppler=self.delta_doppler, n_frames=self.n_frames)

    @classmethod
    def load(cls, path: str | Path, context: QualityContext) -> EmptyReference:
        """Load a saved reference; refuse it if it was made under a different radar configuration."""
        with np.load(path) as f:
            ref = cls(str(f["config_hash"]), f["background"], f["spread"], float(f["delta_range"]),
                      float(f["delta_doppler"]), int(f["n_frames"]))
        if ref.config_hash != context.config_hash:
            raise ValueError(f"reference {path} is for config {ref.config_hash}, not {context.config_hash}")
        if ref.background.shape != context.shape:
            raise ValueError(f"reference {path} has shape {ref.background.shape}, not {context.shape}")
        return ref


def fit_empty_reference(heatmaps: Sequence[np.ndarray] | np.ndarray, context: QualityContext,
                        policy: Policy) -> EmptyReference:
    """Fit background and spread from frames known to be empty.

    The gradient thresholds come from the same frames, so they are slightly low compared with new
    empty frames.
    """
    stack = np.asarray(heatmaps, dtype=np.float32)  # float before subtraction: uint16 would wrap
    if stack.ndim != 3 or stack.shape[1:] != context.shape:
        raise ValueError(f"expected (T, {context.shape[0]}, {context.shape[1]}) heatmaps, got {stack.shape}")
    if len(stack) < 2:
        raise ValueError("need at least 2 empty frames")
    background = np.median(stack, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(stack - background), axis=0), policy.spread_floor)
    z = ((stack - background) / spread)[:, context.range_roi]
    delta_range = float(np.quantile(np.abs(np.diff(z, axis=1)), policy.delta_quantile))
    delta_doppler = float(np.quantile(np.abs(np.diff(z, axis=2)), policy.delta_quantile))
    return EmptyReference(context.config_hash, background, spread, delta_range, delta_doppler, len(stack))


# Penalties


def shortfall(x: float, minimum: float) -> float:
    """0 at or above the minimum, 1 at zero, quadratic in between."""
    return max(0.0, (minimum - x) / minimum) ** 2


def excess(fraction: float, tolerated: float) -> float:
    """0 at or below the tolerated fraction, 1 at twice that, quadratic in between."""
    return min(1.0, max(0.0, (fraction - tolerated) / tolerated)) ** 2


def score(energy: float) -> float:
    """Display score: 100 at zero energy, about 37 at energy 1. Not a probability."""
    return 100.0 * math.exp(-energy)


# Heatmap descriptors


def normalised_entropy(values: np.ndarray, min_total: float) -> float | None:
    """Entropy across cells / ln(M): 0 when one cell holds all excess, 1 when spread evenly."""
    a = np.ravel(values).astype(np.float64)
    total = a.sum()
    if a.size < 2 or total < min_total:
        return None
    p = a[a > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(a.size))


def gini(values: np.ndarray, min_total: float) -> float | None:
    """0 for equal values, 1 - 1/M for a single non-zero cell."""
    a = np.sort(np.ravel(values).astype(np.float64))
    total = a.sum()
    if a.size < 2 or total < min_total:
        return None
    m = a.size
    i = np.arange(1, m + 1)
    return float(1 - 2 / m * ((a / total) * (m - i + 0.5)).sum())


def contrast(values: np.ndarray, min_total: float) -> float | None:
    """Standard deviation / mean."""
    a = np.ravel(values).astype(np.float64)
    if a.size < 2 or a.sum() < min_total:
        return None
    return float(a.std() / a.mean())


def gradient_detail(z: np.ndarray, delta_range: float, delta_doppler: float) -> float | None:
    """Mean log1p of neighbour differences above the empty-room thresholds, averaged over both axes.

    z is the standardised ROI map (range rows, Doppler columns). Doppler does not wrap around.
    """
    if z.shape[0] < 2 or z.shape[1] < 2:
        return None
    range_detail = np.log1p(np.maximum(np.abs(np.diff(z, axis=0)) - delta_range, 0)).mean()
    doppler_detail = np.log1p(np.maximum(np.abs(np.diff(z, axis=1)) - delta_doppler, 0)).mean()
    return float((range_detail + doppler_detail) / 2)


# Frames


@dataclass
class FrameMetrics:
    frame_num: int
    occupancy: str
    visibility: float | None = None
    changed_fraction: float | None = None
    entropy: float | None = None
    gini: float | None = None
    contrast: float | None = None
    gradient_detail: float | None = None
    ceiling_fraction: float = 0.0
    n_all: int = 0
    n_roi: int = 0
    roi_snr_db: np.ndarray | None = None  # None when the frame has no SNR values
    points_ok: bool = True  # False when the header declares more points than were decoded


def heatmap_problem(heatmap: np.ndarray | None, context: QualityContext) -> str | None:
    """Reason the heatmap cannot be used, or None."""
    if heatmap is None:
        return "heatmap_missing"
    if np.shape(heatmap) != context.shape:
        return f"wrong_heatmap_shape {np.shape(heatmap)}"
    if not np.all(np.isfinite(heatmap)):
        return "heatmap_not_finite"
    return None


def measure_frame(heatmap: np.ndarray, points: np.ndarray, context: QualityContext,
                  reference: EmptyReference | None, policy: Policy, frame_num: int, occupancy: str,
                  declared_points: int | None = None, snr_available: bool = True) -> FrameMetrics:
    """Measurements of one valid frame. points is (N, >=5): x, y, z, Doppler, SNR (dB)."""
    raw = np.asarray(heatmap, dtype=np.float32)
    m = FrameMetrics(frame_num, occupancy, ceiling_fraction=float(np.mean(raw >= HEATMAP_CEILING)))

    if reference is not None:
        roi = raw[context.range_roi]
        z = (roi - reference.background[context.range_roi]) / reference.spread[context.range_roi]
        z_pos = np.maximum(z, 0).ravel()
        k = min(policy.visibility_top_k, z_pos.size)
        m.visibility = float(np.partition(z_pos, -k)[-k:].mean())
        m.changed_fraction = float(np.mean(np.abs(z) > policy.z_change))
        raw_excess = np.maximum(roi - reference.background[context.range_roi], 0)
        m.entropy = normalised_entropy(raw_excess, policy.min_excess)
        m.gini = gini(raw_excess, policy.min_excess)
        m.contrast = contrast(raw_excess, policy.min_excess)
        m.gradient_detail = gradient_detail(z, reference.delta_range, reference.delta_doppler)

    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = points.reshape(0, 7)
    dist = np.linalg.norm(points[:, :3], axis=1)
    lo, hi = context.roi_range_m
    in_roi = (dist >= lo) & (dist <= hi)
    m.n_all = len(points)
    m.n_roi = int(in_roi.sum())
    m.roi_snr_db = points[in_roi, 4] if snr_available else None
    m.points_ok = declared_points is None or declared_points == len(points)
    return m


def points_from_frame(data) -> tuple[np.ndarray, int, bool]:
    """Decoded points, the header's point count, and whether SNR was sent, from parseStandardFrame output.

    The parser allocates one zero row per declared point and fills SNR only from the side-info TLV,
    so all-zero SNR means the side info was missing.
    """
    cloud = data.point_cloud if data.point_cloud is not None else np.zeros((0, 7))
    n = int(data.num_detected_points)
    points = cloud[:n]
    snr_available = n == 0 or bool(np.any(points[:, 4] > 0))
    return points, len(cloud), snr_available


# Windows


@dataclass
class WindowQuality:
    status: str  # "warming_up" or "ready"
    frames: int
    occupancy: str  # present, empty, unknown or mixed
    policy_id: str
    profile: str
    measurements: dict[str, float | None] = field(default_factory=dict)
    penalties: dict[str, float | None] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)  # why a penalty or score is None
    heatmap_score: float | None = None
    pointcloud_score: float | None = None
    combined_score: float | None = None
    warnings: list[str] = field(default_factory=list)


def _median(values: list[float | None]) -> float | None:
    """Median over frames; None if any frame lacks the value."""
    if not values or any(v is None for v in values):
        return None
    return float(np.median(values))


def _median_available(values: list[float | None]) -> float | None:
    """Median over the frames that have the value."""
    present = [v for v in values if v is not None]
    return float(np.median(present)) if present else None


def window_measurements(frames: Sequence[FrameMetrics], policy: Policy) -> dict[str, float | None]:
    n = len(frames)
    vis = [f.visibility for f in frames]
    n_all = sum(f.n_all for f in frames)
    n_roi = sum(f.n_roi for f in frames)
    snr_parts = [f.roi_snr_db for f in frames if f.n_roi]
    snr_known = all(s is not None for s in snr_parts)
    return {
        "visibility_median": _median(vis),
        "visibility_p10": None if _median(vis) is None else float(np.percentile(vis, 10)),
        "changed_fraction_median": _median([f.changed_fraction for f in frames]),
        "entropy_median": _median_available([f.entropy for f in frames]),
        "gini_median": _median_available([f.gini for f in frames]),
        "contrast_median": _median_available([f.contrast for f in frames]),
        "gradient_detail": _median([f.gradient_detail for f in frames]),
        "ceiling_fraction_max": max(f.ceiling_fraction for f in frames),
        "all_points": n_all,
        "roi_points": n_roi,
        "empty_roi_frames": sum(f.n_roi == 0 for f in frames),
        "empty_roi_frame_fraction": sum(f.n_roi == 0 for f in frames) / n,
        "median_point_snr_db": float(np.median(np.concatenate(snr_parts))) if snr_parts and snr_known else None,
        "snr_known": snr_known,
        "point_overflow_fraction": max(n_all - policy.max_points, 0) / n_all if n_all else 0.0,
        "points_ok": all(f.points_ok for f in frames),
    }


def window_penalties(m: dict, policy: Policy) -> tuple[dict[str, float | None], dict[str, str]]:
    """Penalties from window measurements, and the reason for each one that is None."""
    t = policy.thresholds
    p: dict[str, float | None] = {}
    reasons: dict[str, str] = {}

    p["visibility"] = None if m["visibility_median"] is None else shortfall(m["visibility_median"], t["visibility_min"])
    p["focus"] = None if m["gradient_detail"] is None else shortfall(m["gradient_detail"], t["gradient_min"])
    for name in HEATMAP_TERMS:
        if p[name] is None:
            reasons[name] = "reference_missing"

    if not m["points_ok"]:
        for name in POINT_TERMS:
            p[name] = None
            reasons[name] = "point_output_missing"
        return p, reasons
    p["point_count"] = shortfall(m["roi_points"], t["roi_points_min"])
    p["empty_frames"] = excess(m["empty_roi_frame_fraction"], t["empty_frame_fraction_max"])
    if m["roi_points"] == 0:
        p["point_snr"] = 1.0
    elif m["median_point_snr_db"] is None:
        p["point_snr"] = None
        reasons["point_snr"] = "snr_missing"
    else:
        p["point_snr"] = shortfall(max(m["median_point_snr_db"], 0.0), t["snr_min_db"])
    p["point_cap"] = m["point_overflow_fraction"]
    return p, reasons


def combine(penalties: dict[str, float | None], weights: dict[str, float]) -> tuple[float | None, str | None]:
    """Weighted sum; None and the first missing term if a weighted term is missing (no renormalising)."""
    for name in weights:
        if penalties.get(name) is None:
            return None, name
    return sum(w * penalties[name] for name, w in weights.items()), None


def score_window(frames: Sequence[FrameMetrics], policy: Policy) -> WindowQuality:
    states = {f.occupancy for f in frames}
    occupancy = states.pop() if len(states) == 1 else ("mixed" if states else "unknown")
    q = WindowQuality("warming_up", len(frames), occupancy, policy.policy_id, policy.profile)
    if len(frames) < policy.window_len:
        return q
    q.status = "ready"
    m = q.measurements = window_measurements(frames, policy)
    q.penalties, q.reasons = window_penalties(m, policy)
    v_min = policy.thresholds["visibility_min"]
    weak = m["visibility_median"] is not None and m["visibility_median"] < v_min
    if m["ceiling_fraction_max"] > 0:
        q.warnings.append("heatmap cells at the uint16 ceiling")
    if not m["points_ok"]:
        q.warnings.append("header declares more points than were decoded")

    if occupancy != "present":
        for name in ("heatmap", "pointcloud", "combined"):
            q.reasons[name] = f"occupancy_{occupancy}"
        if occupancy == "unknown" and weak:
            q.warnings.append("weak change from reference; occupancy unknown")
        if occupancy == "empty" and m["roi_points"]:
            q.warnings.append(f"empty scene has {m['roi_points'] / len(frames):.1f} ROI points per frame")
        return q

    if weak:
        q.warnings.append("weak change from reference")
    if m["roi_points"] == 0:
        q.warnings.append("no ROI detections")
    elif m["empty_roi_frames"]:
        q.warnings.append(f"point cloud empty in {m['empty_roi_frames']} of {len(frames)} occupied frames")

    energies: dict[str, float | None] = {}
    for name in ("heatmap", "pointcloud"):
        energies[name], missing = combine(q.penalties, policy.weights[name])
        if missing:
            q.reasons[name] = q.reasons.get(missing, f"{missing}_missing")
    energies["combined"], missing = combine(energies, policy.weights["combined"])
    if missing:
        q.reasons["combined"] = f"{missing}_score_missing"
    q.heatmap_score, q.pointcloud_score, q.combined_score = (
        None if energies[k] is None else score(energies[k]) for k in ("heatmap", "pointcloud", "combined"))
    return q


# Live buffer


class QualityMonitor:
    """Rolling quality window over live frames. A frame gap or an invalid frame restarts the window."""

    def __init__(self, context: QualityContext, policy: Policy, reference: EmptyReference | None = None):
        self.context = context
        self.policy = policy
        self.reference = reference
        self.frames: deque[FrameMetrics] = deque(maxlen=policy.window_len)
        self.notice = ""  # last restart reason, for display
        self._last_frame_num: int | None = None
        self._collecting: list[np.ndarray] | None = None

    @property
    def reference_progress(self) -> int | None:
        """Frames collected so far for a new reference, or None when not collecting."""
        return None if self._collecting is None else len(self._collecting)

    def start_reference(self) -> None:
        """Collect the next reference_frames valid frames as the empty-room reference."""
        self._collecting = []

    def restart(self, reason: str) -> None:
        self.frames.clear()
        self._last_frame_num = None
        self.notice = reason

    def push(self, frame_num: int, heatmap: np.ndarray | None, points: np.ndarray, occupancy: str,
             declared_points: int | None = None, snr_available: bool = True) -> WindowQuality:
        problem = heatmap_problem(heatmap, self.context)
        if problem:
            self.restart(f"invalid frame {frame_num}: {problem}")
            return score_window(list(self.frames), self.policy)
        if self._last_frame_num is not None and frame_num != self._last_frame_num + 1:
            self.restart(f"frame gap after {self._last_frame_num}: starting a new quality window")
        self._last_frame_num = frame_num

        if self._collecting is not None:
            self._collecting.append(np.asarray(heatmap, dtype=np.float32))
            if len(self._collecting) >= self.policy.reference_frames:
                self.reference = fit_empty_reference(self._collecting, self.context, self.policy)
                self._collecting = None
                self.restart("new empty-room reference: starting a new quality window")

        self.frames.append(measure_frame(heatmap, points, self.context, self.reference, self.policy,
                                         frame_num, occupancy, declared_points, snr_available))
        return score_window(list(self.frames), self.policy)
