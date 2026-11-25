"""Feasibility spike: do reflection ghosts leave plane peaks in the radar points of a recording?

For two points P (real) and G (ghost) that mirror each other across a plane, the plane is the perpendicular
bisector of P-G. We vote for bisector planes of all point pairs inside one frame and look for peaks.
A pair is kept only if the ghost candidate G is farther from the sensor and has a lower SNR than P.
As a control, the same statistic is computed for pairs of points from two different frames far apart in time.
Ghosts exist only inside a frame, so real planes must give MORE votes inside a frame than across frames.

Sensor frame: x, y, z in metres, y along the boresight (the sensor looks down, y is the distance to the floor).
Plane normal n points from P to G. d is the distance from the sensor to the plane (always > 0).

Usage: python ghost_plane_spike.py POINTS.pkl OUT_DIR
POINTS.pkl is {recording: {"frame_idx", "label", "clouds": [(n, 7) arrays: x, y, z, doppler, snr, ...]}}.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MIN_SEPARATION_M = 0.3  # closer pairs give a noisy normal; the radar angle resolution is coarse
D_EDGES = np.arange(0.3, 4.86, 0.05)
PHI_EDGES = np.arange(-180, 181, 15)
FLOOR_COS = np.cos(np.deg2rad(20))  # floor normal is within 20 degrees of the boresight
WALL_SIN = np.sin(np.deg2rad(20))  # wall normal is within 20 degrees of the horizontal plane
RNG = np.random.default_rng(0)


def pair_planes(a: np.ndarray, b: np.ndarray, same: bool) -> np.ndarray:
    """Bisector planes of the pairs between clouds a and b, as rows (n_x, n_y, n_z, d).

    Keeps only pairs where the farther point has the lower SNR, and that are at least MIN_SEPARATION_M apart.
    With `same` (a is b) every unordered pair is used once.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((0, 4))
    ia, ib = np.meshgrid(np.arange(len(a)), np.arange(len(b)), indexing="ij")
    if same:
        keep = ia < ib
        ia, ib = ia[keep], ib[keep]
    else:
        ia, ib = ia.ravel(), ib.ravel()
    p, g = a[ia], b[ib]
    swap = np.linalg.norm(p[:, :3], axis=1) > np.linalg.norm(g[:, :3], axis=1)  # make g the farther one
    p, g = np.where(swap[:, None], g, p), np.where(swap[:, None], p, g)
    delta = g[:, :3] - p[:, :3]
    dist = np.linalg.norm(delta, axis=1)
    ok = (g[:, 4] < p[:, 4]) & (dist >= MIN_SEPARATION_M)
    delta, dist, p, g = delta[ok], dist[ok], p[ok], g[ok]
    n = delta / dist[:, None]
    d = np.einsum("ij,ij->i", n, (p[:, :3] + g[:, :3]) / 2)
    return np.column_stack([n, d])


def collect(clouds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Planes from pairs inside a frame, and from pairs between a frame and one far away in time (control)."""
    within, control = [], []
    count = len(clouds)
    for i, cloud in enumerate(clouds):
        within.append(pair_planes(cloud, cloud, same=True))
        j = (i + int(RNG.integers(100, count - 100))) % count
        control.append(pair_planes(cloud, clouds[j], same=False))
    return np.concatenate(within), np.concatenate(control)


def floor_hist(planes: np.ndarray) -> np.ndarray:
    sel = planes[:, 1] > FLOOR_COS
    return np.histogram(planes[sel, 3], D_EDGES)[0].astype(float)


def wall_hist(planes: np.ndarray) -> np.ndarray:
    sel = np.abs(planes[:, 1]) < WALL_SIN
    phi = np.degrees(np.arctan2(planes[sel, 2], planes[sel, 0]))
    return np.histogram2d(phi, planes[sel, 3], [PHI_EDGES, D_EDGES])[0]


def excess(within: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Within-frame minus control, with the control scaled to the same total, and the excess in sigma."""
    scale = within.sum() / max(control.sum(), 1)
    expected = control * scale
    return within - expected, (within - expected) / np.sqrt(within + expected + 1)


def analyse(name: str, rec: dict, out: Path) -> dict:
    clouds = [c[:, :5] for c in rec["clouds"]]
    points = np.concatenate([c for c in clouds if len(c)])
    lying = np.concatenate([c for c, label in zip(clouds, rec["label"]) if len(c) and label.startswith("lying")] or [np.zeros((0, 5))])
    within, control = collect(clouds)
    f_w, f_c = floor_hist(within), floor_hist(control)
    w_w, w_c = wall_hist(within), wall_hist(control)
    f_ex, f_sig = excess(f_w, f_c)
    w_ex, w_sig = excess(w_w, w_c)
    centres = (D_EDGES[:-1] + D_EDGES[1:]) / 2
    peak = int(np.argmax(f_sig))
    wi, wj = np.unravel_index(np.argmax(w_sig), w_sig.shape)
    summary = {
        "points": int(len(points)),
        "pairs_within": int(len(within)),
        "pairs_control": int(len(control)),
        "median_y_all_m": float(np.median(points[:, 1])),
        "median_y_lying_m": float(np.median(lying[:, 1])) if len(lying) else None,
        "y_p95_m": float(np.percentile(points[:, 1], 95)),
        "x_abs_p99_m": float(np.percentile(np.abs(points[:, 0]), 99)),
        "z_abs_p99_m": float(np.percentile(np.abs(points[:, 2]), 99)),
        "floor_peak_d_m": float(centres[peak]),
        "floor_peak_sigma": float(f_sig[peak]),
        "wall_peak_d_m": float(centres[wj]),
        "wall_peak_phi_deg": float((PHI_EDGES[wi] + PHI_EDGES[wi + 1]) / 2),
        "wall_peak_sigma": float(w_sig[wi, wj]),
        "n_cells_wall_over_5_sigma": int((w_sig > 5).sum()),
    }

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    ax[0].plot(centres, f_w / f_w.sum(), label="inside a frame")
    ax[0].plot(centres, f_c / f_c.sum(), label="control: frames far apart")
    ax[0].axvline(summary["median_y_all_m"], color="gray", ls=":", label="median y of all points")
    if summary["median_y_lying_m"]:
        ax[0].axvline(summary["median_y_lying_m"], color="k", ls="--", label="median y, lying frames")
    ax[0].set(title="Floor candidates: plane distance d (normal near the boresight)", xlabel="d (m)", ylabel="share of pairs")
    ax[0].legend(fontsize=7)
    ax[1].plot(centres, f_sig)
    ax[1].axhline(5, color="r", ls=":")
    ax[1].set(title="Floor: excess over control (sigma)", xlabel="d (m)", ylabel="sigma")
    im = ax[2].imshow(w_sig, origin="lower", aspect="auto", cmap="coolwarm", vmin=-8, vmax=8,
                      extent=[D_EDGES[0], D_EDGES[-1], PHI_EDGES[0], PHI_EDGES[-1]])
    ax[2].set(title="Wall candidates: excess (sigma), horizontal normals", xlabel="d (m)", ylabel="normal direction in x-z plane (deg)")
    fig.colorbar(im, ax=ax[2])
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=110)
    plt.close(fig)
    return summary


def main(points_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = pickle.load(open(points_path, "rb"))
    summaries = {name: analyse(name, rec, out) for name, rec in data.items()}
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
