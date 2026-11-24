"""Feasibility spike 2: does a floor bounce leave a second point straight behind a real point?

The sensor looks down along y. A signal that bounces off the floor (y = h) after it hits a point P at height y
returns as a ghost G at about y_G = 2h - y_P, at the same x and z. So a ghost is a point in the same frame that
is (a) farther along y, (b) at nearly the same x and z, (c) weaker. Its floor estimate is h = (y_P + y_G) / 2.
A real floor must make h pile up at one value (about 1.8 m) more often inside a frame than across frames.

This relaxes spike 1, which dropped all pairs closer than 0.3 m (a ghost of a person lying on the floor is close).
Variant "doppler" also requires nearly equal Doppler.

Usage: python floor_ghost_spike.py POINTS.pkl OUT_DIR
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

LATERAL_M = 0.3  # x and z must agree within this; the radar's sideways resolution is coarse
MIN_DY_M = 0.1  # ghost must be at least this much farther along y (about 2 range bins)
DOPPLER_MS = 0.2  # "doppler" variant: Doppler must agree within this (one bin is 0.13 m/s)
H_EDGES = np.arange(0.8, 3.01, 0.05)
RNG = np.random.default_rng(0)


def pair_h(a: np.ndarray, b: np.ndarray, same: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Floor estimates h of the ghost-like pairs between clouds a and b, whether Doppler also agrees, and the pair count."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0), np.zeros(0, bool), 0
    ia, ib = np.meshgrid(np.arange(len(a)), np.arange(len(b)), indexing="ij")
    if same:
        keep = ia < ib
        ia, ib = ia[keep], ib[keep]
    else:
        ia, ib = ia.ravel(), ib.ravel()
    candidates = len(ia)
    p, g = a[ia], b[ib]
    swap = p[:, 1] > g[:, 1]  # make g the one farther along y
    p, g = np.where(swap[:, None], g, p), np.where(swap[:, None], p, g)
    ok = (
        (g[:, 1] - p[:, 1] >= MIN_DY_M)
        & (np.abs(g[:, 0] - p[:, 0]) <= LATERAL_M)
        & (np.abs(g[:, 2] - p[:, 2]) <= LATERAL_M)
        & (g[:, 4] < p[:, 4])
    )
    p, g = p[ok], g[ok]
    return (p[:, 1] + g[:, 1]) / 2, np.abs(g[:, 3] - p[:, 3]) <= DOPPLER_MS, candidates


def collect(clouds: list[np.ndarray]):
    out = {}
    for key in ("within", "control"):
        out[key] = {"h": [], "dop": [], "cand": 0}
    count = len(clouds)
    for i, cloud in enumerate(clouds):
        h, dop, n = pair_h(cloud, cloud, same=True)
        out["within"]["h"].append(h), out["within"]["dop"].append(dop)
        out["within"]["cand"] += n
        j = (i + int(RNG.integers(100, count - 100))) % count
        h, dop, n = pair_h(cloud, clouds[j], same=False)
        out["control"]["h"].append(h), out["control"]["dop"].append(dop)
        out["control"]["cand"] += n
    for key in out:
        out[key]["h"] = np.concatenate(out[key]["h"])
        out[key]["dop"] = np.concatenate(out[key]["dop"])
    return out


def sigma(within_h: np.ndarray, within_cand: int, control_h: np.ndarray, control_cand: int):
    """Histogram of h inside a frame, the expected histogram from the control (scaled by candidate pairs), sigma."""
    w = np.histogram(within_h, H_EDGES)[0].astype(float)
    c = np.histogram(control_h, H_EDGES)[0].astype(float)
    expected = c * within_cand / max(control_cand, 1)
    return w, expected, (w - expected) / np.sqrt(w + expected + 1)


def analyse(name: str, rec: dict, out: Path) -> dict:
    clouds = [c[:, :5] for c in rec["clouds"]]
    points = np.concatenate([c for c in clouds if len(c)])
    lying = np.concatenate([c for c, label in zip(clouds, rec["label"]) if len(c) and label.startswith("lying")] or [np.zeros((0, 5))])
    data = collect(clouds)
    centres = (H_EDGES[:-1] + H_EDGES[1:]) / 2
    summary = {
        "median_y_all_m": float(np.median(points[:, 1])),
        "median_y_lying_m": float(np.median(lying[:, 1])) if len(lying) else None,
        "candidate_pairs_within": data["within"]["cand"],
        "candidate_pairs_control": data["control"]["cand"],
    }
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for variant, mask in (("all pairs", False), ("Doppler agrees", True)):
        wh = data["within"]["h"][data["within"]["dop"]] if mask else data["within"]["h"]
        ch = data["control"]["h"][data["control"]["dop"]] if mask else data["control"]["h"]
        w, e, s = sigma(wh, data["within"]["cand"], ch, data["control"]["cand"])
        peak = int(np.argmax(s))
        key = "doppler" if mask else "all"
        summary[key] = {
            "pairs_within": int(len(wh)),
            "pairs_expected_from_control": float(e.sum()),
            "ratio_within_over_control": float(len(wh) / max(e.sum(), 1)),
            "peak_h_m": float(centres[peak]),
            "peak_sigma": float(s[peak]),
            "sigma_in_window_1p6_2p0": float(s[(centres > 1.6) & (centres < 2.0)].max()),
        }
        ax[0 if not mask else 1].plot(centres, w, label="inside a frame")
        ax[0 if not mask else 1].plot(centres, e, label="expected from control")
        ax[0 if not mask else 1].axvline(summary["median_y_all_m"], color="gray", ls=":", label="median y of all points")
        if summary["median_y_lying_m"]:
            ax[0 if not mask else 1].axvline(summary["median_y_lying_m"], color="k", ls="--", label="median y, lying frames")
        ax[0 if not mask else 1].set(title=f"Floor estimate h = (y_P + y_G)/2, {variant}", xlabel="h (m)", ylabel="pairs")
    ax[0].legend(fontsize=7)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out / f"floor_{name}.png", dpi=110)
    plt.close(fig)
    return summary


def main(points_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = pickle.load(open(points_path, "rb"))
    summaries = {name: analyse(name, rec, out) for name, rec in data.items()}
    (out / "floor_summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
