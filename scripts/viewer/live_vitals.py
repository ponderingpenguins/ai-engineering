"""Live people tracking, breathing rate and heart rate from a TI mmWave radar, computed on the PC.

    uv run python scripts/viewer/live_vitals.py

The board keeps its own firmware; nothing is flashed.

People: each frame's point cloud goes through the library's Python port of TI GTRACK
(vt_radar.mmwave.tracking), the same group tracker TI runs on-chip in its people-tracking demos.

Vital signs: the zero-Doppler range-azimuth heatmap (TLV 8 on AOP boards, TLV 4 otherwise) gives one
complex sample per range bin and virtual antenna. At the chest's range bin its phase moves by
4*pi/wavelength per metre of chest motion. The viewer picks the bin whose phase oscillates most in the
breathing band (near the tracked person if there is one), and takes the spectral peak over a sliding
window: breathing 6-36 /min, heart 48-120 /min (breathing harmonics excluded). Peaks on a band edge
are rejected as drift or leakage.

Sit still 0.3-2 m in front of the board, facing it. Rates appear after ~15 s.
Heart rate from a 5 fps stream is a rough estimate, not a medical measurement.
"""

import argparse
import threading
import time
import warnings
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from live_range_doppler import connect_radar, live_frames
from matplotlib.animation import FuncAnimation
from vt_radar.config.ti_config import ConfigFile
from vt_radar.mmwave.tracking import ekf_utils, gtrack_module, gtrack_test
from vt_radar.parsing_utils.parseFrame import parseStandardFrame

# The tracker port casts float64 max to float32 (inf) on purpose and divides by a point's Doppler,
# which is exactly 0 for static points; both are harmless and would flood the console.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"vt_radar\.mmwave\.tracking")

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "config/radar/vitals_oob.cfg"
WINDOW_S = 30.0  # sliding window for the rate estimates
MIN_S = 15.0  # data needed before a rate is shown
BREATH_BAND = (0.1, 0.6)  # Hz, 6-36 breaths/min
HEART_BAND = (0.8, 2.0)  # Hz, 48-120 beats/min
SEARCH_M = (0.25, 2.0)  # range searched for a chest when nobody is tracked
TRACK_GATE_M = 0.4  # search +- this around a tracked person's range
NFFT = 4096
MAX_TRACKS = 5
COLORS = plt.get_cmap("tab10").colors


def make_tracker(dt: float, max_vel: float, vel_res: float):
    cfg = ekf_utils.gtrack_moduleConfig()
    cfg.deltaT = dt
    cfg.maxRadialVelocity = max_vel
    cfg.radialVelocityResolution = vel_res
    cfg.maxAcceleration = 2.0
    cfg.maxNumPoints = 200
    cfg.maxNumTracks = MAX_TRACKS
    # The out-of-box CFAR gives only a few points per person, so allocate on 2 moving points.
    cfg.advParams.allocationParams = ekf_utils.gtrack_allocationParams(
        snrThre=30.0, velocityThre=0.1, pointsThre=2, maxDistanceThre=1.0, maxVelThre=2.0
    )
    # Keep a still (sitting) person for ~10 s without moving points.
    cfg.advParams.stateParams = ekf_utils.gtrack_stateParams(
        det2actThre=3, det2freeThre=3, active2freeThre=10, static2freeThre=50, exit2freeThre=5
    )
    return gtrack_test.create(cfg)


def band_peak(f, power, band, exclude=(), exclude_hz=0.05):
    """Frequency of the largest peak inside band, and its height over the band median.

    A maximum sitting on the band edge is a tail from outside the band (drift, or breathing leaking into
    the heart band), not a peak, so it scores 0.
    """
    inband = (f >= band[0]) & (f <= band[1])
    mask = inband.copy()
    for fx in exclude:
        mask &= np.abs(f - fx) > exclude_hz
    if not mask.any():
        return None, 0.0
    i = int(np.argmax(np.where(mask, power, -np.inf)))
    q = float(power[i] / (np.median(power[inband]) + 1e-12))
    if not (0 < i < len(power) - 1 and power[i] > power[i - 1] and power[i] > power[i + 1]) or not inband[i - 1] or not inband[i + 1]:
        q = 0.0
    return float(f[i]), q


def bandpass(sig, fps, band):
    spec = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1 / fps)
    spec[(f < band[0]) | (f > band[1])] = 0
    return np.fft.irfft(spec, len(sig))


def detrend(x, axis=0):
    t = np.arange(x.shape[axis], dtype=float)
    t -= t.mean()
    xm = x - x.mean(axis=axis, keepdims=True)
    shape = [1] * x.ndim
    shape[axis] = -1
    slope = (xm * t.reshape(shape)).sum(axis=axis, keepdims=True) / (t**2).sum()
    return xm - slope * t.reshape(shape)


def chest_phases(H, bins):
    """Unwrapped phase over time for each candidate bin, antennas combined by a matched filter."""
    X = H[:, bins, :]  # (T, C, A)
    ref = X.mean(axis=0)  # spatial signature of whatever reflects in each bin
    z = np.einsum("tca,ca->tc", X, ref.conj())
    return np.unwrap(np.angle(z), axis=0), np.abs(z).mean(axis=0)


def analyse(H, cands, fps, heart_band, wavelength_mm, prev_bin=None):
    """Pick the chest range bin among cands and estimate breathing and heart rate.

    H is (frames, range_bins, antennas) complex. Returns the chosen bin, chest displacement (mm) and the
    breathing/heart frequencies (Hz) with their peak scores (0 = no usable peak).
    """
    phase, mag = chest_phases(H, cands)
    phase = detrend(phase)
    f = np.fft.rfftfreq(NFFT, 1 / fps)
    spec = np.abs(np.fft.rfft(phase * np.hanning(len(phase))[:, None], NFFT, axis=0)) ** 2
    b = (f >= BREATH_BAND[0]) & (f <= BREATH_BAND[1])
    band_power = spec[b].sum(0)
    frac = band_power / (spec[f > 0.05].sum(0) + 1e-12)
    # A chest bin: a real reflector (not an empty bin whose phase is noise), whose phase motion is mostly
    # in the breathing band, with a true peak there (not drift piling up at the band edge).
    valid = (mag >= np.median(mag)) & (frac >= 0.5)
    for k in np.flatnonzero(valid):
        valid[k] = band_peak(f, spec[:, k], BREATH_BAND)[1] > 0
    # Among those, the chest moves millimetres; furniture picking up a little of it moves far less.
    score = np.where(valid, band_power, 0.0)
    best = int(np.argmax(score if valid.any() else frac))
    if prev_bin in cands:
        j = int(np.flatnonzero(cands == prev_bin)[0])
        if score[best] < 1.5 * score[j]:
            best = j  # hysteresis: do not hop bins on small score changes

    br_f, br_q = band_peak(f, spec[:, best], BREATH_BAND)
    # Breathing is not a pure sine; skip its harmonics that land in the heart band.
    # Orders above ~6 are negligible, and excluding them would blank the whole band for slow breathing.
    excl = [k * br_f for k in range(2, 7)] if br_f else []
    hr_f, hr_q = band_peak(f, spec[:, best], heart_band, exclude=excl, exclude_hz=0.03)
    return {
        "bin": int(cands[best]),
        "disp_mm": phase[:, best] * wavelength_mm / (4 * np.pi),
        "br_f": br_f, "br_q": br_q, "hr_f": hr_f, "hr_q": hr_q,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg-file", default=str(DEFAULT_CFG))
    parser.add_argument("--breath-mm", type=float, default=5.0, help="fixed +- y-range of the breathing plot (mm)")
    parser.add_argument("--heart-mm", type=float, default=2.0, help="fixed +- y-range of the heartbeat plot (mm)")
    args = parser.parse_args()

    cfg = ConfigFile()
    cfg.parseCfg(args.cfg_file)
    sensor = cfg.sensor()
    range_bins = np.asarray(sensor.range_bins)
    frame_cfg = next(line.split() for line in cfg.cfg if line.startswith("frameCfg"))
    fps = 1000 / float(frame_cfg[5])
    profile = next(line.split() for line in cfg.cfg if line.startswith("profileCfg"))
    wavelength_mm = 299.792458 / float(profile[2])  # c / start frequency (GHz)
    heart_band = (HEART_BAND[0], min(HEART_BAND[1], 0.9 * fps / 2))
    window = int(WINDOW_S * fps)

    tracker_args = (1 / fps, float(np.abs(sensor.doppler_bins).max()), float(sensor.doppler_resolution))
    tracker = make_tracker(*tracker_args)
    meas = [ekf_utils.gtrack_measurementPoint() for _ in range(200)]
    targets = [ekf_utils.gtrack_targetDesc() for _ in range(MAX_TRACKS)]

    uart = connect_radar(cfg)
    lock = threading.Lock()
    stop = threading.Event()
    state: dict = {
        "heatmaps": deque(maxlen=window),
        "points": deque(maxlen=3),
        "tracks": [],
        "n": 0,
        "t": deque(maxlen=20),
        "missing_heatmap": 0,
    }

    def reader():
        nonlocal tracker
        frames = live_frames(uart)
        for raw in frames:
            if stop.is_set():
                break
            if not raw:
                continue
            try:
                data = parseStandardFrame(raw, cfg)
            except Exception as e:  # a partial/garbled frame (e.g. right after reconnecting): skip it
                print(f"skipped corrupt frame: {type(e).__name__}: {e}", flush=True)
                continue
            pc = data.point_cloud
            n = data.num_detected_points if pc is not None else 0
            pts = pc[:n] if n else np.empty((0, 7))
            m = 0
            for x, y, _z, dop, snr, *_ in pts[: len(meas)]:
                r = float(np.hypot(x, y))
                if r < 0.2:
                    continue
                meas[m].range, meas[m].angle = r, float(np.arctan2(x, y))
                meas[m].doppler, meas[m].snr = float(dop), float(10 ** (snr / 10))
                m += 1
            t_num = [0]
            try:
                gtrack_module.step(tracker, meas, None, m, targets, t_num, 0)
            except (OverflowError, ValueError, ZeroDivisionError, np.linalg.LinAlgError) as e:
                # A numerically broken track must not kill the reader (and freeze the vitals); start over.
                print(f"tracker reset after {type(e).__name__}: {e}", flush=True)
                tracker = make_tracker(*tracker_args)
                t_num = [0]
            tracks = [(int(t.uid), *map(float, t.S[:4])) for t in targets[: t_num[0]]]
            tracks = [t for t in tracks if np.isfinite(t[1:]).all()]
            with lock:
                if data.azimuth_static_heatmap is not None:
                    state["heatmaps"].append(data.azimuth_static_heatmap)
                else:
                    state["missing_heatmap"] += 1
                state["points"].append(pts[:, :2].copy())
                state["tracks"] = tracks
                state["n"] += 1
                state["t"].append(time.monotonic())
        frames.close()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    ax_top = fig.add_subplot(grid[0, 0])
    ax_txt = fig.add_subplot(grid[0, 1])
    ax_br = fig.add_subplot(grid[1, 0])
    ax_hr = fig.add_subplot(grid[1, 1])

    ax_top.set_xlim(-3, 3)
    ax_top.set_ylim(0, 5)
    ax_top.set_aspect("equal")
    ax_top.grid(alpha=0.3)
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_title("people (GTRACK)")
    ax_top.plot(0, 0, "k^", ms=9)
    pts_scat = ax_top.scatter([], [], s=8, c="0.6")
    trk_scat = ax_top.scatter([], [], s=260, alpha=0.6)
    trk_labels = [ax_top.text(0, 0, "", fontsize=10, weight="bold", ha="center", va="center") for _ in range(MAX_TRACKS)]
    (chest_marker,) = ax_top.plot([], [], color="crimson", lw=6, alpha=0.4, solid_capstyle="round")

    ax_txt.axis("off")
    br_text = ax_txt.text(0.02, 0.78, "", fontsize=30, weight="bold", color="tab:blue", transform=ax_txt.transAxes)
    hr_text = ax_txt.text(0.02, 0.50, "", fontsize=30, weight="bold", color="crimson", transform=ax_txt.transAxes)
    info_text = ax_txt.text(0.02, 0.08, "", fontsize=11, family="monospace", transform=ax_txt.transAxes, va="bottom")

    t_axis = np.arange(window) / fps - WINDOW_S
    (br_line,) = ax_br.plot([], [], color="tab:blue")
    ax_br.set_xlim(-WINDOW_S, 0)
    ax_br.set_xlabel("time (s)")
    ax_br.set_ylabel("chest motion (mm)")
    ax_br.set_ylim(-args.breath_mm, args.breath_mm)  # fixed so the waveform is easy to read
    ax_br.set_title("breathing waveform")
    (hr_line,) = ax_hr.plot([], [], color="crimson")
    ax_hr.set_xlim(-WINDOW_S, 0)
    ax_hr.set_xlabel("time (s)")
    ax_hr.set_ylabel("chest motion, heart band (mm)")
    ax_hr.set_ylim(-args.heart_mm, args.heart_mm)
    ax_hr.set_title("heartbeat waveform")
    title = fig.suptitle("waiting for radar frames ...")

    est = {"bin": None, "br": deque(maxlen=10), "hr": deque(maxlen=10)}

    def update(_):
        with lock:
            H = np.array(state["heatmaps"]) if state["heatmaps"] else None
            pts = np.concatenate(list(state["points"])) if state["points"] else np.empty((0, 2))
            tracks = list(state["tracks"])
            n, ts, missing = state["n"], list(state["t"]), state["missing_heatmap"]

        rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0
        title.set_text(f"frame {n}  |  {rate:.1f} fps  |  {len(tracks)} people tracked")

        pts_scat.set_offsets(pts if len(pts) else np.empty((0, 2)))
        trk_scat.set_offsets(np.array([[t[1], t[2]] for t in tracks]) if tracks else np.empty((0, 2)))
        trk_scat.set_color([COLORS[t[0] % 10] for t in tracks])
        for i, lab in enumerate(trk_labels):
            if i < len(tracks):
                uid, x, y, vx, vy = tracks[i]
                lab.set_position((x, y))
                lab.set_text(f"P{uid}")
            else:
                lab.set_text("")

        if H is None or len(H) < 8:
            hint = "no heatmap TLV: check guiMonitor 4th field = 1" if missing > 10 else "collecting ..."
            br_text.set_text("Breathing  --")
            hr_text.set_text("Heart      --")
            info_text.set_text(hint)
            return

        # Restrict the chest search to the nearest tracked person, if any.
        lo, hi = SEARCH_M
        moving = False
        if tracks:
            uid, x, y, vx, vy = min(tracks, key=lambda t: np.hypot(t[1], t[2]))
            r = float(np.hypot(x, y))
            lo, hi = max(SEARCH_M[0], r - TRACK_GATE_M), r + TRACK_GATE_M
            moving = float(np.hypot(vx, vy)) > 0.3
        cands = np.flatnonzero((range_bins >= lo) & (range_bins <= hi))
        if len(cands) == 0:
            return

        a = analyse(H, cands, fps, heart_band, wavelength_mm, est["bin"])
        est["bin"] = a["bin"]
        chest_r = float(range_bins[a["bin"]])
        disp_mm, br_f, br_q, hr_f, hr_q = a["disp_mm"], a["br_f"], a["br_q"], a["hr_f"], a["hr_q"]
        seconds = len(disp_mm) / fps

        ready = seconds >= MIN_S and not moving
        if ready and br_f and br_q > 4:
            est["br"].append(br_f * 60)
        if ready and hr_f and hr_q > 3:
            est["hr"].append(hr_f * 60)
        if moving:
            est["br"].clear()
            est["hr"].clear()

        br_val = f"{np.median(est['br']):4.0f} /min" if est["br"] else "  --"
        hr_val = f"{np.median(est['hr']):4.0f} bpm" if est["hr"] else "  --"
        br_text.set_text(f"Breathing {br_val}")
        hr_text.set_text(f"Heart     {hr_val}")
        status = "person moving: hold still" if moving else (f"collecting {seconds:.0f}/{MIN_S:.0f} s" if seconds < MIN_S else "measuring")
        info_text.set_text(
            f"status      {status}\n"
            f"chest range {chest_r:.2f} m (bin {est['bin']})\n"
            f"breath peak {br_q:5.1f}x  (shown if > 4)\n"
            f"heart peak  {hr_q:5.1f}x  (shown if > 3)\n"
            f"window      {seconds:.0f} s at {fps:.0f} fps"
        )

        ang = np.linspace(-np.pi / 3, np.pi / 3, 30)
        chest_marker.set_data(chest_r * np.sin(ang), chest_r * np.cos(ang))

        breath = bandpass(disp_mm - disp_mm.mean(), fps, BREATH_BAND)
        heart = bandpass(disp_mm - disp_mm.mean(), fps, heart_band)
        tt = t_axis[-len(breath):]
        br_line.set_data(tt, breath)
        hr_line.set_data(tt, heart)

    anim = FuncAnimation(fig, update, interval=200, cache_frame_data=False)  # noqa: F841
    plt.tight_layout()
    try:
        plt.show()
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
