"""Live recording-quality energy functions from a TI mmWave radar connected over USB.

    uv run python scripts/viewer/live_energy.py --reference data/quality/room1_reference.npz
    uv run python scripts/viewer/live_energy.py --replay data/raw/floor_posture_dataset/data/room1/<session>

Keys:
    b   collect a new empty-room reference (leave the room first); saved to --reference if given
    o   occupancy: unknown -> present -> empty -> unknown

Scores appear only when occupancy is "present" for the whole 16-frame window. Heatmap measurements
need an empty-room reference. Thresholds and weights: config/quality.yaml. See src/vt_radar/quality.py.
"""

import argparse
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from live_range_doppler import DEFAULT_CFG, connect_radar, live_frames, replay_frames, to_db
from matplotlib.animation import FuncAnimation
from vt_radar.config.ti_config import ConfigFile
from vt_radar.parsing_utils.parseFrame import parseStandardFrame
from vt_radar.quality import (
    HEATMAP_TERMS,
    OCCUPANCY_STATES,
    POINT_TERMS,
    EmptyReference,
    QualityMonitor,
    context_from_config,
    load_policy,
    points_from_frame,
)

HISTORY_S = 60.0
TERMS = HEATMAP_TERMS + POINT_TERMS


def fmt(value, digits=0):
    return "n/a" if value is None else f"{value:.{digits}f}"


def status_lines(w, monitor, occupancy, reference_frames):
    lines = [f"occupancy: {occupancy}   (o to change)   profile: {w.profile}"]
    progress = monitor.reference_progress
    if progress is not None:
        lines.append(f"reference: collecting {progress}/{reference_frames}, keep the room empty")
    elif monitor.reference is None:
        lines.append("reference: missing (b to collect); heatmap measurements unavailable")
    else:
        lines.append(f"reference: ready ({monitor.reference.n_frames} frames)")
    if w.status == "warming_up":
        lines.append(f"warming up: {w.frames}/{monitor.policy.window_len} consecutive frames")
    else:
        m = w.measurements
        lines.append(f"heatmap {fmt(w.heatmap_score)} | points {fmt(w.pointcloud_score)} | "
                     f"combined {fmt(w.combined_score)}")
        lines.append(f"visibility {fmt(m['visibility_median'], 1)} (p10 {fmt(m['visibility_p10'], 1)}) | "
                     f"ROI points {m['roi_points']} | ROI empty {m['empty_roi_frames']}/{w.frames} | "
                     f"SNR {fmt(m['median_point_snr_db'], 1)} dB")
        lines.append(f"entropy {fmt(m['entropy_median'], 2)} | gini {fmt(m['gini_median'], 2)} | "
                     f"contrast {fmt(m['contrast_median'], 2)} | gradient {fmt(m['gradient_detail'], 2)}")
        for name in ("heatmap", "pointcloud"):
            if name in w.reasons:
                lines.append(f"no {name} score: {w.reasons[name]}")
    lines.extend(f"warning: {text}" for text in w.warnings)
    if monitor.notice:
        lines.append(monitor.notice)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg-file", default=str(DEFAULT_CFG))
    parser.add_argument("--replay", help="recording directory to play back instead of the live radar")
    parser.add_argument("--reference", type=Path, help="empty-room reference .npz to load, and to save a new one to")
    parser.add_argument("--profile", help="scoring profile from config/quality.yaml (default: its default_profile)")
    parser.add_argument("--occupancy", choices=OCCUPANCY_STATES, default="unknown")
    parser.add_argument("--roi-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="ROI range in metres")
    args = parser.parse_args()

    cfg_path = args.cfg_file
    if args.replay:
        cfgs = sorted(Path(args.replay).glob("*.cfg"))
        cfg_path = str(cfgs[0]) if cfgs else cfg_path
    cfg = ConfigFile()
    cfg.parseCfg(cfg_path)
    policy = load_policy(args.profile)
    if args.roi_range:
        from dataclasses import replace

        policy = replace(policy, roi_range_m=tuple(args.roi_range))
    context = context_from_config(cfg, policy)
    sensor = cfg.sensor()
    range_bins, doppler_bins = sensor.range_bins, sensor.doppler_bins
    frame_cfg = next(line.split() for line in cfg.cfg if line.startswith("frameCfg"))
    fps = 1000 / float(frame_cfg[5])

    reference = None
    if args.reference and args.reference.exists():
        try:
            reference = EmptyReference.load(args.reference, context)
            print(f"loaded reference {args.reference}")
        except ValueError as e:
            print(f"reference not used: {e}")
    monitor = QualityMonitor(context, policy, reference)

    uart = None if args.replay else connect_radar(cfg)

    lock = threading.Lock()
    stop = threading.Event()
    state: dict = {"occupancy": args.occupancy, "window": None, "heatmap": None,
                   "history": deque(), "saved": reference}

    def reader():
        frames = replay_frames(args.replay, cfg, fps) if args.replay else live_frames(uart)
        for raw in frames:
            if stop.is_set():
                break
            if not raw:
                continue
            try:
                data = parseStandardFrame(raw, cfg)
            except Exception as e:  # a partial/garbled frame (e.g. right after reconnecting)
                with lock:
                    monitor.restart(f"corrupt frame skipped ({type(e).__name__}): starting a new quality window")
                continue
            points, declared, snr_available = points_from_frame(data)
            with lock:
                window = monitor.push(data.frame_num, data.range_doppler_heatmap, points, state["occupancy"],
                                      declared, snr_available)
                state["window"] = window
                state["heatmap"] = data.range_doppler_heatmap
                now = time.monotonic()
                state["history"].append((now, window.heatmap_score, window.pointcloud_score, window.combined_score))
                while state["history"] and now - state["history"][0][0] > HISTORY_S:
                    state["history"].popleft()
                new_reference = monitor.reference
            if new_reference is not None and new_reference is not state["saved"]:
                state["saved"] = new_reference
                if args.reference:
                    args.reference.parent.mkdir(parents=True, exist_ok=True)
                    new_reference.save(args.reference)
                    print(f"saved reference {args.reference}")
        frames.close()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    plt.rcParams["keymap.zoom"] = [k for k in plt.rcParams["keymap.zoom"] if k != "o"]
    plt.rcParams["keymap.back"] = [k for k in plt.rcParams["keymap.back"] if k != "b"]
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1.4, 1])
    ax_map = fig.add_subplot(grid[0, 0])
    ax_bars = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, 0])
    ax_hist = fig.add_subplot(grid[1, 1])

    extent = (doppler_bins[0], doppler_bins[-1], range_bins[0], range_bins[-1])
    image = ax_map.imshow(np.zeros(context.shape), origin="lower", aspect="auto", extent=extent, cmap="magma")
    colorbar = fig.colorbar(image, ax=ax_map)
    for r in policy.roi_range_m:
        ax_map.axhline(r, color="cyan", lw=0.8, ls="--")
    ax_map.set_xlabel("Doppler (m/s)")
    ax_map.set_ylabel("Range (m)")

    y = np.arange(len(TERMS))
    bars = ax_bars.barh(y, np.zeros(len(TERMS)), color="tab:gray")
    ax_bars.set_yticks(y, TERMS)
    ax_bars.invert_yaxis()
    ax_bars.set_xlim(0, 1.45)
    ax_bars.axvline(1, color="k", lw=0.5)
    ax_bars.set_xlabel("penalty (0 = no problem)")
    bar_labels = [ax_bars.text(0, i, "", va="center", fontsize=9) for i in y]

    ax_text.axis("off")
    text = ax_text.text(0, 1, "waiting for radar frames ...", va="top", family="monospace", fontsize=9,
                        transform=ax_text.transAxes)

    lines = {name: ax_hist.plot([], [], label=name)[0] for name in ("heatmap", "pointcloud", "combined")}
    ax_hist.set_xlim(-HISTORY_S, 0)
    ax_hist.set_ylim(0, 105)
    ax_hist.set_xlabel("seconds ago")
    ax_hist.set_ylabel("score")
    ax_hist.legend(loc="lower left", fontsize=8)
    ax_hist.grid(alpha=0.3)

    def on_key(event):
        with lock:
            if event.key == "b":
                monitor.start_reference()
            elif event.key == "o":
                i = OCCUPANCY_STATES.index(state["occupancy"])
                state["occupancy"] = OCCUPANCY_STATES[(i + 1) % len(OCCUPANCY_STATES)]

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        with lock:
            window, heatmap = state["window"], state["heatmap"]
            history = list(state["history"])
            occupancy = state["occupancy"]
            reference = monitor.reference
            lines_text = status_lines(window, monitor, occupancy, policy.reference_frames) if window else None
        if window is None:
            return
        if heatmap is not None and np.shape(heatmap) == context.shape:
            if reference is None:
                shown, label = to_db(heatmap), "power (dB, approx.)"
                lo, hi = np.percentile(shown, [5, 99.5])
            else:
                shown, label = reference.standardise(heatmap), "Z (change / empty-room spread)"
                lo, hi = -policy.z_change * 2, max(policy.thresholds["visibility_min"] * 2, 1.0)
            image.set_data(np.fft.fftshift(shown, axes=1))
            image.set_clim(lo, hi)
            colorbar.set_label(label)
            ax_map.set_title("change from empty room" if reference is not None else "raw heatmap (no reference)")

        for i, name in enumerate(TERMS):
            value = window.penalties.get(name)
            bars[i].set_width(0 if value is None else value)
            bars[i].set_color("tab:gray" if value is None else plt.get_cmap("RdYlGn_r")(min(value, 1.0)))
            bar_labels[i].set_x(0.02 if value is None else min(value, 1.0) + 0.02)
            reason = window.reasons.get(name)
            bar_labels[i].set_text(f"{value:.2f}" if value is not None else f"n/a: {reason}" if reason else "")
        used = set(policy.weights["heatmap"]) | set(policy.weights["pointcloud"])
        for label, name in zip(ax_bars.get_yticklabels(), TERMS):
            label.set_alpha(1.0 if name in used else 0.4)
        ax_bars.set_title("penalties" if window.status == "ready" else "penalties (warming up)")

        text.set_text("\n".join(lines_text))
        if history:
            now = history[-1][0]
            t = np.array([h[0] - now for h in history])
            for j, name in enumerate(("heatmap", "pointcloud", "combined"), start=1):
                lines[name].set_data(t, np.array([np.nan if h[j] is None else h[j] for h in history]))

    anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)  # noqa: F841
    plt.tight_layout()
    try:
        plt.show()
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
