"""Live 3D point cloud from a TI mmWave radar connected over USB.

    uv run python scripts/viewer/live_point_cloud.py
    uv run python scripts/viewer/live_point_cloud.py --persist 10
    uv run python scripts/viewer/live_point_cloud.py --replay data/raw/floor_posture_dataset/data/room1/<session>

Sensor coordinates: x = left/right, y = straight out from the board, z = up.
Points are coloured by radial velocity. The cfg must enable detected points
(1st guiMonitor field = 1). Config_4 does.
"""

import argparse
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from live_range_doppler import DEFAULT_CFG, connect_radar, live_frames, replay_frames
from matplotlib.animation import FuncAnimation
from vt_radar.config.ti_config import ConfigFile
from vt_radar.parsing_utils.parseFrame import parseStandardFrame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg-file", default=str(DEFAULT_CFG))
    parser.add_argument("--replay", help="recording directory to play back instead of the live radar")
    parser.add_argument("--persist", type=int, default=5, help="show points from the last N frames")
    args = parser.parse_args()

    cfg_path = args.cfg_file
    if args.replay:
        cfgs = sorted(Path(args.replay).glob("*.cfg"))
        cfg_path = str(cfgs[0]) if cfgs else cfg_path
    cfg = ConfigFile()
    cfg.parseCfg(cfg_path)
    sensor = cfg.sensor()
    max_range = float(sensor.range_bins[-1])
    max_vel = float(np.abs(sensor.doppler_bins).max())
    frame_cfg = next(line.split() for line in cfg.cfg if line.startswith("frameCfg"))
    fps = 1000 / float(frame_cfg[5])

    uart = None if args.replay else connect_radar(cfg)

    history: deque = deque(maxlen=max(1, args.persist))
    latest: dict = {"n": 0, "points": 0, "t": deque(maxlen=20)}
    lock = threading.Lock()
    stop = threading.Event()

    def reader():
        frames = replay_frames(args.replay, cfg, fps) if args.replay else live_frames(uart)
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
            pts = pc[:n, :4].copy() if n else np.empty((0, 4))  # x, y, z, doppler
            with lock:
                history.append(pts)
                latest["n"] += 1
                latest["points"] = n
                latest["t"].append(time.monotonic())
        frames.close()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)
    norm = plt.Normalize(-max_vel, max_vel)
    cmap = "coolwarm"

    scat3d = ax3d.scatter([], [], [], c=[], cmap=cmap, norm=norm, s=12)
    ax3d.set_xlim(-max_range / 2, max_range / 2)
    ax3d.set_ylim(0, max_range)
    ax3d.set_zlim(-2, 2)
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.scatter([0], [0], [0], marker="^", c="k", s=40)  # radar position

    scat_top = ax_top.scatter([], [], c=[], cmap=cmap, norm=norm, s=14)
    ax_top.plot(0, 0, "k^", ms=8)
    ax_top.set_xlim(-max_range / 2, max_range / 2)
    ax_top.set_ylim(0, max_range)
    ax_top.set_aspect("equal")
    ax_top.grid(alpha=0.3)
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_title("top-down")
    fig.colorbar(scat_top, ax=ax_top, label="radial velocity (m/s)")
    title = fig.suptitle("waiting for radar frames ...")

    def update(_):
        with lock:
            if not history:
                return
            pts = np.concatenate(list(history))
            n, last, ts = latest["n"], latest["points"], list(latest["t"])
        scat3d._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        scat3d.set_array(pts[:, 3])
        scat_top.set_offsets(pts[:, :2])
        scat_top.set_array(pts[:, 3])
        rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0
        title.set_text(f"frame {n}  |  {rate:.1f} fps  |  {last} points this frame, {len(pts)} shown")

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)  # noqa: F841
    plt.tight_layout()
    try:
        plt.show()
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
