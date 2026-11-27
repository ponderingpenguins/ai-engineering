"""Live range-Doppler heatmap from a TI mmWave radar connected over USB.

    uv run python scripts/viewer/live_range_doppler.py
    uv run python scripts/viewer/live_range_doppler.py --replay data/raw/floor_posture_dataset/data/room1/<session>

The cfg must enable the range-Doppler TLV (6th guiMonitor field = 1). Config_4 does.
"""

import argparse
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from vt_radar.config.ti_config import ConfigFile
from vt_radar.parsing_utils.parseFrame import parseStandardFrame

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "config/radar/Config_4.cfg"


def to_db(heatmap: np.ndarray) -> np.ndarray:
    # TI sends log2 magnitude in Q9; x / 512 * 6 is roughly dB.
    return heatmap.astype(np.float32) / 512 * 6


def connect_radar(cfg: ConfigFile):
    from vt_radar.recording_utils.record_tlv import find_com_ports, uartParser

    cli, data = find_com_ports()
    if cli is None or data is None:
        raise SystemExit(
            "Radar not found on any serial port. Check that the EVM shows on USB "
            "(System Information > USB) and that the Silicon Labs CP210x driver is installed."
        )
    uart = uartParser("SDK Out of Box Demo")
    uart.connectComPorts(cli, data)
    uart.sendCfg(cfg.cfg)
    return uart


def live_frames(uart):
    try:
        yield from uart.read_continuous()
    finally:
        uart.cliCom.write(b"sensorStop\n")
        uart.cliCom.close()
        uart.dataCom.close()


def replay_frames(directory: str, cfg: ConfigFile, fps: float):
    from vt_radar.data.dataset import read_directory

    for frame in read_directory(directory, config=cfg):
        yield frame.frame
        time.sleep(1 / fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg-file", default=str(DEFAULT_CFG))
    parser.add_argument("--replay", help="recording directory to play back instead of the live radar")
    parser.add_argument("--remove-static", action="store_true", help="subtract a running background (hides static clutter)")
    args = parser.parse_args()

    cfg_path = args.cfg_file
    if args.replay:
        cfgs = sorted(Path(args.replay).glob("*.cfg"))
        cfg_path = str(cfgs[0]) if cfgs else cfg_path
    cfg = ConfigFile()
    cfg.parseCfg(cfg_path)
    sensor = cfg.sensor()
    range_bins, doppler_bins = sensor.range_bins, sensor.doppler_bins
    frame_cfg = next(line.split() for line in cfg.cfg if line.startswith("frameCfg"))
    fps = 1000 / float(frame_cfg[5])

    uart = None if args.replay else connect_radar(cfg)

    latest: dict = {"heatmap": None, "n": 0, "t": deque(maxlen=20)}
    stop = threading.Event()

    def reader():
        frames = replay_frames(args.replay, cfg, fps) if args.replay else live_frames(uart)
        for raw in frames:
            if stop.is_set():
                break
            if not raw:
                continue
            try:
                heatmap = parseStandardFrame(raw, cfg).range_doppler_heatmap
            except Exception as e:  # a partial/garbled frame (e.g. right after reconnecting): skip it
                print(f"skipped corrupt frame: {type(e).__name__}: {e}", flush=True)
                continue
            if heatmap is None:
                continue
            latest["heatmap"] = heatmap
            latest["n"] += 1
            latest["t"].append(time.monotonic())
        frames.close()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    fig, (ax_rd, ax_rp) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2, 1]})
    blank = np.zeros((len(range_bins), len(doppler_bins)))
    extent = (doppler_bins[0], doppler_bins[-1], range_bins[0], range_bins[-1])
    image = ax_rd.imshow(blank, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    fig.colorbar(image, ax=ax_rd, label="power (dB, approx.)")
    ax_rd.set_xlabel("Doppler (m/s)")
    ax_rd.set_ylabel("Range (m)")
    (profile_line,) = ax_rp.plot(blank[:, 0], range_bins)
    ax_rp.set_xlabel("power at 0 m/s (dB)")
    ax_rp.set_ylim(range_bins[0], range_bins[-1])
    title = fig.suptitle("waiting for radar frames ...")
    background: dict = {"db": None}

    def update(_):
        heatmap = latest["heatmap"]
        if heatmap is None:
            return
        db = to_db(heatmap)
        profile = db[:, 0]
        if args.remove_static:
            bg = background["db"]
            background["db"] = db if bg is None else 0.95 * bg + 0.05 * db
            db = db - background["db"]
        db = np.fft.fftshift(db, axes=1)
        image.set_data(db)
        lo, hi = np.percentile(db, [5, 99.5])
        image.set_clim(lo, hi)
        profile_line.set_xdata(profile)
        ax_rp.set_xlim(profile.min() - 1, profile.max() + 1)
        ts = latest["t"]
        rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0
        title.set_text(f"frame {latest['n']}  |  {rate:.1f} fps")

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)  # noqa: F841
    plt.tight_layout()
    try:
        plt.show()
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
