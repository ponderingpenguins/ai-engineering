"""Slim recording module that uses uartParser + Webcam directly.

This bypasses the full framework chain (IntakeRadar -> RadarIntake -> BaseIntake -> Producer)
so that recording can run on resource-constrained devices like a Raspberry Pi.
"""
import time
import math
import threading
import signal
import os
import shutil

import pandas as pd
from datetime import datetime
from loguru import logger
from halo import Halo

from vt_radar.recording_utils.record_tlv import uartParser
from vt_radar.recording_utils.webcam import Webcam
from vt_radar.config.ti_config import ConfigFile

# Thread-safe abort flag
abort_please = threading.Event()


def signal_handler(sig, fr):
    logger.info("Stop recording on Ctrl-C")
    abort_please.set()


def do_record(args):
    cfg_file = args.cfg_file
    logger.info("Start recording usb camera frames")

    rows = []
    FRAMES_PER_FILE = 500

    cfg = ConfigFile()
    cfg.parseCfg(cfg_file)

    # Connect to radar directly — no framework needed
    demo_type = "SDK Out of Box Demo"
    uart = uartParser(demo_type)
    uart.auto_connect()
    uart.sendCfg(cfg.cfg)

    webcam = Webcam(None, capture_id=args.camera_port)

    directory = getattr(args, "out_dir", "recording")
    spinner = Halo(text='Recording... ', spinner='bouncingBar', color='green', text_color='cyan')
    spinner.start()

    # Get the current date and time in a sortable format
    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
    recording_dir = os.path.join(directory, formatted_datetime)

    if not os.path.exists(recording_dir):
        os.makedirs(recording_dir)

    logger.info(f"Recording to destination: {recording_dir}")

    config_filename = os.path.basename(cfg_file)
    destination = os.path.join(recording_dir, config_filename)
    shutil.copy2(cfg_file, destination)

    signal.signal(signal.SIGINT, signal_handler)

    def write_chunk(chunk_rows: list, chunk_index: int) -> None:
        filename = 'data_' + "%08d" % chunk_index + '.h5'
        fn = os.path.join(recording_dir, filename)
        tmp = fn + '.tmp'
        pd.DataFrame(chunk_rows).to_hdf(tmp, key='df', mode='w')
        os.replace(tmp, fn)

    frame_idx = 0
    for image, frame_data in zip(webcam.read_continuous(), uart.read_continuous()):
        row = {"idx": frame_idx, "frame": frame_data, "timestamp": datetime.utcnow()}
        if image is not None:
            row["image"] = image
        rows.append(row)
        frame_idx += 1

        if len(rows) == FRAMES_PER_FILE:
            write_chunk(rows, math.floor(row["idx"] / FRAMES_PER_FILE))
            rows = []

        if abort_please.is_set():
            logger.error("Aborting recording, writing last bits ..")
            if len(rows) > 0:
                write_chunk(rows, math.floor(row["idx"] / FRAMES_PER_FILE))
                rows = []
            break

    logger.info("Done")
    spinner.stop()
