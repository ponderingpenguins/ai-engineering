"""CLI entry point for live TI mmWave radar + webcam recording.

Usage: python main.py record --cfg-file config/radar/Config_4.cfg [--camera-port 1] [--out-dir recording]
"""
import argparse
import os

from vt_radar.recording_utils import recorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Valtes RadarToolbox recorder")
    parser.add_argument("--cfg-file", dest="cfg_file", help="Path to the TI cfg input file", type=str)
    parser.add_argument("--camera-port", dest="camera_port", help="Camera port", type=int, default=1)
    parser.add_argument("--out-dir", dest="out_dir", help="Directory where session folders are written", type=str,
                        default="recording")

    args = parser.parse_args()

    if not args.cfg_file:
        parser.error("You must provide a filename.")

    if not os.path.isfile(args.cfg_file):
        parser.error(f"The provided config file '{args.cfg_file}' is not a valid file.")

    recorder.do_record(args)


if __name__ == "__main__":
    main()
