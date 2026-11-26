"""Starts the annotation tool.

    python -m annotation_tool [--port 8765] [--csv data/annotations/lying_intervals.csv] [--no-browser] [--lan] [key=value ...]

The recordings offered are the ones the pipeline keeps for the config (`selection`). `key=value` pairs override the config like in `main.py`. The marks are saved in the CSV file at once.
Stop the tool with Ctrl+C.
"""

import argparse
import logging
import socket
import sys
import webbrowser
from typing import cast

from datasets import Dataset, load_from_disk
from omegaconf.errors import OmegaConfBaseException

from annotation_tool.server import Annotator, Catalogue, make_server
from data.annotations import LYING_INTERVALS, REPO
from data.preprocess import select_recordings
from config import load_data_config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("annotation_tool")


def lan_address() -> str:
    """The IP address of this machine on the local network (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
        except OSError:
            return "<ip of this machine>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--csv", default=None, help="where the marks are saved")
    parser.add_argument("--lan", action="store_true",
                        help="also serve on the local network (for a phone). The frames show people and anyone on the "
                             "network can open the page, so use a network you trust")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    args, overrides = parser.parse_known_args()
    try:
        cfg = load_data_config(REPO, overrides)
    except (ValueError, TypeError, OmegaConfBaseException) as error:
        sys.exit(f"Error: {error}")
    path = REPO / cfg.paths.baked_dir
    if not path.is_dir():
        sys.exit(f"No baked dataset at {path}.\nBuild it with: python main.py bake")

    ds = select_recordings(cast(Dataset, load_from_disk(str(REPO / cfg.paths.baked_dir))), cfg.selection)
    csv_path = REPO / args.csv if args.csv is not None else REPO / cfg.paths.annotations_dir / LYING_INTERVALS.name
    catalogue = Catalogue(ds)
    server = make_server(Annotator(catalogue, csv_path), args.port,  # an absolute path stays as it is
                         "0.0.0.0" if args.lan else "127.0.0.1")
    address = f"http://127.0.0.1:{server.server_port}/"
    if args.lan:
        logger.info("on the network: http://%s:%d/", lan_address(), server.server_port)
    logger.info("%d recordings, marks saved in %s", len(catalogue.names), csv_path)
    logger.info("open %s (Ctrl+C stops the tool)", address)
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
