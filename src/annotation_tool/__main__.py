"""Starts the annotation tool.

    python -m annotation_tool [--port 8765] [--csv data/processed/lying_intervals.csv] [--no-browser] [key=value ...]

The recordings offered are the ones the pipeline keeps for the config (`data.preprocess`, `data.exclude`, known
person). `key=value` pairs override the config like in `main.py`. The marks are saved in the CSV file at once.
Stop the tool with Ctrl+C.
"""

import argparse
import logging
import sys
import webbrowser
from typing import cast

from datasets import Dataset, load_from_disk
from omegaconf.errors import OmegaConfBaseException

from annotation_tool.server import Annotator, Catalogue, make_server
from data.annotation import LYING_INTERVALS, REPO
from data.preprocess import PREPROCESSORS, RowFilter, build_pipeline
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("annotation_tool")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--csv", default=str(LYING_INTERVALS), help="where the marks are saved")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    args, overrides = parser.parse_known_args()
    try:
        cfg = load_config(REPO, overrides)
    except OmegaConfBaseException as error:
        sys.exit(f"Error: {error}")
    path = REPO / cfg.data.baked
    if not path.is_dir():
        sys.exit(f"No baked dataset at {path}.\nBuild it with: python -m data.extract")

    # only the filters that choose recordings: the heatmap steps would be slow and the tool does not show heatmaps
    steps = [name for name in cfg.data.preprocess if issubclass(PREPROCESSORS[name], RowFilter)]
    ds = build_pipeline(steps, cfg.data.exclude)(cast(Dataset, load_from_disk(str(path))))
    catalogue = Catalogue(ds)
    server = make_server(Annotator(catalogue, REPO / args.csv), args.port)  # an absolute path stays as it is
    address = f"http://127.0.0.1:{server.server_port}/"
    logger.info("%d recordings, marks saved in %s", len(catalogue.names), args.csv)
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
