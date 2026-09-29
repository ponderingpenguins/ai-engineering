import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Bundle root. Adjust if you moved the notebook out of the bundle folder.
ROOT = "data/raw/floor_posture_dataset/"
if not os.path.isdir(os.path.join(ROOT, "data")):
    raise SystemExit(
        f"No data/ folder under {os.path.abspath(ROOT)}.\n"
        "Set ROOT to the folder that contains data/, recorder/ and previews/."
    )

RECORDER = os.path.abspath(os.path.join(ROOT, "recorder"))


def _load_reader():
    '''Import the bits of the reader this notebook uses.'''
    from loguru import logger
    from vt_radar.data.dataset import TLVDataset
    from vt_radar.data.rangedoppler import RangeDoppler
    from vt_radar.config.ti_config import ConfigFile
    return logger, TLVDataset, RangeDoppler, ConfigFile



def main():
    # The reader lives in recorder/. If it is not installed in the kernel running
    # this notebook, fall back to the copy sitting next to us rather than making you
    # fight with virtualenvs -- a common way to lose an afternoon is to pip install
    # into one Python and run Jupyter from another.
    try:
        logger, TLVDataset, RangeDoppler, ConfigFile = _load_reader()
    except ModuleNotFoundError:
        if os.path.isdir(RECORDER) and RECORDER not in sys.path:
            sys.path.insert(0, RECORDER)
        try:
            logger, TLVDataset, RangeDoppler, ConfigFile = _load_reader()
            print(f"vt_radar was not installed in this kernel; loaded it from {RECORDER}")
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"Did you forget to activate your environment?\n\n"
                f"source .venv/bin/activate"
            )

    logger.remove()          # the loader is chatty; silence it

    print("bundle  :", os.path.abspath(ROOT))
    print("python  :", sys.executable)
    print("reader  :", os.path.dirname(sys.modules["vt_radar.data.dataset"].__file__))

if __name__ == "__main__":
    main()
