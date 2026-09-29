import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Need this because were not in the same dir
ROOT = "data/raw/floor_posture_dataset/"
if not os.path.isdir(os.path.join(ROOT, "data")):
    raise SystemExit(
        f"No data/ folder under {os.path.abspath(ROOT)}.\n"
        "Set ROOT to the folder that contains data/, recorder/ and previews/."
    )

RECORDER = os.path.abspath(os.path.join(ROOT, "recorder"))

try:
    from loguru import logger
    from vt_radar.data.dataset import TLVDataset
    from vt_radar.data.rangedoppler import RangeDoppler
    from vt_radar.config.ti_config import ConfigFile
    logger.remove() # the loader is chatty; silence it
except ModuleNotFoundError:
    raise SystemExit(
        f"Did you forget to activate your environment?\n\n"
        f"source .venv/bin/activate"
    )
finally:

    print("bundle  :", os.path.abspath(ROOT))
    print("python  :", sys.executable)
    print("reader  :", os.path.dirname(sys.modules["vt_radar.data.dataset"].__file__))




def load_cfg(rec: str) -> tuple[ConfigFile, str]:
    '''Parse the .cfg that shipped with a recording.'''
    d = rec_dir(rec)
    cfgs = sorted(glob.glob(os.path.join(d, "*.cfg")))
    if not cfgs:
        raise FileNotFoundError(f"no .cfg in {d}")
    c = ConfigFile()
    c.parseCfg(cfgs[0])
    return c, os.path.basename(cfgs[0])

def rec_dir(rec):
    '''Directory for a recording name, whichever room it is in.'''
    hits = glob.glob(os.path.join(ROOT, "data", "*", rec))
    if not hits:
        raise FileNotFoundError(f"{rec} is not in this folder (another zip part?)")
    return hits[0]


def cfg_summary(rec):
    c, name = load_cfg(rec)
    s = c.sensor()
    return {
        "recording": rec,
        "cfg": name,
        "range_res_m": round(s.range_resolution, 4),
        "range_max_m": round(s.range_max, 2),
        "range_bins": s.num_range_bins,
        "doppler_bins": s.num_doppler_bins,
        "doppler_res_m_s": round(s.doppler_resolution, 4),
        "max_velocity_m_s": round(s.max_velocity, 3),
        "tx": s.num_tx_antennas,
        "rx": s.num_rx_antennas,
    }


def main() -> None:

    man = pd.read_csv(os.path.join(ROOT, "MANIFEST.csv"))
    shapes = pd.read_csv(os.path.join(ROOT, "HEATMAP_SHAPES.csv"))
    man = man.merge(shapes[["rec", "heatmap", "cfg"]], left_on="dataset", right_on="rec", how="left")

    available = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "data", "*", "*"))}
    man["present"] = man.dataset.isin(available)

    print(f"{man.present.sum()} of {len(man)} recordings present in this folder")
    print(f"{man.loc[man.present, 'frames'].sum():,} frames, {man.loc[man.present, 'size_mb'].sum():.0f} MB")
    print(man.loc[man.present, ["dataset", "room", "frames", "size_mb", "heatmap", "recorded", "has_labels"]])
    print()

    rows = []
    for rec in man.dataset:
        try:
            rows.append(cfg_summary(rec))
        except Exception as e:
            print(e)
            rows.append({"recording": rec, "cfg": f"<{type(e).__name__}>"})
    cfgs = pd.DataFrame(rows)
    print(cfgs.head())


    REC = man.dataset.iloc[0]          # change this to explore another recording
    ds = TLVDataset(rec_dir(REC))
    frames = list(ds)
    print(f"{REC}: {len(frames)} frames")

    f = frames[len(frames) // 2]
    print("timestamp     :", f.timestamp)
    print("heatmap       :", np.asarray(f.output.range_doppler_heatmap).shape)
    print("point cloud   :", np.asarray(f.output.pointcloud).shape, "(x, y, z, doppler, snr, noise, track)")
    print("camera image  :", np.asarray(f.image).shape)
    print("tracks        :", f.output.num_detected_tracks)


if __name__ == "__main__":
    main()













