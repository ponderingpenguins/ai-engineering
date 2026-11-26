"""Builds a small standalone viewer of the Config 4 camera frames and their pose labels.

    python scripts/viewer/config4_viewer.py [--out results/viewer] [--width 224]

Writes `index.html` (plain HTML and JS, no libraries), `data.js` (the labels) and `frames/<recording>/<n>.jpg`
(downscaled camera frames) into the output folder, then open `index.html` in a browser. The folder is under
`results/`, which git ignores: the frames show identifiable people, so keep it local.
"""

import argparse
import io
import json
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import Image as HFImage
from datasets import load_from_disk
from PIL import Image

from data.annotations import capture_positions

REPO = Path(__file__).resolve().parents[2]
CONFIG_4 = "Config_4.cfg"
PERIOD_S = 0.2  # seconds per frame
# one character per frame in data.js
CLASSES = [
    ("s", "standing", "#6baed6"),
    ("m", "moving", "#c6dbef"),
    ("t", "sitting", "#fdae6b"),
    ("l", "lying_left", "#d7301f"),
    ("r", "lying_right", "#a50f15"),
    ("?", "not labelled", "#dddddd"),
]
KEY = {name: key for key, name, _ in CLASSES}


def lying_runs(labels: str) -> list[list[int]]:
    """[first, last] frame of every run of consecutive lying frames."""
    flags = np.array([KEY["lying_left"] == c or KEY["lying_right"] == c for c in labels], dtype=int)
    edges = np.diff(np.r_[0, flags, 0])
    return [[int(a), int(b) - 1] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results/viewer", help="output folder, relative to the repo")
    parser.add_argument("--width", type=int, default=224, help="frame width in pixels; the height keeps 4:3")
    parser.add_argument("--quality", type=int, default=55, help="JPEG quality")
    args = parser.parse_args()

    out = REPO / args.out
    ds = cast(Dataset, load_from_disk(str(REPO / "data/baked/floor_posture")))
    meta = cast(pd.DataFrame, ds.select_columns(["recording", "person", "cfg_file", "timestamp", "frame_idx", "label_pose"]).to_pandas())
    meta = meta[meta.cfg_file == CONFIG_4].copy()
    meta["position"] = capture_positions(meta)
    meta = meta.sort_values(["recording", "position"])
    images: Any = ds.select_columns(["image"]).cast_column("image", HFImage(decode=False))  # raw JPEG bytes
    size = (args.width, args.width * 3 // 4)

    shutil.rmtree(out / "frames", ignore_errors=True)
    recordings = []
    for name, rec in meta.groupby("recording"):
        folder = out / "frames" / name
        folder.mkdir(parents=True)
        for n, row in enumerate(rec.index):
            im = Image.open(io.BytesIO(images[int(row)]["image"]["bytes"]))
            im.draft("RGB", (size[0] * 2, size[1] * 2))  # lets the JPEG decoder skip work
            im.convert("RGB").resize(size, Image.Resampling.BILINEAR).save(folder / f"{n:04d}.jpg", quality=args.quality, optimize=True)
        labels = "".join(KEY.get(p, KEY["not labelled"]) if isinstance(p, str) else KEY["not labelled"]
                         for p in rec.label_pose)
        person = rec.person.iloc[0]
        recordings.append({"name": name, "person": "unknown" if person == "NA" else person,
                           "labels": labels, "runs": lying_runs(labels)})
        print(f"{name}: {len(rec)} frames, {len(recordings[-1]['runs'])} lying runs")

    viewer = {"period": PERIOD_S, "classes": [{"key": k, "name": n, "colour": c} for k, n, c in CLASSES],
              "recordings": recordings}
    (out / "data.js").write_text("const VIEWER = " + json.dumps(viewer) + ";\n")
    shutil.copy(Path(__file__).with_name("viewer_template.html"), out / "index.html")
    megabytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"\nwrote {out} ({megabytes:.0f} MB). Open {out / 'index.html'} in a browser.")


if __name__ == "__main__":
    main()
