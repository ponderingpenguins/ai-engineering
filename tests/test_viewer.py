import io
import json
import sys
from datetime import datetime, timedelta

from datasets import Dataset
from PIL import Image

from scripts.viewer import config4_viewer as viewer


def test_exported_frames_and_labels_follow_capture_order(tmp_path, monkeypatch):
    # frame_idx 3-5 belong between 0 and 1; timestamps 1 and 2 also tie.
    frame_ids = [0, 3, 4, 5, 1, 2]
    labels = ["standing", "lying_left", "lying_right", "moving", "sitting", "standing"]
    images = []
    for frame_idx in frame_ids:
        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), (40 * frame_idx, 0, 0)).save(buffer, "JPEG", quality=95)
        images.append({"bytes": buffer.getvalue(), "path": None})
    start = datetime(2026, 1, 1)
    ds = Dataset.from_dict({
        "recording": ["a"] * 6, "person": ["Bob"] * 6, "cfg_file": [viewer.CONFIG_4] * 6,
        "frame_idx": frame_ids,
        "timestamp": [start + timedelta(seconds=i) for i in (0, 1, 1, 2, 3, 3)],
        "label_pose": labels, "image": images,
    }).shuffle(seed=0).flatten_indices()
    monkeypatch.setattr(viewer, "load_from_disk", lambda path: ds)
    monkeypatch.setattr(viewer, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["config4_viewer.py", "--out", "viewer", "--width", "16", "--quality", "95"])

    viewer.main()

    out = tmp_path / "viewer"
    saved = json.loads((out / "data.js").read_text().removeprefix("const VIEWER = ").rstrip(";\n"))
    assert saved["recordings"] == [{"name": "a", "person": "Bob", "labels": "slrmts", "runs": [[1, 2]]}]
    for n, frame_idx in enumerate(frame_ids):
        with Image.open(out / "frames" / "a" / f"{n:04d}.jpg") as image:
            assert abs(image.getpixel((0, 0))[0] - 40 * frame_idx) <= 3
    assert (out / "index.html").is_file()
