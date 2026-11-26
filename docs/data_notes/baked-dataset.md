# The baked dataset

## Baking

The inputs are raw recordings in `data/raw/` (Git LFS) and source CSVs in `data/annotations/` (normal Git text). Every bake automatically reads both and joins the annotations into the generated dataset. `data/baked/` is entirely gitignored and can be deleted and rebuilt without losing annotation work.

`src/data/bake.py` reads every recording once through `vt_radar` and writes a single Hugging Face dataset (one row per frame) to `data/baked/floor_posture` (gitignored):

```
python main.py bake [paths.raw_root=...] [paths.baked_dir=...] [--num-proc 4] [--overwrite] [--recording a --recording b]
```

After extracting the data, synchronize the environment before opening the notebooks:

```bash
uv sync
source .venv/bin/activate
```

Recording names passed through `--recording` are literal strings; repeat the flag to select several. It takes about 20 s with `--num-proc 6`, and the result is about 4.4 GB.

To use the processed dataset:

```python
from datasets import load_from_disk

ds = load_from_disk("data/baked/floor_posture")
ds = ds.select_columns(["recording", "heatmap", "label_pose"])   # or ds.remove_columns([...])
```

Columns: `recording`, `room`, `frame_idx`, `timestamp`, `radar_frame_num`, `image` (camera JPEG), `radar_payload` (raw TLV bytes), `heatmap` (range x doppler, its size varies per recording), `pointcloud`, `n_points`/`n_tracks`/`n_heights`, the sensor config (`range_resolution_m`, `num_range_bins`, ...), and `label_<track>` plus `label_<track>_confidence` for every label track found. Tracks a recording doesn't have are null.

Two things to know about `heatmap`. It is stored raw, with zero velocity at doppler bin 0, so apply `np.fft.fftshift(heatmap, axes=-1)` to centre it (see the notebook). And 8 frames across 7 recordings have an empty heatmap (`[]`, and `heatmap_ok` is `False`), so filter on `heatmap_ok` before stacking.

`notebooks/exploration/explore_baked_dataset.ipynb` loads the dataset and plots a frame (camera, heatmap, point cloud), the heatmap over time with a label track, and label distributions.

Keep in mind that the baked data contains identifiable feeds.

## Manual annotations

Three columns of the baked dataset are not in the raw data: they were annotated by hand and are joined in by `src/data/annotations.py`. The source CSVs live in `data/annotations/` and are tracked as normal Git text. After changing them, refresh all three columns without re-parsing raw recordings:

```sh
python main.py bake --refresh-annotations
python main.py bake --refresh-annotations paths.annotations_dir=data/annotations paths.baked_dir=data/baked/floor_posture
```

Refresh requires an existing baked dataset and cannot be combined with `--recording` or `--overwrite`. It writes a sibling copy, releases Arrow memory maps, then swaps folders, restoring the old dataset if installation fails. Allow disk space for a second copy. If interrupted, inspect the sibling `.floor_posture.refresh_tmp` and `.floor_posture.old` recovery folders before retrying; refresh refuses to overwrite them. Use `--overwrite` for a full raw-data rebuild.

| Column | Source file | Made with | Meaning |
| ------ | ----------- | --------- | ------- |
| `person` | `data/annotations/person_map.csv` (`recording, person`) | `notebooks/labelling/label_persons.ipynb` | Pseudonym of the participant in the recording (for example `Bob`), so evaluation can be split by person. Recordings that are not labelled get `"NA"`. |
| `person_present` | `data/annotations/presence_events.csv` (`recording, frame_idx, event`) | `notebooks/labelling/label_presence.ipynb` | `True` while someone is in view of the camera, `False` for an empty scene. |
| `lying_annotation` | `data/annotations/lying_intervals.csv` (`recording, first_capture_idx, last_capture_idx, reviewed`) | `python -m annotation_tool` | `lying` or `not_lying` per frame, made from the camera video. `None` for recordings that are not reviewed yet. Training uses it as the target by default. |

- **`person`** is one value per recording. The notebook shows the best face crops of each recording and you pick a name from a fixed list of pseudonyms. Unlabelled recordings are dropped by the `known_person` preprocessor.
- **`person_present`** is stored as `enter` and `leave` events on frame indices. A person is present from each `enter` until the next `leave` (or the end of the recording), so a recording that starts with someone already in view needs an `enter` on frame 0. Recordings without any events get `None`, not `False`, so filter on `person_present.notna()` before training on it. This is separate from the raw `label_presence` track, which is not reliable (it is `False` wherever it exists).
- **`lying_annotation`** is stored as lying intervals on capture positions (the time order of a recording, not always `frame_idx`). In a reviewed recording every frame outside the intervals is `not_lying`. Recordings that are not reviewed get `None`, which is not the same as `not_lying`. Refresh it with `python main.py bake --refresh-annotations`, or call `attach_lying_annotation` from `data.annotations` on an in-memory dataset.
