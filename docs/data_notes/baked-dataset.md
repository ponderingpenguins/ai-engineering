# The baked dataset

## Baking

`src/data/extract.py` reads every recording once through `vt_radar` and writes a single Hugging Face dataset (one row per frame) to `data/processed/floor_posture` (gitignored):

```
python -m data.extract [data.root=...] [data.baked=...] [num_proc=4] [overwrite=true] ['recordings=["a","b"]']
```

After extracting the data, synchronize the environment before opening the notebooks:

```bash
uv sync
source .venv/bin/activate
```

Quote recording names: YAML reads an unquoted `2311__20231120_144541` as a number. It takes about 20 s with `num_proc=6`, and the result is about 4.4 GB.

To use the processed dataset:

```python
from datasets import load_from_disk

ds = load_from_disk("data/processed/floor_posture")
ds = ds.select_columns(["recording", "heatmap", "label_pose"])   # or ds.remove_columns([...])
```

Columns: `recording`, `room`, `frame_idx`, `timestamp`, `radar_frame_num`, `image` (camera JPEG), `radar_payload` (raw TLV bytes), `heatmap` (range x doppler, its size varies per recording), `pointcloud`, `n_points`/`n_tracks`/`n_heights`, the sensor config (`range_resolution_m`, `num_range_bins`, ...), and `label_<track>` plus `label_<track>_confidence` for every label track found. Tracks a recording doesn't have are null.

Two things to know about `heatmap`. It is stored raw, with zero velocity at doppler bin 0, so apply `np.fft.fftshift(heatmap, axes=-1)` to centre it (see the notebook). And 8 frames across 7 recordings have an empty heatmap (`[]`, and `heatmap_ok` is `False`), so filter on `heatmap_ok` before stacking.

`notebooks/explore_baked_dataset.ipynb` loads the dataset and plots a frame (camera, heatmap, point cloud), the heatmap over time with a label track, and label distributions.

Keep in mind that the baked data contains identifiable feeds.

## Manual annotations

Three columns of the baked dataset are not in the raw data: they were annotated by hand and are joined in by `src/data/extract.py`. The source CSVs are small and tracked in git (see `.gitignore`), so rebuild the dataset (`overwrite=true`) after changing them.

| Column | Source file | Made with | Meaning |
| ------ | ----------- | --------- | ------- |
| `person` | `data/processed/person_map.csv` (`recording, person`) | `notebooks/label_persons.ipynb` | Pseudonym of the participant in the recording (for example `Bob`), so evaluation can be split by person. Recordings that are not labelled get `"NA"`. |
| `person_present` | `data/processed/presence_events.csv` (`recording, frame_idx, event`) | `notebooks/label_presence.ipynb` | `True` while someone is in view of the camera, `False` for an empty scene. |
| `lying_annotation` | `data/processed/lying_intervals.csv` (`recording, first_capture_idx, last_capture_idx, reviewed`) | `python -m annotation_tool` | `lying` or `not_lying` per frame, made from the camera video. `None` for recordings that are not reviewed yet. Training uses it as the target by default. |

- **`person`** is one value per recording. The notebook shows the best face crops of each recording and you pick a name from a fixed list of pseudonyms. Unlabelled recordings are dropped by the `known_person` preprocessor.
- **`person_present`** is stored as `enter` and `leave` events on frame indices. A person is present from each `enter` until the next `leave` (or the end of the recording), so a recording that starts with someone already in view needs an `enter` on frame 0. Recordings without any events get `None`, not `False`, so filter on `person_present.notna()` before training on it. This is separate from the raw `label_presence` track, which is not reliable (it is `False` wherever it exists).
- **`lying_annotation`** is stored as lying intervals on capture positions (the time order of a recording, not always `frame_idx`). In a reviewed recording every frame outside the intervals is `not_lying`. Recordings that are not reviewed get `None`, which is not the same as `not_lying`. Add the column to an existing dataset without a full rebuild by running `_add_lying_annotation` from `data/extract.py` on it, or rebuild with `overwrite=true`.
