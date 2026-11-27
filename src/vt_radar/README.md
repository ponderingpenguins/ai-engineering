# vt-radar recorder

Record and replay Texas Instruments (TI) mmWave radar data, with an optional synchronized
webcam.

This is the standalone **recording toolkit** — it captures radar frames (and video) to disk
and reads them back for analysis. It contains only what's needed for that: nothing about
tracking, detection, or the higher-level analytics.

- **Record** live radar → HDF5 datasets over the TI serial link, one webcam image per radar
  frame.
- **Replay** any recorded dataset by path, as `Frame` objects with point clouds, tracks and
  images.
- **Analyze** with pandas / matplotlib — everything converts to a `DataFrame`.

---

## 1. Requirements

- **Python 3.11** (3.9+ works)
- A **TI mmWave EVM** (e.g. IWR6843AOP) connected over USB
- Optional: a **webcam** for synchronized video
- Drivers for the EVM's USB-serial bridge:
  - **Windows** — the Silicon Labs CP210x VCP driver (Windows Update usually installs it;
    otherwise get it from Silicon Labs)
  - **macOS** — the Silicon Labs CP210x VCP driver, installed manually
  - **Linux** — nothing to install, but you must be in the `dialout` group:
    `sudo usermod -aG dialout $USER`, then log out and back in

## 2. Install

`vt_radar` is part of this repository's `src/` package and is installed automatically by the root environment setup:

```bash
uv sync
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows (PowerShell / cmd)
```

**Check it worked** — this needs no radar hardware:

```bash
uv run python -c "import vt_radar.main; print('install OK')"
```

If that prints `install OK` you have a working install, and any later problem is hardware
or configuration, not the package.

> **Don't upgrade numpy.** Root `pyproject.toml` pins `numpy<2` on purpose: pandas 2.2.x crashes
> on numpy 2.x when reading the HDF5 recordings. If you hit an unrelated error and are
> tempted to upgrade numpy, don't — it will break reading your own data in a
> confusing way.

## 3. Record

```bash
python main.py record --cfg-file config/radar/Config_4.cfg
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--cfg-file` | *(required)* | Path to a TI radar `.cfg`. Ready-to-use configs are bundled in `config/radar/`. |
| `--camera-port` | `1` | Webcam index for synchronized video. See below. |
| `--out-dir` | `recording` | Base output directory where `<YYYYMMDD_HHMMSS>/` session folders are written. |

A spinner shows it is running. Stop with **Ctrl-C** — the last partial chunk is flushed to
disk before exit, so you never lose more than the frames still in memory.

### What lands on disk

Each session is written to `recording/<YYYYMMDD_HHMMSS>/`, relative to the directory you
ran the command from:

```
recording/20240115_143359/
├── tracking_default.cfg   # copy of the exact cfg used for this recording
├── data_00000000.h5       # frames 0–499
├── data_00000001.h5       # frames 500–999
└── ...
```

- 500 frames per `.h5` chunk. Each chunk is written to a `.tmp` file and then atomically
  renamed, so a chunk on disk is always complete — killing the process mid-write cannot
  corrupt an earlier chunk.
- The `.cfg` copy is what makes a recording self-describing: the replay code finds it
  automatically and uses it to decode the TLV data. **Keep it with the recording.**
- Each row holds `idx`, `frame` (raw TLV bytes), `timestamp` (UTC), and `image`
  (JPEG bytes) when a camera is present.

### Recording video

Video is not a separate mode — the webcam runs alongside the radar and one image is stored
with every radar frame, so images and radar are index-aligned by construction. There is no
separate video file and no timestamp matching to do later.

**Pick the camera index.** `--camera-port` is an OpenCV capture index, and the default is
`1`, *not* `0`. On most laptops the built-in webcam is `0` and an external USB camera is
`1`. If you want the built-in camera:

```bash
python main.py record --cfg-file config/radar/tracking_default.cfg --camera-port 0
```

Not sure which index is which? Probe them before recording:

```python
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    print(i, "opened" if cap.isOpened() else "-")
    cap.release()
```

**Recording without a camera.** If the index cannot be opened, you get a single
`Webcam on '<n>' not opened` error and **recording continues normally, radar-only**. So an
absent or wrong camera costs you the video, not the session.

**Camera notes**

- On Windows the DirectShow backend is used automatically; on macOS/Linux the OpenCV
  default.
- The capture buffer is set to 1 frame so each stored image is the *most recent* one.
  Without this the camera buffers frames while the radar serial read blocks, and the video
  drifts steadily behind the radar.
- Images are stored as JPEG-encoded bytes, which is why recordings stay reasonably small.
- The radar and camera advance in lockstep. If the camera stalls, the recording loop stalls
  with it — see troubleshooting.

### Serial ports

The recorder auto-detects the EVM's two UARTs (CLI @ 115200, data @ 921600). On the TI EVMs
these are exposed by a Silicon Labs CP2105 dual-UART bridge (`10c4:ea70`); detection works
by VID/PID + interface number, so it is robust on macOS where the port descriptions are
generic.

### Recording troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| No ports found, or connect fails | Unplug/replug the EVM and retry. The radar often needs a USB power-cycle between runs — a previous run can leave the sensor in a started state. |
| Permission denied on the serial port (Linux) | You are not in the `dialout` group. See Requirements. |
| Spinner runs, but chunks stay empty / no frames | The `.cfg` was not accepted by this EVM. Try one of the other configs in `cfgs/` — they target different SDK demos and antenna layouts. |
| `Webcam on '1' not opened` | Wrong `--camera-port`, or the camera is in use by another app (Teams, Zoom, the browser). Radar recording still proceeds. |
| Recording stops on its own after a while | The camera stopped returning frames, which ends the capture loop. Re-run without video (`--camera-port 99`) to confirm, then check the USB camera / hub. |
| Nothing at all is written | You stopped before the first 500 frames *and* the process was killed rather than Ctrl-C'd. Only Ctrl-C flushes the partial chunk. |

## 4. Replay and analyze

Any recorded dataset replays by its directory path — no central index or registry needed.
The `.cfg` inside the directory is picked up automatically.

```python
from vt_radar.data.dataset import TLVDataset

ds = TLVDataset("/absolute/path/to/recording/20240115_143359")
print(f"{len(ds)} frames")

for frame in ds:
    pc = frame.output.pointcloud       # PointCloud, shape (N, 7)
    print(frame.timestamp, len(pc))
```

### What a frame gives you

| Accessor | Type | Contents |
| --- | --- | --- |
| `frame.timestamp` | `datetime` | UTC capture time |
| `frame.output.pointcloud` | `PointCloud` `(N, 7)` | detected points this frame |
| `frame.output.trackdata` | `TrackData` `(M, 16)` | tracked targets (tracking configs only) |
| `frame.output.num_detected_points` | `int` | point count |
| `frame.output.num_detected_tracks` | `int` | track count |
| `frame.image` | `Image` (H, W, 3) RGB | decoded webcam image, empty if none |
| `frame.frame` | `bytes` | the raw TLV payload, if you want to parse it yourself |

TLV parsing and JPEG decoding are **lazy and cached** — iterating a dataset and touching
only `frame.timestamp` costs almost nothing.

### Point cloud columns

`PointCloud` is a numpy array, so it does everything a numpy array does.
`to_pandas()` names the columns for you:

```python
df = frame.output.pointcloud.to_pandas()
```

| Column | Unit | Meaning |
| --- | --- | --- |
| `x`, `y`, `z` | m | position, sensor-relative cartesian |
| `Doppler` | m/s | radial velocity (+ away from the sensor) |
| `SNR` | see below | signal strength |
| `Noise` | see below | noise floor |
| `Track index` | — | id of the track this point was assigned to |

Which columns are actually populated depends on the TLV the cfg makes the device emit, and
this trips people up:

- **The 3D people-tracking configs** (`cfgs/3D_people_tracking/`, and the AOP configs) send
  *compressed spherical* points. `x`, `y`, `z`, `Doppler` and `SNR` are filled;
  **`Noise` is always 0** because that TLV does not carry it. `SNR` is scaled by a factor
  the device sends per frame, so treat it as a relative strength, not dB — on a typical
  recording it ranges from ~5 to a few thousand. The pre-conversion spherical values
  (`elevation`, `azimuth`, `rng`, `snr`) are appended as extra columns by `to_pandas()`.
- **The plain detected-points configs** send `SNR` and `Noise` in 0.1 dB steps, and no
  spherical extras.
- **`Track index` is 255 for unassociated points** — that is TI's "belongs to no track"
  marker, not track number 255. Filter with `df[df["Track index"] < 250]` before grouping
  by track, or you will get one giant fake track.

### Track columns

`frame.output.trackdata.to_pandas()` gives one row per tracked target:

`target_id`, `x`, `y`, `z` (m), `v_x`, `v_y`, `v_z` (m/s), `a_x`, `a_y`, `a_z` (m/s²),
`G` (gating gain), `conf` (confidence), plus four unused reserved fields.

Frames with nobody in view have no tracks, and there `to_pandas()` returns a completely
empty DataFrame — **no columns at all**, so `df["x"]` raises `KeyError`. Guard with
`if len(frame.output.trackdata):` before using it.

### Whole recording → one DataFrame

The usual starting point for analysis:

```python
import pandas as pd
from vt_radar.data.dataset import TLVDataset

ds = TLVDataset("recording/20240115_143359")

rows = []
for i, frame in enumerate(ds):
    df = frame.output.pointcloud.to_pandas()
    df["frame_idx"] = i
    df["timestamp"] = frame.timestamp
    rows.append(df)

points = pd.concat(rows, ignore_index=True)

print(points.groupby("frame_idx").size().describe())   # points per frame
print(points["SNR"].describe())                        # detection quality
```

From here everything is ordinary pandas — filter by `SNR`, bin by `z` to look at height,
group by `Track index` to follow a person, resample on `timestamp`.

### Plotting

Point cloud for one frame (a dataset iterates but does not index, so materialize once):

```python
import matplotlib.pyplot as plt

frames = list(ds)
df = frames[100].output.pointcloud.to_pandas()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df["x"], df["y"], df["z"], s=4, c=df["Doppler"], cmap="coolwarm")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
plt.show()
```

Radar and camera side by side — the sanity check that answers "is the radar seeing what I
think it is seeing?":

```python
frame = frames[100]
df = frame.output.pointcloud.to_pandas()

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
frame.image.plot(ax=left)                       # handles the no-image case
left.set_title(f"camera @ {frame.timestamp}")
right.scatter(df["x"], df["y"], s=6)
right.set_xlabel("x (m)"); right.set_ylabel("y (m)")
right.set_xlim(-4, 4); right.set_ylim(0, 8)
right.set_title(f"{len(df)} points")
plt.show()
```

### Large recordings

`TLVDataset` loads every frame into memory up front, which is fine for a few minutes of
data. For long recordings, stream instead — this reads one chunk file at a time:

```python
from vt_radar.data.dataset import read_directory, load_radar_config

config = load_radar_config("recording/20240115_143359/tracking_default.cfg")
for frame in read_directory("recording/20240115_143359", config=config):
    ...   # frames are yielded, not accumulated
```

`read_directory` also takes `min_timestamp` / `max_timestamp` to slice a time window, and
`sample_size` to read a random subset of chunk files.

### Known rough edges

- **A recording with zero readable chunks looks like an empty dataset**, not an error:
  `len(ds) == 0` with a warning in the log. If a dataset is unexpectedly empty, check the
  log output rather than assuming there were no detections.

Two earlier entries here — slicing a `PointCloud` breaking `to_pandas()`, and `plot_3d()`
indexing rows instead of columns — are fixed. `pc[pc[:, 4] > 15].to_pandas()` and
`pc.plot_3d()` both behave as you would expect now. If you are working from an export made
before August 2026, they still apply; ask for a newer drop.

## 5. Layout

```
src/vt_radar/
├── main.py            # record entry point (python main.py record)
├── recording_utils/   # serial capture, TLV recording, webcam
├── config/            # TI .cfg parsing
├── parsing_utils/     # frame / TLV decode
├── data/              # Frame, PointCloud, Image, dataset (replay)
├── mmwave/            # OpenRadar-derived readers (Apache-2.0)
├── tlv_constants.py
└── exceptions.py

config/radar/          # bundled radar configs
```

## 6. Updating

This toolkit is exported from a larger internal codebase, so updates arrive as a new drop
rather than through this folder's own history.

- **If you received a git repo** (it has a `.git` directory), a newer release is published
  to its `main`:

  ```bash
  git pull
  pip install -e .          # only if dependencies changed
  ```

- **If you received an archive**, you will simply be sent a newer one. Unpack it alongside
  the old copy rather than over it, and re-run `pip install -e .` in the new folder. Your
  `recording/` directories are yours — they are never part of a drop, so nothing you
  captured is at risk either way.

`vt_radar/mmwave/` is derived from TI's OpenRadar and is Apache-2.0. The rest of this
toolkit is Valtes-internal — ask before redistributing it.
