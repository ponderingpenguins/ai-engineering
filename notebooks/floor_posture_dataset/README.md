# Floor-posture radar dataset

TI mmWave radar recordings of a person **lying on the floor**, with a
synchronized downward-facing camera as visual reference.

Recorded Sep–Nov 2023 in a single campaign. The camera looks straight down at a
carpeted floor from the ceiling; the radar is mounted alongside it, looking at
the same patch of floor. Every frame has both.

**35 recordings · ~5 GB · ~41,000 frames · two rooms.**

## The problem

Work out what the radar can tell you about a person lying on the floor — that
there is one, where they are, and which way they're lying — using the radar
alone. The camera is there so you can see what actually happened; it is
reference, not input.

This is deliberately open. Part of the exercise is deciding what is learnable
from this signal and proposing an approach.

## Layout

```
students_floor_posture/
├── explore_the_data.ipynb   START HERE — guided tour of the data and the radar config
├── MANIFEST.csv             one row per recording: frames, size, date, room
├── HEATMAP_SHAPES.csv       heatmap geometry per recording
├── README.md
├── previews/                one camera frame per recording — browse these first
├── recorder/                the package that reads the data
└── data/
    ├── room1/               31 recordings — the main room
    └── room2/                4 recordings — a second room, same rig
```

`explore_the_data.ipynb` is the fastest way in: it opens a recording, shows what
the radar config says the sensor can resolve, and puts the camera frame,
range-doppler heatmap and point cloud side by side. It runs on however many
recordings you have, so you can start from part 1 of the transfer alone.

```bash
pip install -e recorder
pip install jupyter
jupyter lab            # then open explore_the_data.ipynb
```

`previews/` holds a single frame from the middle of each recording, named to
match the directory in `data/`. Flip through those to see what each recording
contains before loading any of it.

Each recording directory contains:

| file | what it is |
| --- | --- |
| `data_*.h5` | 500 frames per chunk: raw radar payload + timestamp + JPEG camera image |
| `*.cfg` | the radar configuration this recording actually ran with |
| `labels/` | present on some recordings — see below |

`room2` is the same rig in a different room. Keep it separate: anything that
works in one room and not the other is telling you something important.

## Reading it

Install the bundled `vt-radar-recorder` package (see its own README), then:

```python
from vt_radar.data.dataset import TLVDataset

ds = TLVDataset('data/room1/2310__20231006_154117')
print(len(ds), 'frames')

for frame in ds:
    heat = frame.output.range_doppler_heatmap   # (256, 16) uint16 range-doppler
    pc   = frame.output.pointcloud              # (N, 7): x,y,z,doppler,snr,noise,track
    img  = frame.image                          # camera view, RGB
    t    = frame.timestamp
```

The `.cfg` inside each directory is picked up automatically — it's what tells the
parser how to decode the payload, so keep it with the data.

Raw, without the package: `pd.read_hdf('data_0.h5')` returns a DataFrame with
`frame` (radar bytes), `timestamp`, and `image` (JPEG bytes).

## What the signal looks like

Worth knowing before you start, because all three of these surprise people:

- **The point cloud is very sparse — 1 to 6 points per frame**, and the radar's
  built-in tracker reports **zero tracks**. A motionless person lying on a floor
  is a weak, still target, and the tracker is built for people walking. This is
  the nature of the problem, not a fault in the recordings.
- **Most of the usable information is in the range-doppler heatmap**, present on
  every frame of every recording here.
- **The heatmap is not the same size in every recording.** The doppler axis is
  always 16 bins, but the range axis varies with the radar config:

  | shape | recordings |
  | --- | --- |
  | 64 × 16 | 18 |
  | 128 × 16 | 6 |
  | 256 × 16 | 11 |

  `HEATMAP_SHAPES.csv` lists the shape per recording. You cannot simply stack
  them into one array — resample, crop, or train per-geometry, but decide
  deliberately. Note the range *resolution* differs too, so a bin index does not
  mean the same distance across recordings.
- Frame rate is ~18 fps, so consecutive frames are nearly identical. If you
  build any kind of train/test split, split by *recording* — splitting by frame
  will flatter your results enormously.

## Labels

Some recordings carry labels in `labels/`, and `MANIFEST.csv` flags which. Where
they exist the encoding is:

```
0 = moving (changing position)    1 = back    2 = side    3 = front    99 = standing
```

Treat these as a starting reference rather than gospel: they were produced by
more than one tool over the campaign, `front` only appears in part of it, and
boundary frames around a posture change are approximate. You are not obliged to
use them.

## Provenance

Recordings contain identifiable video of consenting participants. Use them for
this project only; don't redistribute them or publish frames without checking
first.
