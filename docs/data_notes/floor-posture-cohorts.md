# Floor-posture recordings: what was removed, and how to group the rest

## Ten recordings were removed

Every recording using `Experiment_1.cfg` has been deleted from `data/raw/`.
They cannot give you a point cloud, and they were costing you time.

The reason is one line in their config:

```
guiMonitor -1 0 0 0 0 1 0        Experiment_1.cfg   -- the 10 removed
guiMonitor -1 1 1 0 0 1 1        every other config
```

The second field is `detectedObjects`. At `0` the firmware never sends the
point-cloud TLV, so the reader produces a single placeholder `(0, 0, 0)` per
frame. That reads as "one detected point" when it means "no point cloud at
all" — which is why these recordings look like a person who never reflects
anything.

They were unusable for a second reason too. Their range bin is **0.7687 m**
across a 49 m span, against 0.0237–0.0469 m for everything else. A person
lying 1.9 m from the sensor occupies two or three bins, and the whole room
occupies four.

Removed (10 recordings, 167 `.h5` files):

```
room1/2309__20230927_145000     room1/2310__20231027_134002
room1/2310__20231006_154117     room1/2310__20231027_140700
room1/2310__20231006_155527     room1/2310__20231027_142000
room1/2310__20231006_161334     room1/2310__20231027_143141
room1/2310__20231020_143847     room1/2310__20231027_150820
```

They are still in the Valtes archive, so nothing is lost — ask if you ever
want them back. **25 recordings remain**, and they hold every labelled frame:
the removed ten had no manual labels at all.

## Group what remains by the grid, not the filename

`floor-posture-cohorts.csv`, beside this file, has one row per recording.
The remaining 25 fall into six groups that must not be pooled without thought:

| cohort | bins | m/bin | span | clutter | recs | labelled |
|---|---|---|---|---|---|---|
| A | 256 × 16 | 0.0237 | 6.07 m | on | 9 | 14,652 |
| B | 128 × 16 | 0.0234 | 3.00 m | on | 2 | 2,700 |
| C | 64 × 16 | 0.0469 | 3.00 m | on | 4 | 1,000 |
| D | 64 × 16 | 0.0469 | 3.00 m | **off** | 4 | 0 |
| G | 128 × 16 | 0.0436 | 5.58 m | on | 4 | 0 |
| H | 256 × 16 | 0.0237 | 6.07 m | **off** | 2 | 0 |

Cohort A holds 80% of the labelled frames and is the only group containing
`front` or `standing`.

Three things that will bite otherwise:

- **`Config_4.cfg` names three different files** here, and `Config_3.cfg`
  another three. Group on `cfg_md5` or on `bin_spacing_m` + `range_bins`.
- **`(64, 16)` and `(256, 16)` each occur at two different scales.** The
  tensors stack without complaint and mean different things.
- **`bin_spacing_m` converts an index to metres; `true_res_m` says what can be
  told apart.** They differ where `numAdcSamples` is not a power of two,
  because the range FFT zero-pads. Do not use one for the other.

## Two smaller things

**The label files are defective and are being reissued.** `pose.csv` and
`activity.csv` carry UTC nanosecond timestamps while the others carry local
seconds — an hour or two apart, following daylight saving — and no label file
has a `session` column, so `frame_num` is not unique. A corrected set is coming
as its own addon. Until then, do not join label tracks on `timestamp`.

**Frames do not arrive in capture order.** The `.h5` files are read
alphabetically, so `data_10.h5` follows `data_1.h5`. `explore_the_data.ipynb`
now has `frames_of(rec, order)`, which sorts by each frame's own timestamp —
use it instead of `list(TLVDataset(...))` for anything plotted against time or
aligned positionally against labels.
