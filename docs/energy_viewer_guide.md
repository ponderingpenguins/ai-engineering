# Energy viewer: user guide

The energy viewer shows the quality of the radar data while you record. It calculates penalties (0 = no problem, 1 = bad) and scores (100 = no problem) for the heatmap and the point cloud. For the theory, see [energy_research_1.md](energy_research_1.md).

## Start the viewer

Connect the radar over USB, then run:

```
uv run python scripts/viewer/live_energy.py --reference data/quality/room1_reference.npz
```

To play back a recording without the radar:

```
uv run python scripts/viewer/live_energy.py --replay data/raw/floor_posture_dataset/data/room1/<session>
```

| Flag | Function |
|---|---|
| `--reference PATH` | Load the empty-room reference from `PATH`. A new reference is saved to `PATH`. |
| `--replay DIR` | Play back a recording directory. |
| `--occupancy STATE` | Start state: `unknown` (default), `present` or `empty`. |
| `--profile NAME` | Scoring profile from `config/quality.yaml`. Default: `visibility_only_v1`. |
| `--roi-range MIN MAX` | Range of the region of interest in metres. Default: 0.3 to 4.85. |
| `--cfg-file PATH` | Radar configuration. Default: `config/radar/Config_4.cfg`. |

## Procedure

1. Start the viewer.
2. Make sure that nobody is in the room. Stay out of the radar's field of view.
3. Push `b`. The viewer collects 50 frames (approximately 10 s). The status text shows "reference: collecting 23/50".
4. When the status text shows "reference: ready", the viewer saves the reference to the `--reference` path.
5. Go into the room. Push `o` until the status text shows "occupancy: present".
6. Wait 16 frames (approximately 3 s). The scores then appear.
7. Before you leave the room, push `o` again to set `empty` or `unknown`.

| Key | Function |
|---|---|
| `b` | Collect a new empty-room reference. |
| `o` | Change occupancy: unknown → present → empty → unknown. |

## The panels

- **Top left:** the change from the empty room (Z), over range and Doppler. Bright cells increased, dark cells decreased. The dashed lines show the ROI. Without a reference, the panel shows the raw heatmap.
- **Top right:** one bar for each penalty. Grey labels are penalties that the profile does not use. "n/a" shows the reason why a penalty is missing.
- **Bottom left:** the status text, with occupancy, reference state, scores, measurements and warnings.
- **Bottom right:** the heatmap, point-cloud and combined scores for the last 60 s.

## Penalties

| Penalty | Problem it shows |
|---|---|
| `visibility` | The heatmap does not change sufficiently from the empty room. |
| `focus` | The change has too little structure. Only `visibility_focus_v1` uses it. |
| `point_count` | Too few points in the ROI in the 16-frame window. |
| `empty_frames` | Too many frames have no points in the ROI. |
| `point_snr` | The median SNR of the ROI points is low. |
| `point_cap` | More points than the model can use (128). |

## Rules

- **Scores appear only when occupancy is `present` for all 16 frames.** With `unknown` or `empty`, you see the measurements and the warnings, but no scores. The viewer does not find occupancy by itself.
- **Make a new reference after each change to the room, the radar position or the configuration.** The viewer refuses a reference file that was made with a different configuration.
- **Never collect a reference while a person is in the room.** The person then becomes part of the background.
- **A frame gap or a corrupt frame starts a new window.** The viewer then shows "warming up" for 16 frames.
- **A score is not a probability.** Compare scores only under the same profile and `policy_id`.
- **Visibility shows presence, not posture.** A good heatmap score does not mean that the model can see that a person is lying.

## Thresholds

The thresholds are in `config/quality.yaml`. They come from the Config_4 recordings, but only 13 empty windows were available. To tune them again:

```
uv run python scripts/analysis/tune_quality_policy.py
```

The script prints proposed values. It does not change the YAML file. Record more empty-room data to get better values.
