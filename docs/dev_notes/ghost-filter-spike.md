# Ghost-point filter and room-size estimate: abandoned after two spikes

## Idea

Radar points can be reflections (ghosts) of a real point. We wanted to find the reflecting planes (floor, walls) from the points, remove the ghost points before training, and estimate the room size as a side result.

Decisions made before the spikes:

- A filter that deletes ghost points in the frame-level preprocess stage. The ghost list is saved to a side file.
- One geometry fit per recording, because the sensor moved between sessions.
- Floor and walls only, one bounce only.
- A ghost is the farther point of a pair, and only if its SNR is also lower.
- Tune on the agreement of wall distances across recordings. Use the lying score only once at the end.
- Same recordings as training (`data.exclude`), Config_4 only.

## Spike 1: bisector planes of point pairs

`scripts/ghost_plane_spike.py`, `notebooks/spike_ghost_planes.ipynb`.

For all pairs in one frame, the plane that mirrors P onto G is the perpendicular bisector of P-G. We voted for the planes and compared with a control: pairs of frames that are far apart in time. Pairs closer than 0.3 m were dropped.

Result on 3 recordings of `room1` (`20231115_135843`, `20231120_144541`, `20231120_153152`):

- Floor: best peak 2 to 4 sigma, at 1.2 m, 2.9 m and 3.0 m. The floor is at about 1.8 m. Below the 5 sigma rule.
- Walls: no cell above 5 sigma.
- About 2 pairs per frame passed the rules.

## Spike 2: floor ghost straight behind a real point

`scripts/floor_ghost_spike.py`, `notebooks/spike_floor_ghost.ipynb`.

The sensor looks down. A floor bounce puts a ghost at about `y_G = 2h - y_P`, at the same x and z. Spike 1 could not see this, because it dropped close pairs. Here we look for a weaker point 0.1 m or more farther along y, within 0.3 m in x and z. Each pair gives `h = (y_P + y_G) / 2`.

Result on the same 3 recordings:

- Best peak for all pairs: 2.2 m, 2.0 m, 0.8 m. Only one recording has a peak near the floor (1.98 m, 5.0 sigma).
- There are 1.3 times more pairs inside a frame than in the control in two recordings, and 1.2 to 1.7 times with Doppler agreement in all three.
- This excess is probably the person's body. Two body parts have almost the same x and z, a different y and a similar Doppler. The control cannot remove them.

## Decision

Stop. Do not build the filter or the room-size estimate. Reasons:

- No stable floor or wall peak in the points, in any recording.
- The radar gives 0 to 16 points per frame, has clutter removal on, and has a coarse sideways resolution (3 TX, 4 RX).
- The only hint of structure has a simple explanation that is not multipath.
- We cannot check a filter without ghost labels or a measured room size.

## Not tried

A test on the heatmap. It keeps all range bins, so a floor bounce would show as a copy of the person at a longer range. It needs a new spike. Do it only if someone needs the room size.

## Notes

- The spikes ran on `haukurs_pc` with the points of the 3 recordings exported from the baked dataset. The exported points are not in the repo.
- The figures are in `results/ghost_spike/` (not tracked). The notebooks hold them as outputs.
