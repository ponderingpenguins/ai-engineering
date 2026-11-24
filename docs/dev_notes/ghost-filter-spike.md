# Ghost-point filter and room-size estimate: abandoned

## Idea

Radar points can be reflections (ghosts) of a real point. We wanted to find the reflecting planes (floor, walls) from the points, remove the ghosts before training, and estimate the room size.

## Spikes

Both ran on 3 `room1` recordings from 3 sessions (`20231115_135843`, `20231120_144541`, `20231120_153152`). Each compares point pairs inside one frame with pairs from frames far apart in time. A real reflection exists only inside one frame.

1. **Bisector planes of point pairs** (`scripts/ghost_plane_spike.py`, `notebooks/spike_ghost_planes.ipynb`). No floor or wall peak.
2. **Floor ghost straight behind a real point** (`scripts/floor_ghost_spike.py`, `notebooks/spike_floor_ghost.ipynb`). The floor estimate is at a different distance in each recording. The one weak excess is most likely the person's own body.

## Decision

Stop. We did not build the filter or the room-size estimate.

- The radar gives 0 to 16 points per frame, has clutter removal on, and has a coarse sideways resolution (3 TX, 4 RX). The points do not hold enough to find reflecting planes.
- We have no ghost labels and no measured room size, so we could not check a filter.

## Not tried

A test on the heatmap. A floor bounce would show as a copy of the person at a longer range. Only worth it if someone needs the room size.

## Notes

- The spikes ran on `haukurs_pc` with points exported from the baked dataset. The export is not in the repo.
- The figures are in `results/ghost_spike/` (not tracked). The notebooks hold them as outputs.
