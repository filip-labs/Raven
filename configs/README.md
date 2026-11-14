# Experiment configs

`preprocessing/` contains ordered JSON arrays of Raven pipeline stage mappings:

- `range.json`: inclusive X [0,70], Y [-40,40], Z [-3,1] meter KITTI crop.
- `voxel.json`: 0.2 m voxel means.
- `full.json`: range → voxel → statistical filtering (K=16, ratio=1) → noise
  (std=0.01 m, seed=42).

These are explicit exploratory examples, not defaults or optimal detector settings.
They encode the existing [pipeline API](../docs/preprocessing-pipeline.md), with no
additional configuration wrapper. Use them with `raven visualize SCAN --compare CONFIG`.
See [comparison usage and controls](../docs/cloud-comparison.md).

Future experiment/evaluation configuration work remains separate. `kitti_splits/`
contains the previously documented split lists and their notices.
