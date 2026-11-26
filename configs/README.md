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

Experiment schema and execution are documented in the configuration/runner guides. `kitti_splits/`
contains the previously documented split lists and their notices.

`experiments/modal-pointpillars.json` is the explicit schema-v2 optional Modal
example with placeholder dataset/output paths and a one-frame spending guard.
See [setup and authorization requirements](../docs/modal-inference.md); parsing
this file does not authorize or execute cloud work.

`benchmarks/cpu.json` and `benchmarks/local-cuda.json` are bounded schema-v1 timing
specifications for `raven benchmark-experiment`. They reuse the selected experiment
pipeline; no new preprocessing format is introduced. The CPU example experiment
is `experiments/preprocessing-benchmark.json` with placeholder paths. Read the
[measurement/loading/copy and GPU availability rules](../docs/latency-benchmarking.md)
before comparing results.
