# Controlled XYZ noise (RV-016)

```python
from raven.perception.preprocessing.noise import inject_noise
noisy = inject_noise(cloud, position_std=0.05, seed=42)
```

This NumPy-only function returns a new PointCloud with independent, read-only
float32 storage. It adds independent Gaussian samples to X, Y and Z of each row:

`XYZ_noisy = float64(XYZ_original) + Normal(mean=0, variance=position_std**2)`

`position_std` is the **per-axis standard deviation**, not variance or mean
Euclidean displacement. Units follow the caller's coordinates; for KITTI
Velodyne they are meters (X forward, Y left, Z up). There is no transform,
resampling, dropout, intensity noise, fake object creation or implicit filtering.
Every intensity value is preserved exactly, including signed zero. Point count,
row correspondence/order, frame ID and coordinate frame remain unchanged.

## Parameters and random-number policy

- `position_std`: one finite, nonnegative real scalar convertible to float64.
  Python and NumPy real numeric scalars are supported. Per-axis arrays are not
  supported. Zero (including negative zero) gives an exact independent copy.
  Nonzero values that convert to float64 zero are rejected rather than silently
  becoming a zero-noise request.
- `seed`: a required nonnegative integer (including NumPy integer scalars).
  There is no default seed or implicit entropy source. Negative seeds, floats,
  strings, booleans, arrays, None and existing generator instances are rejected.
  Arbitrary-size nonnegative Python integers are accepted by PCG64 seed
  initialization. Booleans are also rejected for `position_std`.
- Invalid types raise TypeError; invalid numeric values raise ValueError.
  Both parameters are validated even for zero noise or an empty cloud.

Each nontrivial call constructs a fresh local `np.random.Generator(np.random.PCG64(seed))`
and draws `normal(loc=0.0, scale=position_std, size=(N, 3))` once, in row/XYZ order.
NumPy's [normal API](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html)
defines `scale` as standard deviation. No global RNG state or unrelated generator
is reset or advanced. Empty inputs return an independent `(0, 4)` cloud; singleton
inputs receive three samples. Empty/zero-noise calls need no random draw.

Same input rows, parameters, seed and software environment reproduce the output.
Different seeds normally give different nonzero-noise XYZ, but float32 rounding
may erase very small perturbations. Reordering or subsetting a cloud changes
which samples are assigned to particular returns. Reusing the same seed restarts
the stream on every call; use distinct seeds for separate experiment repetitions.
Applying the function twice to an already noisy cloud applies noise twice.

PCG64 is explicit rather than relying on the default bit generator chosen by
`default_rng`. This does not promise bitwise Gaussian samples across all versions,
builds, CPUs or platforms. NumPy's
[RNG compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html)
ties reproducibility to the generator, seed, call arguments and environment.
Record NumPy/Python versions, environment, seed, operation order and input hash
for controlled experiments. A changed draw shape or distribution implementation
can change the stream even if the seed is unchanged.

## Numerical behavior and resource cost

Coordinates are promoted from stored float32 to float64 before sample addition.
The complete result must be finite, and every noisy XYZ must lie within
`[-finfo(float32).max, +finfo(float32).max]` before conversion. Otherwise the call
raises ValueError without returning a partial result or modifying the source.
This includes overflowing float64 samples/sums. No clipping or redraw is used.
Gaussian noise is unbounded, so even finite parameters can fail this check for
extreme inputs/scales. Zero noise remains valid at the float32 extrema.

Float64 computation cannot restore precision already lost in input storage.
Final float32 rounding (including subnormal underflow to zero) is allowed; it may
remove a tiny displacement or alter its measured magnitude. Retained intensities
make an exact float32-to-float64-to-float32 round trip and are not perturbed.
The operation uses vectorized O(N) time and O(N) extra memory for samples, working
arrays, validation and output; it creates no pairwise matrix or neighbor index.
There are no new dependencies or GPU/visualization imports, and no new CLI flag.

## Example with your KITTI scan

```python
import numpy as np
from raven.perception.data.kitti_loader import load_kitti_bin
from raven.perception.preprocessing.noise import inject_noise

cloud = load_kitti_bin('/path/to/your/kitti/training/velodyne/000000.bin')
for std in (0.01, 0.05, 0.10):
    noisy = inject_noise(cloud, position_std=std, seed=42)
    delta = noisy.points[:, :3].astype(np.float64) - cloud.points[:, :3].astype(np.float64)
    print(std, cloud.count, noisy.count, delta.mean(axis=0), delta.std(axis=0))
    assert noisy.points.tobytes() == inject_noise(cloud, std, 42).points.tobytes()
    assert noisy.points[:, 3].tobytes() == cloud.points[:, 3].tobytes()
```

Read-only smoke test on 2026-11-14: frame 000000, **115,384 input and output points**
for each configuration; seed 42, Python 3.11.12, NumPy 2.4.6, PCG64,
macOS 26.6.2 arm64. Input SHA256:
`0e09c85e3f6078ecbdd1e706ee9624519f1bd29417437167a9ed7fbe6f54b4b1`.
The following are observed statistics of displacement **after float32 storage**,
in meters; standard deviations use ddof=0.

| Requested std | Axis means (X, Y, Z) | Axis std (X, Y, Z) | Mean Euclidean norm | RMS Euclidean norm | Maximum norm |
| --- | --- | --- | --- | --- | --- |
| 0.01 | (-0.00001663, -0.00002788, 0.00004643) | (0.01003552, 0.00998055, 0.00999503) | 0.01595924 | 0.01732706 | 0.05695984 |
| 0.05 | (-0.00008317, -0.00013941, 0.00023217) | (0.05017762, 0.04990274, 0.04997515) | 0.07979620 | 0.08663528 | 0.28479941 |
| 0.10 | (-0.00016634, -0.00027881, 0.00046433) | (0.10035523, 0.09980548, 0.09995030) | 0.15959241 | 0.17327057 | 0.56959863 |

Repeated calls were bitwise identical; intensity, input array and file SHA256
were unchanged. No source scan or generated degraded scan is committed. These
observations establish controlled perturbation and reproducibility only, without
measuring detection accuracy or latency.

## Composition and sensor-model limitations

This is a simplified isotropic Cartesian error model, not a physics-based LiDAR
simulator. It does not model range/angle dependence, correlations, occlusion,
beam patterns, material reflectance, weather or sensor calibration errors.
Independent draws do not imply real LiDAR errors are independent or Gaussian.
No detection-quality or performance effect follows from the model alone.

The caller can explicitly combine [range filtering](range-filtering.md),
[voxel downsampling](voxel-downsampling.md),
[statistical outlier removal](statistical-outlier-removal.md) and noise. Stage order
matters: noise can move returns across range/voxel boundaries or change neighbor
statistics; averaging can alter the noise distribution. No previous stage is
repeated automatically. RV-017 will provide configurable pipeline composition;
this issue adds only the individual noise operation.

## Validation

`tests/test_noise.py` covers repeatability, different seeds, exact PCG64/normal
parameterization, independent-axis sample statistics, zero noise, empty/singleton
clouds, intensity bits and row correspondence, metadata, independent storage,
input immutability, untouched RNG state, invalid parameters/seeds, extreme scales
and float32 overflow/rounding, explicit stage composition, and execution without
SciPy/GPU/Open3D/OpenPCDet imports. Tests need no KITTI dataset.

```bash
python -m pytest -q
python -m ruff check .
python -m pip check
git diff --check
```
