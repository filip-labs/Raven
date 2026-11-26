# Pinned upstream sources

The unmodified eval.py and rotate_iou.py files come from
pcdet/datasets/kitti/kitti_object_eval_python/ in open-mmlab/OpenPCDet revision
233f849829b6ac19afb8af8837a0246890908755. LICENSE.evaluator is the directory's
MIT license; rotate_iou.py preserves its original author/RRPN attribution.
box_utils.py comes from pcdet/utils/ at the same revision and is covered by the
project's Apache-2.0 LICENSE.OpenPCDet. provenance.json records exact source hashes.
Ruff excludes this directory; .gitattributes permits the upstream trailing spaces
so reference files and licenses remain byte-for-byte unchanged.

Only reviewed function ASTs from eval.py and three box_utils.py conversion functions
are loaded. Imports/decorators are omitted, leaving AP/ignore/matching and conversion
bodies unchanged. No numba, torch, scipy or CUDA code is imported. rotate_iou.py
is retained for audit/reference provenance, NOT executed on CPU. Direct CPU execution
of its device helpers failed coincident-rotated-box tests and was rejected.

The GPU overlap launch is replaced with pinned Shapely/GEOS float64 oriented
polygon intersection and the original vertical-overlap function. There is no
axis-aligned/raster IoU approximation. This geometry dependency is recorded in
results; it is not claimed CUDA-bitwise-equivalent. See docs/kitti-evaluation.md
for independent fixtures, licenses, convention limitations and non-certification
of official KITTI server equivalence.
