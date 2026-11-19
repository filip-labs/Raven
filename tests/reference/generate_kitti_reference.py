"""Independent upstream execution with analytic axis-aligned overlaps, not Raven's backend.

Usage: python tests/reference/generate_kitti_reference.py NEW_OUTPUT.json
Only controlled synthetic boxes are supported. No GPU, numba, Shapely or data downloads.
"""

import ast
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REVISION = "233f849829b6ac19afb8af8837a0246890908755"
SHA256 = "d9260a41f68083d685fc43fecac1c417b14a1bc53df4915a46b2924f3a6a3c64"


def annotation(score=None, cls="Car", x=0.0):
    d = {
        "name": np.array([cls]),
        "bbox": np.array([[0.0, 0.0, 100.0, 100.0]]),
        "dimensions": np.array([[4.0, 2.0, 2.0]]),
        "location": np.array([[x, 1.0, 20.0]]),
        "rotation_y": np.array([0.0]),
        "alpha": np.array([0.0]),
    }
    if score is None:
        d.update(occluded=np.array([0]), truncated=np.array([0.0]))
    else:
        d["score"] = np.array([score])
    return d


def analytic_parts(first, second, metric, num_parts):
    if metric != 2 or num_parts != len(first):
        raise ValueError("fixture oracle supports only per-frame axis-aligned 3D")
    matrices = []
    for a, b in zip(first, second, strict=True):
        if np.any(a["rotation_y"] != 0) or np.any(b["rotation_y"] != 0):
            raise ValueError("fixture geometry must be axis aligned")
        matrix = np.zeros((len(a["name"]), len(b["name"])))
        for i, (pa, da) in enumerate(zip(a["location"], a["dimensions"], strict=True)):
            for j, (pb, db) in enumerate(zip(b["location"], b["dimensions"], strict=True)):
                # Closed-form intersection of [x+-l/2, y-h..y, z+-w/2].
                low_a = pa - np.array([da[0] / 2, da[1], da[2] / 2])
                high_a = pa + [da[0] / 2, 0, da[2] / 2]
                low_b = pb - np.array([db[0] / 2, db[1], db[2] / 2])
                high_b = pb + [db[0] / 2, 0, db[2] / 2]
                intersection = np.maximum(
                    np.minimum(high_a, high_b) - np.maximum(low_a, low_b), 0
                ).prod()
                matrix[i, j] = intersection / (da.prod() + db.prod() - intersection)
        matrices.append(matrix)
    return (
        matrices,
        matrices,
        np.array([len(a["name"]) for a in first]),
        np.array([len(b["name"]) for b in second]),
    )


def generate():
    source = Path(__file__).resolve().parents[2] / "src/raven/evaluation/_vendor/eval.py"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("reference source changed")
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for node in nodes:
        node.decorator_list = []
    namespace = {"np": np, "sysio": io}
    exec(  # noqa: S102 -- Only bundled checksum-pinned upstream function ASTs.
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"
        ),
        namespace,
    )
    namespace["calculate_iou_partly"] = analytic_parts
    values = {}
    for case in (
        "perfect",
        "high_score_false_positive_each_frame",
        "wrong_class",
        "empty_predictions",
    ):
        gt = [annotation() for _ in range(41)]
        dt = [annotation(0.9) for _ in gt]
        if case == "high_score_false_positive_each_frame":
            dt = [
                {k: np.concatenate([row[k], annotation(0.95, x=100)[k]]) for k in row} for row in dt
            ]
        elif case == "wrong_class":
            dt = [annotation(0.9, cls="Cyclist") for _ in gt]
        elif case == "empty_predictions":
            dt = [{k: v[:0] for k, v in row.items()} for row in dt]
        result = namespace["eval_class"](
            gt, dt, [0], [0, 1, 2], 2, np.full((1, 3, 1), 0.7), num_parts=41
        )
        values[case] = namespace["get_mAP_R40"](result["precision"])[0, :, 0].tolist()
    return {
        "reference_revision": REVISION,
        "reference_eval_sha256": SHA256,
        "execution": "upstream Python bodies, decorators omitted; independent analytic 3D overlaps",
        "not_executed": ["CUDA", "Raven evaluator", "GEOS"],
        "frames": 41,
        "difficulties": ["easy", "moderate", "hard"],
        "car_ap_r40_percent": values,
    }


if __name__ == "__main__":
    with Path(sys.argv[1]).open("x", encoding="utf-8") as stream:
        json.dump(generate(), stream, indent=2)
        stream.write("\n")
