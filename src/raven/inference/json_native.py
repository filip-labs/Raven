"""Exact JSON value types for RPC: no framework objects, implicit coercion or pickle hooks."""

import json
import math

MAX_DEPTH = 64


def _walk(value, path, active, depth, normalize):
    if depth > MAX_DEPTH:
        raise ValueError(f"{path}: JSON nesting exceeds {MAX_DEPTH}")
    kind = type(value)
    if kind in (type(None), bool, int, str):
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: JSON numbers must be finite")
        return value
    allowed = (dict, list, tuple) if normalize else (dict, list)
    if kind not in allowed:
        raise TypeError(f"{path}: expected exact JSON-native type, got {kind.__name__}")
    identity = id(value)
    if identity in active:
        raise ValueError(f"{path}: cyclic JSON container")
    active.add(identity)
    try:
        if kind is dict:
            result = {} if normalize else value
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path}: JSON object keys must be exact builtin str")
                child = _walk(item, f"{path}[{json.dumps(key)}]", active, depth + 1, normalize)
                if normalize:
                    result[key] = child
        else:
            result = [] if normalize else value
            for index, item in enumerate(value):
                child = _walk(item, f"{path}[{index}]", active, depth + 1, normalize)
                if normalize:
                    result.append(child)
        return result
    finally:
        active.remove(identity)


def validate_json_native(value):
    """Require exact None/bool/int/finite float/str/list/dict recursively; never coerce."""
    _walk(value, "$", set(), 0, False)


def normalize_json_containers(value):
    """Copy native containers and turn exact tuples into lists; reject every other type.

    This handles existing box/config tuple geometry without changing JSON meaning.
    Framework scalar/version normalization belongs at its explicit producer. No
    default=str, ndarray/tensor materialization or subclass pickle hook is used.
    """
    result = _walk(value, "$", set(), 0, True)
    validate_json_native(result)
    return result
