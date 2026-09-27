"""Array validation and portable analysis output."""

import json
from pathlib import Path

import numpy as np


def finite_matrix(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 2-D matrix")
    return matrix


def symmetric_matrix(value, name):
    matrix = finite_matrix(value, name)
    if matrix.shape[0] != matrix.shape[1] or not np.allclose(matrix, matrix.T, atol=1e-7):
        raise ValueError(f"{name} must be square and symmetric")
    return matrix


def save_json(path, value):
    """Write undefined statistics as JSON null, rather than nonstandard NaN."""

    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(val) for val in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if isinstance(item, Path):
            return str(item)
        return item

    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def permutation_summary(observed, null):
    values = np.asarray(null, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or not len(values):
        return {"p": np.nan, "z": np.nan, "null_mean": np.nan, "null_std": np.nan}
    std = float(values.std())
    return {
        "p": float((1 + np.count_nonzero(values >= observed)) / (len(values) + 1)),
        "z": float((observed - values.mean()) / std) if std > 0 else np.nan,
        "null_mean": float(values.mean()),
        "null_std": std,
    }
