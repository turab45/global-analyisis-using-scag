"""Neuron correlations and positive weighted coactivation networks."""

import numpy as np
from scipy.stats import rankdata

from .utils import finite_matrix, symmetric_matrix


def compute_correlation(activations, method="spearman"):
    """Return R (neurons × neurons) from patient × neuron responses.

    Spatial dimensions, if present, are globally averaged first. Constant units
    have zero correlation, including when they occur in the first two columns.
    The diagonal is zero because self-edges are excluded from SCAG.
    """
    values = np.asarray(activations, dtype=np.float64)
    if values.ndim > 2:
        values = values.mean(axis=tuple(range(2, values.ndim)))
    values = finite_matrix(values, "activations")
    if len(values) < 2 or values.shape[1] == 0:
        raise ValueError("Correlation requires at least two patients and one neuron")
    if method == "spearman":
        values = rankdata(values, axis=0)
    elif method != "pearson":
        raise ValueError("method must be spearman or pearson")
    centered = values - values.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    result = np.clip(normalized.T @ normalized, -1, 1)
    np.fill_diagonal(result, 0)
    return result


def threshold_correlations(correlation, *, threshold=None, density=None):
    """Select positive undirected edges by cutoff or exact requested density.

    Density counts all possible pairs. If too few positive edges exist, retain
    those available. Equal weights are resolved by neuron indices, deterministically.
    """
    matrix = symmetric_matrix(correlation, "correlation")
    if (threshold is None) == (density is None):
        raise ValueError("Specify exactly one of threshold or density")
    upper = np.triu_indices(len(matrix), 1)
    weights = matrix[upper]
    if threshold is not None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        selected = np.flatnonzero((weights > 0) & (weights >= threshold))
    else:
        if not 0 <= density <= 1:
            raise ValueError("density must lie in [0, 1]")
        positive = np.flatnonzero(weights > 0)
        order = np.argsort(-weights[positive], kind="stable")
        selected = positive[order[: round(float(density) * len(weights))]]
    result = np.zeros_like(matrix)
    rows, columns = upper[0][selected], upper[1][selected]
    result[rows, columns] = weights[selected]
    result[columns, rows] = weights[selected]
    return result
