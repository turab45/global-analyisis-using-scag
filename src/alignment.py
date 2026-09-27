"""Correlation R, semantic similarity Q, and their global alignment."""

import numpy as np
from scipy.stats import rankdata

from .coactivation import compute_correlation as compute_correlation
from .utils import finite_matrix, permutation_summary, symmetric_matrix

compute_correaltion = compute_correlation


def compute_semantic_similarity(profiles, mode="cosine"):
    """Compute Q as cosine similarity between neuron–concept profile rows.

    Zero profiles have zero similarity. Optional centered/IDF modes retain the
    sensitivity analyses used in the existing notebooks; cosine is the default.
    """
    values = finite_matrix(profiles, "profiles").copy()
    if mode == "centered":
        if len(values):
            values -= values.mean(axis=0, keepdims=True)
    elif mode == "idf":
        if len(values):
            frequency = np.count_nonzero(values > 0, axis=0)
            values *= np.log(len(values) / np.maximum(frequency, 1))
    elif mode != "cosine":
        raise ValueError("mode must be cosine, centered, or idf")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
    return np.clip(normalized @ normalized.T, -1, 1)


def compute_alignment(R, Q, *, n_permutations=1000, seed=7, positive_only=True, method="spearman"):
    """Correlate matching upper-triangle entries of R and Q.

    R and Q must describe the same neurons in the same order. Permutations
    relabel Q's rows and columns together, rather than shuffling individual edges.
    The one-sided p-value uses the notebook's upper-tail, plus-one correction.
    """
    R, Q = symmetric_matrix(R, "R"), symmetric_matrix(Q, "Q")
    if R.shape != Q.shape:
        raise ValueError("R and Q must have the same shape and neuron order")
    if type(n_permutations) is not int or n_permutations < 0:
        raise ValueError("n_permutations must be a nonnegative integer")
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be spearman or pearson")
    upper = np.triu_indices(len(R), 1)
    x = np.maximum(R[upper], 0) if positive_only else R[upper].copy()
    y = Q[upper].copy()
    result = {
        "r": np.nan,
        "p": np.nan,
        "z": np.nan,
        "n_nodes": len(R),
        "n_pairs": len(x),
        "n_permutations": n_permutations,
        "null_mean": np.nan,
        "null_std": np.nan,
        "status": "insufficient_pairs",
    }
    if len(x) < 3:
        return result
    if method == "spearman":
        x, y = rankdata(x), rankdata(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        result["status"] = "constant_pair_values"
        return result
    observed = float(np.clip(x @ y / denominator, -1, 1))
    result.update(r=observed, status="ok")
    # Node permutations only reorder the off-diagonal values, so their ranks
    # and norms can be computed once, including ties.
    ranked_Q = np.zeros_like(Q)
    ranked_Q[upper] = y
    ranked_Q[(upper[1], upper[0])] = y
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for index in range(n_permutations):
        order = rng.permutation(len(Q))
        permuted = ranked_Q[order[upper[0]], order[upper[1]]]
        null[index] = np.clip(x @ permuted / denominator, -1, 1)
    result.update(permutation_summary(observed, null))
    return result
