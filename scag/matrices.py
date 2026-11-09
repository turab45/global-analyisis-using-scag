import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


def compute_R(activations):
    """Neuron-neuron coactivation matrix using Spearman correlation."""
    pooled = torch.mean(activations, dim=(2, 3)).numpy()
    if pooled.shape[0] < 2:
        raise ValueError("Coactivation requires at least two samples.")
    ranks = rankdata(pooled, axis=0)
    ranks -= ranks.mean(axis=0)
    norms = np.linalg.norm(ranks, axis=0)
    ranks = np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > 0)
    R = np.clip(ranks.T @ ranks, -1, 1)
    np.fill_diagonal(R, 0.0)
    return R


def compute_Q(X):
    """Neuron-neuron semantic similarity using cosine similarity."""
    Xn = X.astype(np.float64).copy()
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
    return Xn @ Xn.T


def align_R_Q(R, Q, n_perm=1000, seed=7):
    """Mantel-like Spearman alignment between R and Q."""
    R = np.clip(R, 0, None)
    iu = np.triu_indices(R.shape[0], 1)

    r = spearmanr(R[iu], Q[iu])[0]
    rng = np.random.default_rng(seed)

    null = np.nan_to_num(np.array([
        spearmanr(R[iu], Q[np.ix_(perm, perm)][iu])[0]
        for perm in (rng.permutation(R.shape[0]) for _ in range(n_perm))
    ]))

    z = (r - null.mean()) / (null.std() + 1e-12)
    p = (np.sum(null >= r) + 1) / (n_perm + 1)
    return r, z, p


def compute_edge_reliability(R_train, R_eval, lab_idx):
    train = R_train[np.ix_(lab_idx, lab_idx)]
    eval_ = R_eval[np.ix_(lab_idx, lab_idx)]
    iu = np.triu_indices(len(lab_idx), 1)
    return spearmanr(train[iu], eval_[iu])[0]
