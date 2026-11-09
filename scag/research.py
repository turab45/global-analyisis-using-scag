"""Patient-level validation, uncertainty and provenance for research runs."""

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from torch.utils.data import DataLoader

from .relations import adjacency


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifests(manifests):
    for split, frame in manifests.items():
        if len(frame) < 2 or frame.pid.duplicated().any() or not frame.pid.astype(str).str.len().gt(0).all():
            raise ValueError(f"{split}: require at least two distinct patient IDs.")
    for left, right in combinations(manifests, 2):
        if set(manifests[left].pid) & set(manifests[right].pid):
            raise ValueError(f"Patient overlap between {left} and {right}.")


def cache_patients(dataset, manifest, batch_size):
    """Verify actual loaded IDs against the manifest before caching immutable samples."""
    samples, observed, slices = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        images, segments, metadata = batch
        if "pid" not in metadata:
            raise ValueError("Loaded samples must expose patient IDs.")
        observed.extend(str(pid) for pid in metadata["pid"])
        if "slice_index" in metadata:
            slices.extend(int(index) for index in metadata["slice_index"])
        samples.extend((images[i].clone(), segments[i].clone()) for i in range(len(images)))
    if observed != manifest.pid.astype(str).tolist():
        raise ValueError("Loaded patient order differs from manifest; refusing mislabelled evidence.")
    if slices:
        manifest["slice_index"] = slices
    return DataLoader(samples, batch_size=batch_size, shuffle=False)


def bootstrap_relations(pooled, S, names, tau, n_bootstrap=200, seed=42):
    """Resample patients, recompute Spearman/thresholding; condition on training S."""
    rng = np.random.default_rng(seed)
    upper = np.triu_indices(len(names))
    null = np.empty((n_bootstrap, len(upper[0])))
    for b in range(n_bootstrap):
        sample = pooled[rng.integers(len(pooled), size=len(pooled))]
        ranks = rankdata(sample, axis=0)
        ranks -= ranks.mean(axis=0)
        norms = np.linalg.norm(ranks, axis=0)
        ranks = np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > 0)
        A = adjacency(np.clip(ranks.T @ ranks, -1, 1), S, tau)
        null[b] = (S.T @ A @ S)[upper]
    lo, hi = np.quantile(null, [.025, .975], axis=0)
    return pd.DataFrame({"concept_c": np.array(names)[upper[0]], "concept_d": np.array(names)[upper[1]], "weight_ci_low": lo, "weight_ci_high": hi, "bootstrap_patients": len(pooled), "bootstrap_repeats": n_bootstrap})


def relation_stability(A_train, A_split, S_train, S_split):
    """Descriptive common-concept comparisons; no neuron-pair independence assumption."""
    upper = np.triu_indices(S_train.shape[1], 1)
    comparisons = {
        "joint_frozen_semantics": ((S_train.T @ A_train @ S_train)[upper], (S_train.T @ A_split @ S_train)[upper]),
        "joint_reestimated_semantics": ((S_train.T @ A_train @ S_train)[upper], (S_split.T @ A_split @ S_split)[upper]),
        "semantic_only": ((S_train.T @ S_train)[upper], (S_split.T @ S_split)[upper]),
    }
    rows = []
    for name, (train, evaluated) in comparisons.items():
        rho = spearmanr(train, evaluated).statistic if len(train) >= 3 and np.ptp(train) > 0 and np.ptp(evaluated) > 0 else np.nan
        rows.append({"representation": name, "n_concept_pairs": len(train), "train_split_spearman": rho})
    return pd.DataFrame(rows)


def concept_evidence(profiles, S_train, names, manifest, top=3):
    rows = []
    per_patient = profiles["per_patient_iou"]
    for neuron in range(per_patient.shape[1]):
        for c, name in enumerate(names):
            order = np.argsort(-per_patient[:, neuron, c], kind="stable")[:top]
            for rank, sample in enumerate(order):
                if per_patient[sample, neuron, c] <= 0:
                    continue
                rows.append({"neuron": neuron, "concept": name, "rank": rank + 1, "sample_index": int(sample), "training_iou": S_train[neuron, c], "patient_iou": per_patient[sample, neuron, c], "intersection": profiles["intersection"][sample, neuron, c], "union": profiles["union"][sample, neuron, c]})
    columns = ["neuron", "concept", "rank", "sample_index", "training_iou", "patient_iou", "intersection", "union"]
    return pd.DataFrame(rows, columns=columns).merge(manifest, on="sample_index", validate="many_to_one")
