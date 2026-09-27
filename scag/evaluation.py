"""Global graph/semantic metrics with label-permutation nulls."""

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from .alignment import compute_semantic_similarity
from .utils import permutation_summary


def evaluate_semantics(graph, communities, profiles, *, n_permutations=500, seed=42):
    """Community purity, AMI, assortativity, and within/between semantic contrast.

    Hard-label permutations preserve concept frequencies and graph structure.
    Soft permutations preserve community sizes. Only originally labeled neurons
    are evaluated, and graph node IDs index the rows of profiles.
    """
    if type(n_permutations) is not int or n_permutations < 0:
        raise ValueError("n_permutations must be nonnegative")
    nodes = sorted(node for node in graph if "concept" in graph.nodes[node])
    metrics = ("purity", "ami", "assortativity", "soft_delta")
    result = {"n_labeled_graph": len(nodes)}
    if not nodes:
        for name in metrics:
            result.update({name: np.nan, f"{name}_p": np.nan, f"{name}_z": np.nan})
        return result
    assignment = {node: index for index, group in enumerate(communities) for node in group}
    partition = np.array([assignment[node] for node in nodes])
    labels = np.array([graph.nodes[node]["concept"] for node in nodes])
    indices = {node: index for index, node in enumerate(nodes)}
    edges = np.array([(indices[a], indices[b]) for a, b in graph.subgraph(nodes).edges()], dtype=int).reshape(-1, 2)
    groups = [np.flatnonzero(partition == value) for value in np.unique(partition)]

    def hard(values):
        purity = np.mean([np.unique(values[group], return_counts=True)[1].max() / len(group) for group in groups])
        ami = adjusted_mutual_info_score(values, partition) if len(values) > 1 else np.nan
        assortativity = np.nan
        if len(edges):
            endpoints = values[edges]
            proportions = np.bincount(endpoints.ravel()) / endpoints.size
            expected = np.square(proportions).sum()
            if expected < 1:
                assortativity = ((endpoints[:, 0] == endpoints[:, 1]).mean() - expected) / (1 - expected)
        return purity, ami, assortativity

    rng = np.random.default_rng(seed)
    observed = hard(labels)
    null = np.array([hard(rng.permutation(labels)) for _ in range(n_permutations)]).reshape(-1, 3)
    for index, name in enumerate(metrics[:3]):
        result[name] = float(observed[index])
        stats = permutation_summary(observed[index], null[:, index])
        result.update({f"{name}_{key}": stats[key] for key in ("p", "z")})
    similarity = compute_semantic_similarity(np.asarray(profiles)[nodes])
    upper = np.triu_indices(len(nodes), 1)
    values = similarity[upper]

    def soft(groups):
        same = groups[upper[0]] == groups[upper[1]]
        return float(values[same].mean() - values[~same].mean()) if same.any() and (~same).any() else np.nan

    result["soft_delta"] = soft(partition)
    soft_null = [soft(rng.permutation(partition)) for _ in range(n_permutations)]
    stats = permutation_summary(result["soft_delta"], soft_null)
    result.update(soft_delta_p=stats["p"], soft_delta_z=stats["z"])
    return result
