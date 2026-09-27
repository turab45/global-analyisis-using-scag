"""NetworkX construction, community detection, enrichment, and stability."""

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import adjusted_mutual_info_score

from .utils import finite_matrix, symmetric_matrix


def build_graph(weights, profiles, labeled, *, drop_isolates=True):
    weights = symmetric_matrix(weights, "weights")
    profiles = finite_matrix(profiles, "profiles")
    labeled = np.asarray(labeled, dtype=bool)
    if len(profiles) != len(weights) or labeled.shape != (len(weights),) or not profiles.shape[1]:
        raise ValueError("Weights, profiles, and labeled mask must describe the same neurons")
    if (weights < 0).any():
        raise ValueError("SCAG community analysis requires nonnegative edge weights")
    graph = nx.Graph()
    for neuron, profile in enumerate(profiles):
        attributes = {"semantic_score": float(profile.max()), "labeled": bool(labeled[neuron])}
        if labeled[neuron]:
            attributes["concept"] = int(profile.argmax())
        graph.add_node(neuron, **attributes)
    rows, cols = np.nonzero(np.triu(weights, 1))
    graph.add_weighted_edges_from((int(i), int(j), float(weights[i, j])) for i, j in zip(rows, cols))
    isolates = list(nx.isolates(graph))
    if drop_isolates:
        graph.remove_nodes_from(isolates)
    graph.graph.update(n_neurons=len(weights), n_isolates=len(isolates), n_graph=graph.number_of_nodes())
    return graph


def detect_communities(graph, *, seed=42, resolution=1.0):
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive")
    if graph.number_of_edges() == 0:
        return [{node} for node in sorted(graph)]
    return sorted(
        nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed),
        key=lambda nodes: (-len(nodes), min(nodes)),
    )


def graph_summary(graph, communities, *, resolution=1.0):
    return {
        "n_graph": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_neurons": graph.graph.get("n_neurons", graph.number_of_nodes()),
        "n_isolates": graph.graph.get("n_isolates", 0),
        "density": nx.density(graph),
        "n_components": nx.number_connected_components(graph),
        "n_communities": len(communities),
        "modularity": nx.community.modularity(graph, communities, weight="weight", resolution=resolution)
        if graph.number_of_edges()
        else np.nan,
    }


def community_enrichment(graph, communities, concept_names, *, min_labeled=3):
    """Hypergeometric enrichment, with BH correction across all reported tests."""
    labeled = {node: graph.nodes[node]["concept"] for node in graph if "concept" in graph.nodes[node]}
    labels = np.array(list(labeled.values()), dtype=int)
    rows = []
    for index, community in enumerate(communities):
        local = [labeled[node] for node in community if node in labeled]
        if len(local) < min_labeled:
            continue
        for concept, name in enumerate(concept_names):
            total = int(np.count_nonzero(labels == concept))
            if not total:
                continue
            observed = local.count(concept)
            rows.append(
                {
                    "community": index,
                    "concept": concept,
                    "name": name,
                    "count": observed,
                    "n_labeled": len(local),
                    "p": float(hypergeom.sf(observed - 1, len(labels), total, len(local))),
                }
            )
    frame = pd.DataFrame(rows, columns=["community", "concept", "name", "count", "n_labeled", "p"])
    if len(frame):
        order = np.argsort(frame.p.to_numpy())
        adjusted = frame.p.to_numpy()[order] * len(frame) / np.arange(1, len(frame) + 1)
        corrected = np.empty(len(frame))
        corrected[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
        frame["p_adjusted"] = corrected
    else:
        frame["p_adjusted"] = pd.Series(dtype=float)
    return frame


def community_stability(graph, *, seeds=range(10), resolution=1.0):
    """Pairwise partition AMI across Louvain seeds on the same neuron set."""
    nodes = sorted(graph)
    seeds = list(seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Community seeds must be distinct")
    partitions = []
    for seed in seeds:
        communities = detect_communities(graph, seed=int(seed), resolution=resolution)
        assignment = {node: index for index, group in enumerate(communities) for node in group}
        partitions.append([assignment[node] for node in nodes])
    rows = [
        {
            "seed_a": seeds[i],
            "seed_b": seeds[j],
            "ami": float(adjusted_mutual_info_score(partitions[i], partitions[j])) if len(nodes) > 1 else np.nan,
        }
        for i, j in combinations(range(len(seeds)), 2)
    ]
    return pd.DataFrame(rows, columns=["seed_a", "seed_b", "ami"])
