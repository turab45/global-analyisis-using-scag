"""Concept-associated coactivation analyses; edges are associations, not causal flow."""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom

from .matrices import compute_Q


def adjacency(R, S, tau):
    R, S = np.asarray(R, dtype=float), np.asarray(S, dtype=float)
    if S.ndim != 2 or not S.size or not np.isfinite(S).all() or (S < 0).any():
        raise ValueError("S must be a finite nonnegative neuron-by-concept matrix.")
    if R.shape != (len(S), len(S)) or not np.isfinite(R).all() or not np.allclose(R, R.T) or (np.abs(R) > 1 + 1e-6).any():
        raise ValueError("R must be a finite symmetric correlation matrix matching S.")
    if not np.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("tau must be in [0, 1].")
    # Match the existing SCAG >= threshold convention, with no self-links.
    A = np.where((R >= tau) & (R > 0), R, 0)
    np.fill_diagonal(A, 0)
    return A


def fdr_bh(p):
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        q[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return q


def null_summary(observed, null):
    mean, sd = null.mean(axis=0), null.std(axis=0)
    delta = observed - mean
    z = np.divide(delta, sd, out=np.full_like(delta, np.nan, dtype=float), where=sd > 1e-12)
    p = (1 + (null >= observed - 1e-12).sum(axis=0)) / (len(null) + 1)
    return mean, delta, z, p


def concept_relations(A, S, names, n_perm=500, seed=42, rewire=False):
    """Whole-profile permutation and degree-stratified sensitivity nulls."""
    if n_perm < 1 or len(names) != S.shape[1]:
        raise ValueError("Positive n_perm and one name per concept are required.")
    sparse = csr_matrix(A)
    W, semantic = S.T @ (sparse @ S), S.T @ S
    upper = np.triu_indices(S.shape[1])
    rows = pd.DataFrame({"concept_c": np.array(names)[upper[0]], "concept_d": np.array(names)[upper[1]], "weight": W[upper], "semantic_only": semantic[upper]})
    rows["weight_per_edge_mass"] = W[upper] / A.sum() if A.sum() else np.nan
    degree = (A > 0).sum(axis=1)
    # Equal-degree neurons stay together; singleton strata honestly yield degenerate nulls.
    strata = np.searchsorted(np.unique(np.quantile(degree, [0.25, 0.5, 0.75])), degree, side="right")
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    ii, jj = np.triu_indices(len(S), 1)
    edge_weights = A[ii, jj][A[ii, jj] > 0]
    rng = np.random.default_rng(seed)
    for kind in (("profile", "degree", "rewire") if rewire else ("profile", "degree")):
        null = np.empty((n_perm, len(upper[0])))
        for b in range(n_perm):
            if kind == "rewire":
                # Uniform edge-position null preserves edge count and weight multiset, not degrees.
                chosen = rng.choice(len(ii), len(edge_weights), replace=False)
                u, v = ii[chosen], jj[chosen]
                weights = rng.permutation(edge_weights)
                rewired = csr_matrix((np.tile(weights, 2), (np.r_[u, v], np.r_[v, u])), shape=A.shape)
                null[b] = (S.T @ (rewired @ S))[upper]
                continue
            permutation = rng.permutation(len(S)) if kind == "profile" else np.arange(len(S))
            if kind == "degree":
                for group in groups:
                    permutation[group] = rng.permutation(group)
            shuffled = S[permutation]
            null[b] = (shuffled.T @ (sparse @ shuffled))[upper]
        mean, delta, z, p = null_summary(W[upper], null)
        for field, values in zip(("null_mean", "delta", "z", "p"), (mean, delta, z, p)):
            rows[f"{kind}_{field}"] = values
        rows[f"{kind}_q"] = fdr_bh(p)
    return rows


def concept_subgraph(A, S, names, concept, gamma=0.015, neighbors=3):
    """Retrieve all seed-seed edges and top external neighbours per concept seed."""
    if concept not in names or not np.isfinite(gamma) or gamma <= 0 or neighbors < 0:
        raise ValueError("Require a known concept, positive gamma and nonnegative neighbors.")
    c = names.index(concept)
    seeds = set(np.flatnonzero(S[:, c] >= gamma).tolist())
    edges = set()
    for i in sorted(seeds):
        linked = np.flatnonzero(A[i] > 0)
        edges.update(tuple(sorted((i, int(j)))) for j in linked if j in seeds)
        external = [j for j in linked if j not in seeds]
        external = sorted(external, key=lambda j: (-A[i, j], j))[:neighbors]
        edges.update(tuple(sorted((i, int(j)))) for j in external)
    nodes = sorted(seeds | {node for edge in edges for node in edge})
    node_rows = [{"neuron": i, "is_seed": i in seeds, "query_score": S[i, c], **{name: S[i, j] for j, name in enumerate(names)}} for i in nodes]
    edge_rows = [{"source": i, "target": j, "coactivation": A[i, j]} for i, j in sorted(edges)]
    return pd.DataFrame(node_rows, columns=["neuron", "is_seed", "query_score", *names]), pd.DataFrame(edge_rows, columns=["source", "target", "coactivation"])


def semantic_functional_pairs(R, S, tau=0.35, similarity=0.8, n_perm=500, seed=42):
    """Signed R versus cosine Q; zero-profile neurons have undefined semantics."""
    if not 0 <= similarity <= 1 or n_perm < 1:
        raise ValueError("Require similarity in [0, 1] and positive n_perm.")
    nodes = np.flatnonzero(np.linalg.norm(S, axis=1) > 0)
    Q = compute_Q(S[nodes])
    i, j = np.triu_indices(len(nodes), 1)
    r, q = R[np.ix_(nodes, nodes)][i, j], Q[i, j]
    categories = np.array(["low_semantic_low_coactivation", "low_semantic_high_coactivation", "high_semantic_low_coactivation", "high_semantic_high_coactivation"])
    code = 2 * (q >= similarity).astype(int) + (r >= tau)
    pairs = pd.DataFrame({"source": nodes[i], "target": nodes[j], "R": r, "Q": q, "category": categories[code]})
    observed = np.bincount(code, minlength=4)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(nodes))
        shuffled = Q[perm[i], perm[j]]
        null.append(np.bincount(2 * (shuffled >= similarity).astype(int) + (r >= tau), minlength=4))
    mean, delta, z, p = null_summary(observed, np.asarray(null))
    summary = pd.DataFrame({"category": categories, "count": observed, "null_mean": mean, "delta": delta, "z": z, "p": p, "q": fdr_bh(p)})
    return pairs, summary


def community_organisation(A, S, names, gamma=0.015, n_perm=500, seed=42, partition=None):
    """Multi-label enrichment and SVE on nonisolated, nonzero-profile neurons."""
    graph = nx.from_numpy_array(A)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if partition is None:
        communities = nx.community.louvain_communities(graph, seed=seed, weight="weight") if graph.number_of_edges() else []
        membership = {node: c for c, community in enumerate(communities) for node in community}
    else:
        membership = dict(zip(partition.neuron.astype(int), partition.community.astype(int)))
    nodes = np.array(sorted(n for n in membership if S[n].sum() > 0), dtype=int)
    labels = np.array([membership[n] for n in nodes], dtype=int)
    profiles = S[nodes]
    present = profiles >= gamma
    rows = []
    for c in np.unique(labels):
        mask = labels == c
        for concept, name in enumerate(names):
            k, total = int(present[mask, concept].sum()), int(present[:, concept].sum())
            rows.append({"community": c, "concept": name, "size": int(mask.sum()), "count": k, "population": len(nodes), "population_count": total, "p": hypergeom.sf(k - 1, len(nodes), total, int(mask.sum()))})
    enrichment = pd.DataFrame(rows, columns=["community", "concept", "size", "count", "population", "population_count", "p"])
    enrichment["q"] = fdr_bh(enrichment.p.to_numpy())
    total_variance = ((profiles - profiles.mean(axis=0)) ** 2).sum() if len(nodes) else 0

    def sve(partition):
        within = sum(((profiles[partition == c] - profiles[partition == c].mean(axis=0)) ** 2).sum() for c in np.unique(partition))
        return 1 - within / total_variance

    stats = {"n_nodes": len(nodes), "n_communities": len(np.unique(labels)), "labelled_coverage": float((S.max(axis=1) >= gamma).mean()), "partition_coverage": len(nodes) / len(S), "sve": np.nan, "null_mean": np.nan, "delta": np.nan, "z": np.nan, "p": np.nan}
    if total_variance > 0:
        observed = sve(labels)
        rng = np.random.default_rng(seed)
        null = np.array([sve(rng.permutation(labels)) for _ in range(n_perm)])
        mean, delta, z, p = null_summary(observed, null)
        stats.update(dict(zip(("sve", "null_mean", "delta", "z", "p"), (observed, mean, delta, z, p))))
    partition = pd.DataFrame({"neuron": list(membership), "community": list(membership.values())})
    return enrichment, pd.DataFrame([stats]), partition


def concept_rankings(A, S, concept):
    scores = {"functional": A.sum(axis=1), "semantic": S[:, concept], "joint": S[:, concept] * (A @ S[:, concept])}
    return {name: np.argsort(-values, kind="stable") for name, values in scores.items()}, scores
