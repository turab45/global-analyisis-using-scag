# scag/graph.py

from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community as nx_comm
from scipy.stats import hypergeom, spearmanr
from sklearn.metrics import adjusted_mutual_info_score as ami_score

try:
    from statsmodels.stats.multitest import multipletests
    HAVE_SM = True
except ImportError:
    HAVE_SM = False


def threshold_R(R, tau):
    return np.where(R >= tau, R, 0.0)


def density_threshold(R, density):
    iu = np.triu_indices(R.shape[0], 1)
    return float(np.quantile(R[iu], 1 - density))


def build_graph(R_thresholded, X, labeled):
    G = nx.Graph()

    for i in range(R_thresholded.shape[0]):
        attrs = {"semantic_score": float(X[i].max())}
        if labeled[i]:
            attrs["concept"] = int(X[i].argmax())
        G.add_node(i, **attrs)

    ii, jj = np.nonzero(R_thresholded)
    for i, j in zip(ii, jj):
        if i < j:
            G.add_edge(int(i), int(j), weight=float(R_thresholded[i, j]))

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    return G, len(isolates)


def analyze_graph(R, X, labeled, tau, n_perm=500, seed=42):
    rng = np.random.default_rng(seed)

    W = threshold_R(R, tau)
    G, n_isolates = build_graph(W, X, labeled)

    communities = sorted(
        nx_comm.louvain_communities(G, weight="weight", seed=seed),
        key=len, reverse=True
    )

    node_to_comm = {
        node: ci for ci, community in enumerate(communities)
        for node in community
    }

    lab_nodes = [n for n in G.nodes if "concept" in G.nodes[n]]
    labels = np.array([G.nodes[n]["concept"] for n in lab_nodes])
    comm_labels = np.array([node_to_comm[n] for n in lab_nodes])
    G_lab = G.subgraph(lab_nodes).copy()

    results = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_isolates": n_isolates,
        "n_comm": len(communities),
        "modularity": nx_comm.modularity(G, communities, weight="weight"),
    }

    def hard_metrics(current_labels):
        purities = []
        for ci in np.unique(comm_labels):
            mask = comm_labels == ci
            _, counts = np.unique(current_labels[mask], return_counts=True)
            purities.append(counts.max() / mask.sum())

        nx.set_node_attributes(
            G_lab, dict(zip(lab_nodes, current_labels.tolist())), "concept"
        )

        return (
            float(np.mean(purities)),
            ami_score(current_labels, comm_labels),
            nx.attribute_assortativity_coefficient(G_lab, "concept"),
        )

    observed = hard_metrics(labels)
    null = np.array(
        [hard_metrics(rng.permutation(labels)) for _ in range(n_perm)],
        dtype=np.float64,
    )

    for name, value, null_values in zip(
        ["purity", "ami", "assort"], observed, null.T
    ):
        results[name] = float(value)
        finite = null_values[np.isfinite(null_values)]

        if np.isfinite(value) and len(finite):
            results[f"{name}_z"] = (value - finite.mean()) / (finite.std() + 1e-12)
            results[f"{name}_p"] = (np.sum(finite >= value) + 1) / (len(finite) + 1)
        else:
            results[f"{name}_z"] = np.nan
            results[f"{name}_p"] = np.nan

    # Soft semantic similarity within vs between communities
    Xn = X[np.array(lab_nodes)].astype(np.float64)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
    Q = Xn @ Xn.T

    iu = np.triu_indices(len(lab_nodes), 1)
    same = comm_labels[iu[0]] == comm_labels[iu[1]]
    delta = Q[iu][same].mean() - Q[iu][~same].mean()

    null_delta = []
    for _ in range(n_perm):
        perm = rng.permutation(comm_labels)
        same_perm = perm[iu[0]] == perm[iu[1]]
        null_delta.append(Q[iu][same_perm].mean() - Q[iu][~same_perm].mean())

    null_delta = np.array(null_delta)
    results["soft_delta"] = float(delta)
    results["soft_z"] = (delta - null_delta.mean()) / (null_delta.std() + 1e-12)
    results["soft_p"] = (np.sum(null_delta >= delta) + 1) / (n_perm + 1)

    # R-Q alignment after graph thresholding
    R_lab = W[np.ix_(lab_nodes, lab_nodes)]
    r = spearmanr(R_lab[iu], Q[iu])[0]

    null_r = np.nan_to_num(np.array([
        spearmanr(R_lab[iu], Q[np.ix_(perm, perm)][iu])[0]
        for perm in (rng.permutation(len(lab_nodes)) for _ in range(n_perm))
    ]))

    results["alignment_tau_r"] = float(r)
    results["alignment_tau_z"] = (r - null_r.mean()) / (null_r.std() + 1e-12)
    results["alignment_tau_p"] = (np.sum(null_r >= r) + 1) / (n_perm + 1)

    artifacts = {
        "G": G,
        "communities": communities,
        "node_to_comm": node_to_comm,
        "lab_nodes": lab_nodes,
        "labels": labels,
    }

    return results, artifacts


def seed_stability(G, lab_nodes, labels, n_seeds=10):
    rng = np.random.default_rng(0)
    partitions, semantic_amis = [], []

    for seed in range(n_seeds):
        communities = nx_comm.louvain_communities(G, weight="weight", seed=seed)
        node_to_comm = {
            node: ci for ci, community in enumerate(communities)
            for node in community
        }

        partition = np.array([node_to_comm[n] for n in lab_nodes])
        partitions.append(partition)

        ami = ami_score(labels, partition)
        semantic_amis.append(ami)

        null = np.array([
            ami_score(labels, rng.permutation(partition)) for _ in range(200)
        ])
        z = (ami - null.mean()) / (null.std() + 1e-12)
        print(f"  seed {seed:2d}: AMI={ami:.4f} z_null={z:+.1f}")

    pairwise = [
        ami_score(partitions[i], partitions[j])
        for i, j in combinations(range(n_seeds), 2)
    ]

    return (
        float(np.mean(semantic_amis)),
        float(np.std(semantic_amis)),
        float(np.mean(pairwise)),
    )


def community_cards(artifacts, concept_names, concept_categories):
    G = artifacts["G"]
    communities = artifacts["communities"]
    lab_nodes = artifacts["lab_nodes"]

    N = len(lab_nodes)
    labset = set(lab_nodes)
    total_counts = Counter(G.nodes[n]["concept"] for n in lab_nodes)
    rows = []

    for ci, community in enumerate(communities):
        current = [n for n in community if n in labset]
        if len(current) < 3:
            continue

        counts = Counter(G.nodes[n]["concept"] for n in current)
        for concept, k in counts.items():
            rows.append({
                "comm": ci,
                "concept": concept,
                "k": k,
                "p": hypergeom.sf(k - 1, N, total_counts[concept], len(current)),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    if HAVE_SM:
        df["p_adj"] = multipletests(df["p"], method="fdr_bh")[1]

    key = "p_adj" if HAVE_SM else "p"

    for ci in sorted(df["comm"].unique()):
        top = df[df["comm"] == ci].nsmallest(5, key)
        text = [
            f"{concept_names[int(r.concept)]}"
            f"[{concept_categories[int(r.concept)]}]x{int(r.k)}"
            for r in top.itertuples()
        ]
        print(f"  comm {ci:02d}: " + ", ".join(text))

    return df