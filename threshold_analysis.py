"""Sweep the coactivation edge threshold without rerunning model inference."""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

from scag.graph import build_graph, threshold_R


def analyze_thresholds(R, X, labeled, thresholds=None, seed=42):
    """Print and return metric lists aligned with ``threshold``.

    Density and largest-component fraction use the original neuron count;
    node, component, community and mean-degree statistics exclude isolates.
    Undefined modularity/AMI values are NaN. Louvain uses the same seed and
    edge weights as the main pipeline; no permutation tests are run.
    """
    R, X, labeled = np.asarray(R), np.asarray(X), np.asarray(labeled)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or not np.isfinite(R).all() or not np.allclose(R, R.T):
        raise ValueError("R must be a finite, symmetric square matrix.")
    n_neurons = R.shape[0]
    if X.ndim != 2 or X.shape[0] != n_neurons or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("X must contain finite concept scores for every neuron.")
    if labeled.shape != (n_neurons,) or labeled.dtype != np.bool_:
        raise ValueError("labeled must be a boolean mask with one entry per neuron.")
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = np.asarray(list(thresholds), dtype=float)
    if thresholds.ndim != 1 or not thresholds.size or not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("thresholds must be a nonempty list of finite values between 0 and 1.")

    rows = []
    possible_edges = n_neurons * (n_neurons - 1) // 2
    for tau in thresholds.tolist():
        G, n_isolates = build_graph(threshold_R(R, tau), X, labeled)
        n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
        component_sizes = [len(component) for component in nx.connected_components(G)]
        communities = nx.community.louvain_communities(G, weight="weight", seed=seed) if n_edges else []
        node_to_comm = {node: index for index, community in enumerate(communities) for node in community}
        lab_nodes = [node for node in G if "concept" in G.nodes[node]]
        ami = np.nan
        if len(lab_nodes) >= 2:
            ami = adjusted_mutual_info_score([G.nodes[node]["concept"] for node in lab_nodes], [node_to_comm[node] for node in lab_nodes])
        rows.append({
            "threshold": tau,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_isolates": n_isolates,
            "density": n_edges / possible_edges if possible_edges else 0.0,
            "mean_degree": 2 * n_edges / n_nodes if n_nodes else 0.0,
            "n_components": len(component_sizes),
            "largest_component_fraction": max(component_sizes, default=0) / n_neurons if n_neurons else 0.0,
            "n_comm": len(communities),
            "modularity": nx.community.modularity(G, communities, weight="weight") if n_edges else np.nan,
            "n_labeled": len(lab_nodes),
            "ami": float(ami),
        })

    results = pd.DataFrame(rows).to_dict(orient="list")
    for name, values in results.items():
        print(f"{name} = {values}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Saved <tag>_scag.npz from run_scag.py (uses R_eval, X and labeled).")
    parser.add_argument("--thresholds", nargs="+", type=float, help="Correlation cutoffs to sweep, including endpoints if desired.")
    parser.add_argument("--output", type=Path, help="CSV destination; defaults to <input_stem>_thresholds.csv beside the input.")
    args = parser.parse_args()
    output = args.output if args.output is not None else args.input.with_name(f"{args.input.stem}_thresholds.csv")
    if output.resolve() == args.input.resolve():
        parser.error("--output must differ from the input archive.")
    with np.load(args.input, allow_pickle=False) as data:
        results = analyze_thresholds(data["R_eval"], data["X"], data["labeled"], thresholds=args.thresholds)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
