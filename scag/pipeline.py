"""Run the global SCAG workflow from frozen checkpoints or saved matrices."""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .activations import extract_activations
from .alignment import compute_alignment, compute_semantic_similarity
from .coactivation import compute_correlation, threshold_correlations
from .concepts import network_dissection, semantic_views
from .data import SemanticDataset, collate_samples, load_splits
from .evaluation import evaluate_semantics
from .graph import build_graph, community_enrichment, community_stability, detect_communities, graph_summary
from .models import load_model
from .utils import finite_matrix, save_json, symmetric_matrix


def load_cache(path):
    """Read current outputs or the existing *_semantics.npz notebook caches."""
    with np.load(Path(path).expanduser(), allow_pickle=False) as cache:
        profiles = finite_matrix(cache["X"], "X")
        labeled = np.asarray(cache["labeled"])
        reference_key = "rho_reference" if "rho_reference" in cache else "rho_train"
        graph_key = "rho_graph" if "rho_graph" in cache else "rho_val"
        reference = symmetric_matrix(cache[reference_key], reference_key)
        graph = symmetric_matrix(cache[graph_key], graph_key)
    if labeled.shape != (len(profiles),) or not np.isin(labeled, [False, True]).all():
        raise ValueError("Cache labeled mask must have one boolean value per neuron")
    if reference.shape != graph.shape or reference.shape[0] != len(profiles):
        raise ValueError("Cache matrices and profiles must have the same neuron order and count")
    semantic_views(profiles)
    return profiles, labeled.astype(bool), reference, graph


def extract_inputs(config):
    if not config.data.processed_dir or not config.model.checkpoint:
        raise ValueError("Set data.processed_dir and model.checkpoint, or supply input.cache")
    splits = load_splits(
        config.data.processed_dir,
        dataset=config.data.name,
        classes=list(config.data.classes),
        expected_sizes=config.data.expected_sizes,
        validation_per_class=config.data.validation_per_class,
        split_seed=config.data.split_seed,
    )
    options = dict(
        dataset=config.data.name,
        view=config.data.view,
        minimum_size=config.data.minimum_size,
        masks_root=config.data.masks_root,
    )
    reference = SemanticDataset(config.data.processed_dir, splits, list(config.data.reference_splits), **options)
    graph = SemanticDataset(config.data.processed_dir, splits, list(config.data.graph_splits), **options)
    overlap = set(reference.patient_ids) & set(graph.patient_ids)
    if overlap:
        raise ValueError(f"Reference and graph cohorts overlap: {sorted(overlap)}")
    device = config.model.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, layer, metadata = load_model(
        config.model.checkpoint,
        classes=list(config.data.classes),
        task=config.data.task,
        layer=config.model.layer,
        device=device,
    )
    loader_options = dict(
        batch_size=config.activations.batch_size,
        num_workers=config.activations.workers,
        shuffle=False,
        collate_fn=collate_samples,
    )
    print(f"Extracting {len(reference)} reference and {len(graph)} graph patients at {layer}", flush=True)
    reference_features = extract_activations(model, layer, DataLoader(reference, **loader_options), device)
    profiles, labeled, thresholds = network_dissection(
        reference_features.maps,
        reference_features.masks,
        alpha=config.dissection.alpha,
        eta=config.dissection.eta,
        channel_batch_size=config.dissection.channel_batch_size,
    )
    reference_R = compute_correlation(reference_features.pooled, config.coactivation.method)
    reference_features.maps.clear()
    reference_features.masks.clear()
    graph_features = extract_activations(model, layer, DataLoader(graph, **loader_options), device, keep_maps=False)
    graph_R = compute_correlation(graph_features.pooled, config.coactivation.method)
    metadata.update(
        reference_patients=reference_features.patient_ids,
        graph_patients=graph_features.patient_ids,
        reference_splits=list(config.data.reference_splits),
        graph_splits=list(config.data.graph_splits),
        uses_test_for_analysis="test" in config.data.graph_splits,
        split_audit="verified_disjoint",
        shared_encoder_aggregation="mean_of_ED_ES_maps",
    )
    return profiles, labeled, reference_R, graph_R, thresholds, metadata


def run_analysis(config):
    """Compute profiles → R/Q alignment → thresholded graphs and global metrics."""
    output = Path(config.output.dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not config.output.overwrite:
        raise FileExistsError(f"Output is nonempty: {output}; choose a new directory or set output.overwrite=true")
    if config.input.cache:
        profiles, labeled, reference_R, graph_R = load_cache(config.input.cache)
        thresholds = np.array([], dtype=float)
        provenance = {
            "source_cache": str(Path(config.input.cache).expanduser().resolve()),
            "split_audit": "not_verified_from_legacy_cache",
            "note": "Reference/graph matrices and labeled mask are reused as supplied; no dissection is refitted.",
        }
    else:
        profiles, labeled, reference_R, graph_R, thresholds, provenance = extract_inputs(config)
    views = semantic_views(profiles)
    selected_views = list(config.semantics.views)
    if not selected_views or len(selected_views) != len(set(selected_views)) or not set(selected_views).issubset(views):
        raise ValueError(f"semantics.views must select distinct entries from {list(views)}")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    save_json(output / "provenance.json", provenance)
    np.savez_compressed(
        output / "semantics.npz",
        X=profiles,
        labeled=labeled,
        rho_reference=reference_R,
        rho_graph=graph_R,
        thresholds=thresholds,
    )
    lab = np.flatnonzero(labeled)
    ix = np.ix_(lab, lab)
    reproducibility = compute_alignment(reference_R[ix], graph_R[ix], positive_only=False, n_permutations=0)
    save_json(output / "edge_reproducibility.json", reproducibility)
    alignments, graph_rows = [], []
    graph_specs = [("threshold", {"threshold": config.graph.threshold})]
    if config.graph.density is not None:
        graph_specs.append(("density", {"density": config.graph.density}))
    weights = {name: threshold_correlations(graph_R, **options) for name, options in graph_specs}
    for level in selected_views:
        X, names = views[level]
        level_dir = output / level
        level_dir.mkdir(exist_ok=True)
        pd.DataFrame(X, columns=names).rename_axis("neuron").to_csv(level_dir / "profiles.csv")
        for mode in config.semantics.similarities:
            Q = compute_semantic_similarity(X[lab], mode=mode)
            np.save(level_dir / f"Q_{mode}.npy", Q)
            np.save(level_dir / "neuron_indices.npy", lab)
            for split, R in [("reference", reference_R), ("graph", graph_R)]:
                result = compute_alignment(
                    R[ix],
                    Q,
                    n_permutations=config.alignment.permutations,
                    seed=config.seed,
                    positive_only=config.alignment.positive_only,
                    method=config.alignment.method,
                )
                alignments.append({"view": level, "similarity": mode, "cohort": split, "graph": "full", **result})
            if config.output.figures:
                from .visualization import plot_matrices

                plot_matrices(graph_R[ix], Q, level_dir / f"matrices_{mode}.png")
        for kind, W in weights.items():
            graph = build_graph(W, X, labeled, drop_isolates=config.graph.drop_isolates)
            active_labeled = sorted(node for node in graph if graph.nodes[node]["labeled"])
            for mode in config.semantics.similarities:
                Q_graph = compute_semantic_similarity(X[active_labeled], mode=mode)
                result = compute_alignment(
                    W[np.ix_(active_labeled, active_labeled)],
                    Q_graph,
                    n_permutations=config.alignment.permutations,
                    seed=config.seed,
                    positive_only=config.alignment.positive_only,
                    method=config.alignment.method,
                )
                alignments.append({"view": level, "similarity": mode, "cohort": "graph", "graph": kind, **result})
            communities = detect_communities(graph, seed=config.seed, resolution=config.graph.resolution)
            membership = {node: index for index, community in enumerate(communities) for node in community}
            nx.set_node_attributes(graph, membership, "community")
            summary = graph_summary(graph, communities, resolution=config.graph.resolution)
            summary.update(
                evaluate_semantics(graph, communities, X, n_permutations=config.graph.permutations, seed=config.seed)
            )
            graph_rows.append({"view": level, "graph": kind, **summary})
            nx.write_graphml(graph, level_dir / f"{kind}.graphml")
            community_enrichment(graph, communities, names, min_labeled=config.graph.min_community_labeled).to_csv(
                level_dir / f"{kind}_enrichment.csv", index=False
            )
            community_stability(graph, seeds=config.graph.community_seeds, resolution=config.graph.resolution).to_csv(
                level_dir / f"{kind}_stability.csv", index=False
            )
            if config.output.figures:
                from .visualization import plot_graph

                plot_graph(graph, names, level_dir / f"{kind}_graph.png", config.seed)
        print(f"Analyzed {level}: {len(lab)} labeled neurons", flush=True)
    pd.DataFrame(alignments).to_csv(output / "alignment.csv", index=False)
    pd.DataFrame(graph_rows).to_csv(output / "graph_metrics.csv", index=False)
    save_json(
        output / "summary.json",
        {
            "n_neurons": len(profiles),
            "n_labeled": len(lab),
            "edge_reproducibility": reproducibility,
            "alignment": alignments,
            "graphs": graph_rows,
        },
    )
    print(f"Global SCAG results: {output}", flush=True)
    return output
