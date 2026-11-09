"""Exportable figures for the concept-relation analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .relations import concept_subgraph


def plot_overlays(activations, masks, profiles, S_train, names, loader, manifest, directory, split):
    """Show training-selected neurons with concept-overlap evidence on each split."""
    fig, axes = plt.subplots(len(names), 3, figsize=(10, 3.5 * len(names)), squeeze=False, layout="constrained")
    for c, name in enumerate(names):
        neuron = int(np.argmax(S_train[:, c]))
        iou = profiles["per_patient_iou"][:, neuron, c]
        sample = int(np.argmax(iou))
        images = loader.dataset[sample][0].numpy()
        up = torch.nn.functional.interpolate(activations[sample:sample + 1, neuron:neuron + 1], size=masks.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
        active = up > profiles["thresholds"][neuron]
        target = masks[sample, c].numpy()
        for phase in (0, 1):
            axes[c, phase].imshow(images[phase], cmap="gray")
            if active.any() and not active.all():
                axes[c, phase].contour(active, levels=[.5], colors=["cyan"], linewidths=.7)
            axes[c, phase].set_title(f"{name} / n{neuron} / {manifest.iloc[sample].pid}\n{'ED' if phase == 0 else 'ES'}; cyan = shared activation mask")
        overlay = np.zeros((*target.shape, 3))
        overlay[target] = [1, 0, 1]
        overlay[active] = [0, 1, 1]
        overlay[target & active] = [1, 1, 1]
        axes[c, 2].imshow(overlay)
        axes[c, 2].set_title(f"Shared-grid concept overlap: IoU={iou[sample]:.4f}\nMagenta: concept; cyan: activation; white: both")
        for ax in axes[c]:
            ax.axis("off")
    fig.suptitle(f"{split}: neuron selected on training IoU; patient selected for illustrative overlap")
    fig.savefig(directory / f"concept_overlays_{split}.png", dpi=160)
    plt.close(fig)


def plot_ablation(summary, directory, split):
    concepts = summary.concept.unique()
    fig, axes = plt.subplots(len(concepts), 2, figsize=(12, 3.5 * len(concepts)), squeeze=False, layout="constrained")
    for row, concept in enumerate(concepts):
        for column, metric in enumerate(("kl", "flip")):
            subset = summary[(summary.concept == concept) & (summary.metric == metric)]
            for strategy, values in subset.groupby("strategy"):
                values = values.sort_values("budget")
                axes[row, column].plot(values.budget, values["mean"], marker="o", label=strategy)
                axes[row, column].fill_between(values.budget, values.ci_low, values.ci_high, alpha=.12)
            axes[row, column].set(title=f"{concept}: {metric.upper()}", xlabel="Fraction of layer channels ablated", ylabel="Mean effect (95% patient CI)")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"{split}: training-selected interventions; inspect matched-control diagnostics")
    fig.savefig(directory / f"ablation_budgets_{split}.png", dpi=160)
    plt.close(fig)


def plot_layer(relations, pairs, A, S, names, args, directory, split):
    matrix = np.full((len(names), len(names)), np.nan)
    for row in relations.itertuples():
        c, d = names.index(row.concept_c), names.index(row.concept_d)
        matrix[c, d] = matrix[d, c] = row.profile_z
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    limit = max(1, np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1
    im = axes[0].imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[0].set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names, title=f"{split}: concept coactivation (null z)")
    axes[0].tick_params(axis="x", rotation=45)
    fig.colorbar(im, ax=axes[0])
    if len(pairs):
        density = axes[1].hexbin(pairs.R, pairs.Q, gridsize=40, mincnt=1, bins="log")
        fig.colorbar(density, ax=axes[1], label="Pair count (log scale)")
    axes[1].axvline(args.tau, color="black", linestyle="--")
    axes[1].axhline(args.similarity, color="black", linestyle="--")
    axes[1].set(xlabel="Signed coactivation R", ylabel="Semantic cosine Q", xlim=(-1, 1), ylim=(0, 1), title="Neuron-pair relationship types")
    fig.savefig(directory / f"relations_{split}.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False, layout="constrained")
    fig.suptitle("Within-layer coactivation: statistical associations")
    for concept, ax in zip(names, axes[0]):
        nodes, edges = concept_subgraph(A, S, names, concept, args.gamma, args.neighbors)
        # shortcut: render 12 strongest edges for legibility; CSVs retain the complete query.
        edges = edges.sort_values("coactivation", ascending=False).head(12)
        graph = nx.from_pandas_edgelist(edges, "source", "target", edge_attr="coactivation")
        if graph.number_of_nodes():
            labels = {i: f"n{i}\n{names[int(S[i].argmax())]} {S[i].max():.3f}" if S[i].max() > 0 else f"n{i}\nunlabelled" for i in graph}
            positions = nx.circular_layout(graph)
            seeds = set(nodes.loc[nodes.is_seed, "neuron"])
            nx.draw_networkx(graph, positions, ax=ax, labels=labels, node_size=650, font_size=7, node_color=["#f1ba65" if i in seeds else "#9dccdd" for i in graph])
            nx.draw_networkx_edge_labels(graph, positions, ax=ax, edge_labels={(i, j): f"{v['coactivation']:.2f}" for i, j, v in graph.edges(data=True)}, font_size=6)
        ax.set_title(f"{concept}: strongest query edges\nOrange: query seed; blue: neighbour")
        ax.axis("off")
    fig.savefig(directory / f"evidence_subgraphs_{split}.png", dpi=180)
    plt.close(fig)


def plot_comparison(table, directory):
    table = table[table.concept_c != table.concept_d].copy()
    table["relation"] = table.concept_c + "–" + table.concept_d
    for metric in ("profile_z", "degree_z", "weight_per_edge_mass"):
        pivot = table.pivot_table(index=["model", "layer", "split"], columns="relation", values=metric)
        pivot = pivot.dropna(how="all")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(7, len(pivot.columns)), max(3, 0.3 * len(pivot))), layout="constrained")
        values = pivot.to_numpy()
        limit = max(1e-12, np.nanmax(np.abs(values)))
        im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns, yticks=range(len(pivot)), yticklabels=[" / ".join(index) for index in pivot.index], title=f"Concept relationships across layers/models: {metric}")
        ax.tick_params(axis="x", rotation=45)
        fig.colorbar(im, ax=ax)
        fig.savefig(directory / f"comparison_{metric}.png", dpi=180)
        plt.close(fig)
