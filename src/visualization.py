"""Export global matrix and graph figures without notebook state."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def plot_matrices(R, Q, path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, matrix, title in zip(axes, (R, Q), ("Coactivation R", "Semantic similarity Q")):
        if len(matrix):
            artist = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
            figure.colorbar(artist, ax=axis)
        axis.set(title=title, xlabel="Neuron", ylabel="Neuron")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_graph(graph, concept_names, path, seed=42):
    figure, axis = plt.subplots(figsize=(9, 7))
    positions = nx.spring_layout(graph, seed=seed, weight="weight") if len(graph) else {}
    nodes = list(graph)
    colors = [graph.nodes[node].get("concept", 0) for node in nodes]
    nx.draw_networkx_edges(graph, positions, ax=axis, alpha=0.12, width=0.5)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=nodes,
        node_color=colors,
        cmap=plt.get_cmap("tab10"),
        vmin=0,
        vmax=max(len(concept_names) - 1, 1),
        node_size=18,
        ax=axis,
    )
    cmap = plt.get_cmap("tab10")
    for concept in sorted(set(colors)):
        name = "unlabeled" if concept == 0 else concept_names[concept]
        axis.scatter([], [], color=cmap(concept / max(len(concept_names) - 1, 1)), label=name, s=25)
    if nodes:
        axis.legend(loc="best", fontsize=8)
    axis.set_title(f"SCAG: {len(graph)} neurons, {graph.number_of_edges()} edges")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
