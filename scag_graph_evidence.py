"""Drop-in plotting cell for a concept–neuron SCAG neighbourhood.
Expects the notebook's selected, labels, view, patch_lookup/records/images/masks,
MODEL, LAYER, QUERY_CONCEPT, PATCH_SIZE, PATCH_SPLIT, PATCH_DIR variables.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D


def draw_scag_evidence():
    if not selected:
        print("No selected channels to plot.")
        return None

    # Preserve the existing graph and frozen assignments.
    nodes = sorted(set(int(i) for i in selected))
    names = ["RV", "MYO", "LV"]
    colors = {"RV": "#627FC0", "MYO": "#DF9B37", "LV": "#399C8B"}
    present = [c for c in names if any(str(labels.iloc[i].label) == c for i in nodes)]
    node_ids = {f"n{i}" for i in nodes}

    coact_edges = []
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for u, v, d in view.edges(data=True):
        if d.get("kind") == "coactivation" and u in node_ids and v in node_ids:
            w = float(d.get("weight", 0.0))
            if np.isfinite(w):
                coact_edges.append((u, v, w))
                graph.add_edge(u, v, weight=max(w, 0.001))

    # Data-driven spring positions: color encodes assigned concepts,
    # whereas physical proximity is driven by the observed coactivation graph.
    raw = nx.spring_layout(graph, seed=19, weight="weight", iterations=400,
                           k=1.6 / np.sqrt(max(len(nodes), 2)))
    rx = np.array([raw[n][0] for n in node_ids]); ry = np.array([raw[n][1] for n in node_ids])
    if len(nodes) > 1 and np.ptp(rx) > 1e-9:
        sx = lambda x: .38 + .54 * (x - rx.min()) / (rx.max() - rx.min())
    else:
        sx = lambda x: .64
    if len(nodes) > 1 and np.ptp(ry) > 1e-9:
        sy = lambda y: .12 + .77 * (y - ry.min()) / (ry.max() - ry.min())
    else:
        sy = lambda y: .52
    pos = {n: (sx(raw[n][0]), sy(raw[n][1])) for n in node_ids}
    anchors = {c: (.105, .82 - .30 * idx) for idx, c in enumerate(present)}

    # Representative evidence: at most two high-IoU channels per concept;
    # fill vacant slots by highest IoU overall. The selection is deterministic.
    eligible = [i for i in nodes if (i, 1) in patch_lookup]
    examples = []
    for c in present:
        candidates = sorted((i for i in eligible if str(labels.iloc[i].label) == c),
                            key=lambda i: (-float(labels.iloc[i].top_iou), i))
        examples.extend(candidates[:2])
    for i in sorted(eligible, key=lambda j: (-float(labels.iloc[j].top_iou), j)):
        if len(examples) >= 6: break
        if i not in examples: examples.append(i)
    examples = examples[:6]

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(17.5, 9.4), facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=.035, right=.985,
                             top=.86, bottom=.14, wspace=.08)
    ax = fig.add_subplot(outer[0, 0])
    ax.set(xlim=(0, 1), ylim=(-.015, 1.015)); ax.set_axis_off()

    def curve(a, b, color, lw, alpha, dashed=False, bend=0.0, z=1):
        p = np.asarray(a, dtype=float); q = np.asarray(b, dtype=float)
        delta = q - p
        perp = np.array([-delta[1], delta[0]], dtype=float)
        perp /= max(float(np.linalg.norm(perp)), 1e-8)
        c1 = p + .35 * delta + bend * perp
        c2 = p + .67 * delta + bend * perp
        path = Path([p, c1, c2, q], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color,
                               lw=lw, alpha=alpha, linestyle="--" if dashed else "-",
                               capstyle="round", zorder=z))

    # Background all functional links: no graph edge silently removed.
    for u, v, _ in coact_edges:
        curve(pos[u], pos[v], "#9CA3AF", .66, .20, bend=.020, z=1)

    # Weighted foreground links: highlight top two coactivation neighbours of each neuron.
    to_highlight = set()
    for n in node_ids:
        incident = sorted(((u, v, w) for u, v, w in coact_edges if n in (u, v)),
                          key=lambda e: -e[2])
        for u, v, _ in incident[:2]:
            to_highlight.add(tuple(sorted((u, v))))
    weights = np.array([w for _, _, w in coact_edges])
    lo, hi = (float(weights.min()), float(weights.max())) if len(weights) else (0., 1.)
    for u, v, w in sorted(coact_edges, key=lambda edge: edge[2]):
        if tuple(sorted((u, v))) in to_highlight:
            s = (w - lo) / max(hi - lo, 1e-8)
            curve(pos[u], pos[v], "#59616F", 1.0 + 2.4 * s, .42 + .43 * s,
                  bend=.02, z=2)

    # Semantic association links. Every association is forced by max IoU.
    for i in nodes:
        c = str(labels.iloc[i].label)
        if c not in anchors:
            continue
        curve((anchors[c][0] + .095, anchors[c][1]), pos[f"n{i}"], colors[c],
              .9, .52, dashed=True, bend=0, z=3)

    # Square concept anchors and circular convolutional-channel nodes.
    for c, (x, y) in anchors.items():
        rect = FancyBboxPatch((x - .095, y - .045), .19, .09,
                              boxstyle="round,pad=0.009,rounding_size=0.014",
                              linewidth=0, facecolor=colors[c], zorder=6)
        ax.add_patch(rect)
        count = sum(str(labels.iloc[i].label) == c for i in nodes)
        txt_color = "#17212C" if c == "MYO" else "white"
        ax.text(x, y + .012, c, ha="center", va="center", color=txt_color,
                fontsize=12, fontweight="bold", zorder=7)
        ax.text(x, y - .021, f"{count} channel{'s' if count != 1 else ''}",
                ha="center", va="center", color=txt_color, fontsize=7.5, zorder=7)

    for i in nodes:
        x, y = pos[f"n{i}"]; c = str(labels.iloc[i].label)
        if i in examples:
            ax.scatter([x], [y], s=1050, facecolor="white", edgecolor="#1F2937",
                       linewidth=1.25, zorder=7)
        ax.scatter([x], [y], s=760, color=colors.get(c, "#A0A0A0"),
                   edgecolor="white", linewidth=1.6, zorder=8)
        ax.text(x, y, f"n{i}", ha="center", va="center", color="white" if c != "MYO" else "#1F2937",
                fontsize=9.2, fontweight="bold", zorder=9)

    ax.set_title("A  |  Joint semantic–functional graph", loc="left", fontsize=13,
                 fontweight="bold", color="#14212D", pad=15)
    fig.text(.632, .885, "B  |  Graph-linked MRI evidence", fontsize=13,
             fontweight="bold", color="#14212D")

    slot = outer[0, 1].subgridspec(3, 2, hspace=.44, wspace=.12)
    for j, i in enumerate(examples):
        a = fig.add_subplot(slot[j // 2, j % 2]); a.axis("off")
        idx = patch_lookup[(i, 1)]
        info = patch_records[idx]
        pair = np.asarray(patch_images[idx]); mask = np.asarray(patch_masks[idx])
        ph, pw = pair.shape[-2:]
        vmin = float(info["image_min"])
        vmax = max(float(info["image_max"]), vmin + 1e-8)
        canvas = np.concatenate([pair[0], np.full((ph, 3), vmax), pair[1]], axis=1)
        a.imshow(canvas, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        if mask.shape == (ph, pw) and mask.any() and not mask.all():
            yy, xx = np.mgrid[:ph, :pw]
            color = colors[str(labels.iloc[i].label)]
            a.contour(xx, yy, mask.astype(float), levels=[.5], colors=[color], linewidths=1.0)
            a.contour(xx + pw + 3, yy, mask.astype(float), levels=[.5], colors=[color], linewidths=1.0)
        a.text(.24, .055, "ED", transform=a.transAxes, ha="center", color="white",
               fontsize=8, bbox=dict(facecolor="#101828", alpha=.7, edgecolor="none", pad=1.2))
        a.text(.76, .055, "ES", transform=a.transAxes, ha="center", color="white",
               fontsize=8, bbox=dict(facecolor="#101828", alpha=.7, edgecolor="none", pad=1.2))
        c = str(labels.iloc[i].label)
        a.set_title(f"n{i}  ·  {c}  |  IoU {float(labels.iloc[i].top_iou):.4f}",
                    fontsize=9, weight="bold", color=colors[c], pad=4)
        a.text(.5, -.07, f"{info['patient_id']}  |  peak {float(info['native_peak']):.2f}",
               transform=a.transAxes, ha="center", va="top", fontsize=8, color="#525F6C")

    fig.suptitle(f"Semantic Coactivation Graph (SCAG)  |  {QUERY_CONCEPT} neighbourhood",
                 fontsize=17, fontweight="bold", color="#14212D", x=.035, ha="left", y=.975)
    fig.text(.035, .927, f"{MODEL}  ·  {LAYER}  ·  {len(nodes)} channels  ·  {len(coact_edges)} functional links  ·  {len(present)} anatomical concepts",
             fontsize=10, color="#5B6573")
    handles = [
        Line2D([0], [0], color="#59616F", lw=2.4, label="Coactivation (solid; width reflects correlation)"),
        Line2D([0], [0], color=colors["LV"], lw=1.4, ls="--", label="Neuron–concept association (forced top IoU)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="#1F2937", markeredgewidth=1.3, markersize=10,
               label="Outlined channel = MRI example"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .075),
               ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .038,
             f"Only top-2 strongest neighbours per channel are emphasised; all graph edges remain visible in pale grey. "
             f"Patches: {PATCH_SIZE}×{PATCH_SIZE} windows centred on activation peaks (not receptive fields or causal evidence). Split: {PATCH_SPLIT}.",
             ha="center", fontsize=8, color="#606D7B")
    fig.text(.5, .019, "Semantic labels are forced maximum-IoU assignments; low IoU values should not be interpreted as confirmed concept detectors.",
             ha="center", fontsize=8, color="#606D7B")
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(PATCH_DIR / f"{QUERY_CONCEPT}_scag_graph_evidence.{extension}",
                    dpi=260, facecolor="white", bbox_inches="tight")
    plt.show()
    return fig


if selected:
    draw_scag_evidence()
