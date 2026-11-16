"""Assign every channel its highest training-IoU concept, without threshold gating.

Forced labels are descriptive assignments, not additional semantic evidence.
Original results remain unchanged; training labels stay frozen on evaluation.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from run_semantic_grounding import DEFAULT_RUN
from scag.research import file_sha256


def assign_labels(scores, names, gamma):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != len(names) or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError('Require unique concept names matching a neuron x concept matrix.')
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any() or not 0 <= gamma <= 1:
        raise ValueError('IoUs and threshold must be finite and in [0,1].')
    top = scores.argmax(axis=1)
    maximum = scores[np.arange(len(scores)), top]
    second = np.sort(scores, axis=1)[:, -2]
    return pd.DataFrame(dict(neuron=np.arange(len(scores)), label=np.asarray(names)[top], concept_index=top, training_iou=maximum, runner_up_iou=second, margin=maximum - second, passes_original_threshold=maximum >= gamma, forced_below_threshold=maximum < gamma, tied_maximum=(scores == maximum[:, None]).sum(axis=1) > 1, all_zero=(scores == 0).all(axis=1)))


def run(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = directory.resolve()
    protocol = json.loads((directory.parent.parent / 'protocol.json').read_text())
    archive = dict(np.load(directory / 'representations.npz'))
    names = archive['concepts'].tolist()
    gamma, tau = protocol['gamma'], protocol['tau']
    labels = assign_labels(archive['S_train'], names, gamma)
    channels, top = labels.neuron.to_numpy(), labels.concept_index.to_numpy()
    labels['original_labels'] = ['/'.join(name for c, name in enumerate(names) if row[c] >= gamma) or 'unlabelled' for row in archive['S_train']]
    for split in ['val', 'test']:
        scores = archive[f'S_{split}']
        evaluated = assign_labels(scores, names, gamma)
        labels[f'{split}_iou_for_training_label'] = scores[channels, top]
        labels[f'{split}_passes_threshold'] = scores[channels, top] >= gamma
        labels[f'{split}_argmax_label'] = evaluated.label
        labels[f'{split}_agrees_with_training'] = evaluated.label == labels.label
        labels[f'{split}_unique_positive_maximum'] = ~evaluated.tied_maximum & ~evaluated.all_zero
    output = directory / 'argmax_labels' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True)
    labels.to_csv(output / 'neuron_labels.csv', index=False)
    counts = labels.groupby('label', sort=False).agg(neurons=('neuron', 'size'), originally_supported=('passes_original_threshold', 'sum'), forced_below_threshold=('forced_below_threshold', 'sum'), tied=('tied_maximum', 'sum'), all_zero=('all_zero', 'sum'), test_agreement=('test_agrees_with_training', 'sum'), test_above_threshold=('test_passes_threshold', 'sum')).reindex(names)
    counts.to_csv(output / 'label_counts.csv')
    graph = nx.Graph(description='Argmax labels forced from training IoU; assignments are not proof of concept selectivity')
    graph.add_nodes_from([(f'concept:{name}', dict(kind='concept', name=name)) for name in names])
    for row in labels.itertuples():
        graph.add_node(f'n{row.neuron}', kind='neuron', channel=int(row.neuron), label=row.label)
        graph.add_edge(f'n{row.neuron}', f'concept:{row.label}', kind='argmax_assignment', weight=float(row.training_iou), passes_original_threshold=bool(row.passes_original_threshold), tied=bool(row.tied_maximum), all_zero=bool(row.all_zero))
    R = archive['R_train']
    for i, j in zip(*np.where(np.triu((R >= tau) & (R > 0), 1))):
        graph.add_edge(f'n{i}', f'n{j}', kind='coactivation', weight=float(R[i, j]))
    nx.write_graphml(graph, output / 'joint_graph.graphml')
    colors = dict(zip(names, ['#6c86bf', '#e3a34d', '#58a899']))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, counts.originally_supported, label='Meets original IoU threshold', color=[colors[c] for c in names])
    ax.bar(names, counts.forced_below_threshold, bottom=counts.originally_supported, label='Forced below threshold', color=[colors[c] for c in names], alpha=.35, hatch='//')
    ax.set(ylabel='Neurons', title='Every neuron assigned its highest training-IoU concept')
    ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'label_counts.{suffix}', dpi=180)
    plt.close(fig)
    explanation_sources = []
    for concept in names:
        candidates = sorted(p for p in (directory / 'concept_explanations').glob(f'{concept}_*') if (p / 'EXPLANATION.md').exists())
        if not candidates:
            continue
        source = candidates[-1]
        nodes, edges = pd.read_csv(source / 'nodes.csv'), pd.read_csv(source / 'edges.csv')
        nodes = nodes.rename(columns={'label': 'previous_label'}).merge(labels[['neuron', 'label', 'training_iou', 'forced_below_threshold', 'tied_maximum', 'all_zero']], on='neuron', validate='one_to_one')
        nodes.to_csv(output / f'{concept}_explanation_nodes.csv', index=False)
        edges.to_csv(output / f'{concept}_explanation_edges.csv', index=False)
        small = nx.Graph()
        small.add_nodes_from(nodes.neuron.tolist())
        small.add_weighted_edges_from(edges[['source', 'target', 'train_r']].itertuples(index=False, name=None))
        fig, ax = plt.subplots(figsize=(10, 7))
        positions = nx.spring_layout(small, seed=42)
        text = {row.neuron: f'n{row.neuron}\n{row.label}{"*" if row.forced_below_threshold else ""}\nIoU {row.training_iou:.4f}' for row in nodes.itertuples()}
        nx.draw_networkx(small, positions, ax=ax, labels=text, node_color=[colors[c] for c in nodes.label], node_size=2300, font_size=8, edge_color='#a0a0a0', width=[1 + 2 * small[u][v]['weight'] for u, v in small.edges])
        ax.set_title(f'{concept} query: original channel set and coactivation edges\nHighest-IoU training labels; * = below original association threshold')
        ax.axis('off')
        fig.text(.5, .02, 'Only labels changed. Query seeds, graph selection and previously measured interventions are unchanged.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, 1))
        for suffix in ['png', 'pdf']:
            fig.savefig(output / f'{concept}_explanation_argmax.{suffix}', dpi=180)
        plt.close(fig)
        explanation_sources.append(str(source))
    unique = ~labels.tied_maximum & ~labels.all_zero & labels.test_unique_positive_maximum
    report = f'''# Forced highest-IoU labels

Every one of {len(labels)} neurons is assigned its highest training-IoU concept. Original labelling required at least one IoU >= {gamma}; neurons below that threshold were unlabelled. Here that eligibility condition is removed, not the scores changed.

- Original threshold-supported neurons: {int(labels.passes_original_threshold.sum())}.
- Forced below-threshold assignments: {int(labels.forced_below_threshold.sum())}.
- Exact ties: {int(labels.tied_maximum.sum())}; all-zero profiles: {int(labels.all_zero.sum())}.
- Ties (including all-zero profiles) deterministically choose the first concept in stored order: {names}. They carry no evidence that the chosen concept is preferable.
- Frozen training labels still meet the original threshold on test: {int(labels.test_passes_threshold.sum())}/{len(labels)}.
- Descriptive train/test top-label agreement: {int(labels.test_agrees_with_training.sum())}/{len(labels)}.
- Agreement excluding tied/zero maxima in either split: {int(labels.loc[unique, 'test_agrees_with_training'].sum())}/{int(unique.sum())}.

```
''' + counts.to_string() + '''
```

Training labels remain frozen. Evaluation argmax labels are recorded only to measure stability, not to relabel the graph. Original multi-label associations are preserved in the CSV; argmax reduces each neuron to one assigned label. Zero-weight assignments are retained in GraphML so every channel has an assignment even when all scores are zero.

This achieves complete assignment coverage, not complete semantic validity. A low-IoU or tied assignment should be described as a relative best match within this vocabulary, not a validated detector or causal interpretation. The highest score need not be high in absolute terms. No associations were learned and no new interventions were run.

Outputs: neuron_labels.csv; label_counts.csv; joint_graph.graphml; label_counts.png/pdf; and LV/MYO/RV explanation graph figures where prior explanations exist. Asterisks in figures mark forced below-threshold labels. These figures reuse each latest completed explanation's exact channel/edge set; they do not rerun concept queries with different eligibility rules. The original pipeline and results remain intact.
'''
    (output / 'RESULTS.md').write_text(report)
    (output / 'provenance.json').write_text(json.dumps(dict(source=str(directory), policy='argmax of training IoU, first stored concept wins exact ties', gamma=gamma, tau=tau, concepts=names, representation_sha256=file_sha256(directory / 'representations.npz'), script_sha256=file_sha256(__file__), explanation_sources=explanation_sources), indent=2))
    assert len(labels) == len(archive['S_train']) and labels.label.notna().all()
    assert sum(d['kind'] == 'argmax_assignment' for _, _, d in graph.edges(data=True)) == len(labels)
    print(report[:1800])
    print(f'Saved: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer-dir', type=Path, default=DEFAULT_RUN)
    run(parser.parse_args().layer_dir)
