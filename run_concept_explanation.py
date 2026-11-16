"""Generate one-layer, evidence-linked concept explanations from a frozen research run.

Default: refresh exact selected-channel overlays and interventions using the checkpoint.
--cached-only: produce structural evidence without loading the model; never substitute
an intervention on different neurons. Outputs are written to a new timestamped directory.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DEFAULT_RUN = Path('scag_outputs/research/acdc/anatomy/20261109T154512_822872Z/resnet18/encoder.layer4')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_graph(S, R, concept_index, gamma, tau, seeds=3, neighbors=2):
    if S.ndim != 2 or R.shape != (len(S), len(S)) or not np.isfinite(S).all() or not np.isfinite(R).all():
        raise ValueError('Invalid semantic/coactivation arrays.')
    if (S < 0).any() or (S > 1 + 1e-6).any() or (np.abs(R) > 1 + 1e-6).any() or not np.allclose(R, R.T):
        raise ValueError('Require IoUs in [0,1] and a symmetric correlation matrix.')
    eligible = np.flatnonzero(S[:, concept_index] >= gamma)
    selected = sorted(eligible.tolist(), key=lambda i: (-S[i, concept_index], i))[:seeds]
    if not selected:
        raise ValueError('No training neurons meet the concept threshold; no explanation is asserted.')
    nodes = set(selected)
    for i in selected:
        candidates = [j for j in range(len(S)) if j != i and R[i, j] >= tau and R[i, j] > 0]
        nodes.update(sorted(candidates, key=lambda j: (-R[i, j], j))[:neighbors])
    nodes = sorted(nodes)
    edges = [(i, j) for i in nodes for j in nodes if i < j and R[i, j] >= tau and R[i, j] > 0]
    return selected, nodes, edges


def exact_cached_effect(directory, split, selected):
    path = directory / f'ablation_selections_{split}.csv'
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    keep = rows.neurons.map(lambda value: set(json.loads(value)) == set(selected))
    # One intervention can occur under several strategy names; do not count duplicates.
    return rows.loc[keep].head(1)


def bootstrap_summary(patients, repeats, seed):
    rng = np.random.default_rng(seed)
    ids = patients.pid.drop_duplicates().tolist()
    draws = rng.integers(len(ids), size=(repeats, len(ids)))
    summaries = []
    for strategy, frame in patients.groupby('strategy', sort=False):
        means = frame.groupby('pid')[['kl', 'flip', 'margin_drop']].mean().reindex(ids)
        if means.isna().any().any():
            raise ValueError('Interventions must contain the same patients.')
        for metric in means:
            values = means[metric].to_numpy()
            lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
            summaries.append(dict(strategy=strategy, metric=metric, mean=values.mean(), ci_low=lo, ci_high=hi))
    return pd.DataFrame(summaries)


def refresh(args, directory, protocol, archive, manifest, nodes, seeds, output):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from scag.config import SCAGConfig
    from scag.models import build_model
    from scag.data import load_datasets
    from scag.research import cache_patients
    from scag.activations import extract_activations
    from scag.semantics import concept_masks, concept_profiles
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use cinema on the GPU host or explicitly pass --device cpu.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = SCAGConfig(dataset=protocol['dataset'], model_seed=protocol.get('model_seed', 0))
    checkpoint = json.loads((directory.parent / 'checkpoint.json').read_text())
    actual = cfg.model_root / directory.parent.name / cfg.formulation / cfg.initialization / f'seed_{cfg.model_seed}' / 'best_val_mcc.pt'
    if actual.resolve() != Path(checkpoint['path']).resolve() or digest(actual) != checkpoint['sha256']:
        raise ValueError('Checkpoint path/hash differs from the saved run.')
    print(f'Refreshing evidence on {args.device}: {len(manifest)} patients, {len(nodes)} displayed channels', flush=True)
    model, _, _ = build_model(directory.parent.name, cfg)
    model.to(args.device).eval()
    (output / 'class_names.json').write_text(json.dumps(list(cfg.classes), indent=2))
    hook = model.get_submodule(directory.name)
    dataset = load_datasets(cfg)[{'val': 1, 'test': 2}[args.split]]
    mapping = {str(pid): i for i, pid in enumerate(dataset.metadata.pid)}
    if len(mapping) != len(dataset):
        raise ValueError('Duplicate dataset patient IDs.')
    subset = torch.utils.data.Subset(dataset, [mapping[pid] for pid in manifest.pid])
    loader = cache_patients(subset, manifest.copy(), args.batch_size)
    acts, segments = extract_activations(model, loader, hook, args.device)
    pooled, saved = acts.mean(dim=(2, 3)).numpy(), archive[f'pooled_{args.split}']
    if pooled.shape != saved.shape:
        raise ValueError(f'Activation shape changed: refreshed {pooled.shape}, saved {saved.shape}.')
    if not np.allclose(pooled, saved, rtol=1e-4, atol=1e-5):
        mismatch = ~np.isclose(pooled, saved, rtol=1e-4, atol=1e-5)
        raise ValueError(f'Refreshed activations differ: {mismatch.sum()}/{pooled.size} entries; max absolute error {np.max(np.abs(pooled - saved)):.6g}. Batch size {args.batch_size}, original {protocol["batch_size"]}; TF32 disabled. Investigate model/preprocessing before interpreting evidence.')
    masks, names = concept_masks(segments, protocol['semantic_level'])
    if names != archive['concepts'].tolist():
        raise ValueError('Concept ordering changed.')
    profiles = concept_profiles(acts[:, nodes], masks, thresholds=archive['thresholds'][nodes], device=args.device)
    if not np.allclose(profiles['S'], archive[f'S_{args.split}'][nodes], atol=1e-6, rtol=1e-4):
        raise ValueError('Refreshed concept scores differ from saved evaluation evidence.')
    sample = int(np.random.default_rng(args.seed).integers(len(manifest)))
    c = names.index(args.concept)
    shown = [seeds[0]] + [i for i in nodes if i not in seeds][:1]
    if len(shown) == 1 and len(nodes) > 1:
        shown.append(next(i for i in nodes if i != shown[0]))
    visual = []
    for neuron in shown:
        target_index = c if neuron in seeds else int(archive['S'][neuron].argmax())
        up = torch.nn.functional.interpolate(acts[sample:sample + 1, neuron:neuron + 1], size=masks.shape[-2:], mode='bilinear', align_corners=False)[0, 0].numpy()
        visual.append(dict(neuron=neuron, concept=names[target_index], active=up > archive['thresholds'][neuron], target=masks[sample, target_index].numpy().astype(bool), iou=float(profiles['per_patient_iou'][sample, nodes.index(neuron), target_index])))
    images = loader.dataset[sample][0].numpy()
    np.savez_compressed(output / 'overlay_arrays.npz', images=images, neurons=shown, active=np.stack([v['active'] for v in visual]), target=np.stack([v['target'] for v in visual]), sample_index=sample, pid=manifest.iloc[sample].pid)
    (output / 'overlay_selection.json').write_text(json.dumps(dict(sample_index=sample, pid=manifest.iloc[sample].pid, rule='one seeded patient, independent of overlap', neurons=shown), indent=2))
    S, R = archive['S'], archive['R_train']
    k = len(nodes)
    A = np.where((R >= protocol['tau']) & (R > 0), R, 0).copy()
    np.fill_diagonal(A, 0)
    plans = [('displayed_subgraph', nodes), ('query_seeds', seeds), ('semantic', np.argsort(-S[:, c], kind='stable')[:k].tolist()), ('functional', np.argsort(-A.sum(axis=1), kind='stable')[:k].tolist())]
    rng = np.random.default_rng(args.seed)
    plans.extend((f'random_{i}', sorted(rng.choice(len(S), k, replace=False).tolist())) for i in range(args.random_repeats))
    with torch.inference_mode():
        clean = torch.cat([model(batch[0].to(args.device)).cpu() for batch in loader])
    order = clean.argsort(dim=1, descending=True)
    y, r = order[:, 0], order[:, 1]
    index = torch.arange(len(clean))
    original_margin = clean[index, y] - clean[index, r]
    logp = clean.log_softmax(1)
    cache, rows = {}, []
    for number, (strategy, channels) in enumerate(plans):
        key = tuple(sorted(channels))
        if key not in cache:
            def zero(module, inputs, result):
                result = result.clone()
                result[:, list(key)] = 0
                return result
            handle = hook.register_forward_hook(zero)
            try:
                with torch.inference_mode():
                    logits = torch.cat([model(batch[0].to(args.device)).cpu() for batch in loader])
            finally:
                handle.remove()
            cache[key] = logits
        logits = cache[key]
        kl = (logp.exp() * (logp - logits.log_softmax(1))).sum(1).clamp_min(0).numpy()
        flip = (logits.argmax(1) != y).numpy()
        margin = (original_margin - (logits[index, y] - logits[index, r])).numpy()
        for i, pid in enumerate(manifest.pid):
            rows.append(dict(pid=pid, sample_index=i, strategy='random' if strategy.startswith('random_') else strategy, repeat=number if strategy.startswith('random_') else 0, neurons=json.dumps(list(key)), k=len(key), kl=float(kl[i]), flip=float(flip[i]), margin_drop=float(margin[i]), original_class=int(y[i]), comparator_class=int(r[i]), original_probabilities=json.dumps(logp[i].exp().tolist()), ablated_probabilities=json.dumps(logits[i].softmax(0).tolist())))
        print(f'  Intervention {number + 1}/{len(plans)}: {strategy}', flush=True)
    patients = pd.DataFrame(rows)
    patients.to_csv(output / 'intervention_patients.csv', index=False)
    summary = bootstrap_summary(patients, args.bootstraps, args.seed)
    summary.to_csv(output / 'intervention_summary.csv', index=False)
    return images, visual, manifest.iloc[sample].pid, summary


def draw(args, nodes, edges, names, visual, summary, output):
    fig = plt.figure(figsize=(15, 11), layout='constrained')
    grid = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(grid[0, 0])
    graph = nx.Graph()
    graph.add_nodes_from(nodes.neuron)
    graph.add_edges_from(zip(edges.source, edges.target))
    positions = nx.circular_layout(graph)
    labels = {int(row.neuron): f'n{row.neuron}\n{row.label}\nC{row.community}' for row in nodes.itertuples()}
    nx.draw_networkx(graph, positions, ax=ax, labels=labels, node_color=['#edb967' if seed else '#a4d5df' for seed in nodes.is_seed], node_size=1500, font_size=7, edge_color='#91a4ae', width=[1 + 2 * r.train_r for r in edges.itertuples()])
    ax.margins(.18)
    ax.set_title('A. Training-selected graph; labels use training IoU\nOrange: query seeds; edge weights and evidence in edges.csv')
    ax.axis('off')
    if visual is None:
        ax = fig.add_subplot(grid[0, 1])
        ax.text(.05, .65, 'B. Visual evidence not refreshed\n\nRun without --cached-only for exact-channel\nMRI/mask overlays and new interventions.\n\nSaved supporting patient IDs and overlaps\nare exported in supporting_examples.csv.', va='top')
        ax.axis('off')
    else:
        images, overlays, pid = visual
        axes = grid[0, 1].subgridspec(len(overlays), 3)
        for i, item in enumerate(overlays):
            for phase in range(2):
                ax = fig.add_subplot(axes[i, phase])
                ax.imshow(images[phase], cmap='gray')
                if item['active'].any() and not item['active'].all():
                    ax.contour(item['active'], levels=[.5], colors=['cyan'], linewidths=.7)
                ax.set_title(f"n{item['neuron']} / {pid} / {['ED', 'ES'][phase]}", fontsize=8)
                ax.axis('off')
            ax = fig.add_subplot(axes[i, 2])
            rgb = np.zeros((*item['target'].shape, 3))
            rgb[item['target']] = [1, 0, 1]
            rgb[item['active']] = [0, 1, 1]
            rgb[item['active'] & item['target']] = 1
            ax.imshow(rgb)
            ax.set_title(f"{item['concept']}: IoU={item['iou']:.4f}\nmagenta: mask; cyan: active; white: both", fontsize=8)
            ax.axis('off')
    ax = fig.add_subplot(grid[1, 0])
    train = nodes[[f'train_{name}' for name in names]].to_numpy()
    evaluated = nodes[[f'eval_{name}' for name in names]].to_numpy()
    matrix = np.concatenate([train, evaluated], axis=1)
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=0)
    ax.set(yticks=range(len(nodes)), yticklabels=[f'n{i}' for i in nodes.neuron], xticks=range(2 * len(names)), xticklabels=[f'{split}\n{name}' for split in ['train', args.split] for name in names], title='C. Complete concept profiles (IoU)')
    fig.colorbar(im, ax=ax, shrink=.7)
    ax = fig.add_subplot(grid[1, 1])
    if summary.empty:
        ax.text(.05, .7, 'D. Exact selected-set ablation unavailable\nNo result from a different channel set substituted.', va='top')
        ax.axis('off')
    else:
        values = summary[summary.metric == 'kl'].reset_index(drop=True)
        ax.bar(range(len(values)), values['mean'], color=['#218c80' if s == 'displayed_subgraph' else '#a4b5c8' for s in values.strategy])
        for i, row in values.iterrows():
            if np.isfinite(row.ci_low):
                ax.plot([i, i], [row.ci_low, row.ci_high], color='black')
        ax.set(xticks=range(len(values)), xticklabels=values.strategy, ylabel='KL(original || ablated)', title='D. Exact-set interventions\nQuery seeds use fewer channels; other controls match displayed size')
        ax.tick_params(axis='x', rotation=25)
    retained = int((nodes.loc[nodes.is_seed, f'eval_{args.concept}'] >= args.gamma).sum())
    fig.suptitle(f'{args.concept}: concept-centred explanation | {args.split}\n{retained}/{int(nodes.is_seed.sum())} seeds retain the fixed concept threshold; coactivation is not causal flow', fontsize=14)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'explanation.{suffix}', dpi=170)
    plt.close(fig)


def run(args):
    directory = args.layer_dir.resolve()
    protocol = json.loads((directory.parent.parent / 'protocol.json').read_text())
    if args.batch_size is None:
        args.batch_size = int(protocol['batch_size'])
    with np.load(directory / 'representations.npz') as data:
        archive = dict(data)
    names = archive['concepts'].tolist()
    if args.concept not in names:
        raise ValueError(f'Choose a concept from {names}')
    S, R, E = archive['S'], archive['R_train'], archive[f'S_{args.split}']
    gamma, tau = protocol['gamma'], protocol['tau']
    args.gamma = gamma
    seeds, selected, pairs = select_graph(S, R, names.index(args.concept), gamma, tau, args.seeds, args.neighbors)
    if E.shape != S.shape or not np.isfinite(E).all():
        raise ValueError('Evaluation profiles disagree in shape or contain nonfinite values.')
    communities = pd.read_csv(directory / 'communities_train.csv').set_index('neuron').community.to_dict()
    rows = []
    for i in selected:
        labels = [name for c, name in enumerate(names) if S[i, c] >= gamma]
        row = dict(neuron=i, is_seed=i in seeds, community=int(communities.get(i, -1)), label='/'.join(labels) or 'unlabelled')
        row.update({f'{split}_{name}': float(values[i, c]) for split, values in [('train', S), ('eval', E)] for c, name in enumerate(names)})
        rows.append(row)
    nodes = pd.DataFrame(rows)
    edges = pd.DataFrame([dict(source=i, target=j, train_r=R[i, j], eval_r=archive[f'R_{args.split}'][i, j], eval_passes_threshold=bool(archive[f'R_{args.split}'][i, j] >= tau)) for i, j in pairs], columns=['source', 'target', 'train_r', 'eval_r', 'eval_passes_threshold'])
    root = args.output or directory / 'concept_explanations'
    output = root / f'{args.concept}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")}'
    output.mkdir(parents=True, exist_ok=False)
    nodes.to_csv(output / 'nodes.csv', index=False)
    edges.to_csv(output / 'edges.csv', index=False)
    manifest = pd.read_csv(directory.parent.parent / f'{args.split}_samples.csv', dtype={'pid': str})
    if manifest.pid.duplicated().any() or manifest.sample_index.tolist() != list(range(len(manifest))):
        raise ValueError('Require unique patients and contiguous ordered sample indices.')
    evidence = pd.read_csv(directory / f'concept_evidence_{args.split}.csv', dtype={'pid': str})
    evidence = evidence[evidence.neuron.isin(selected)].copy()
    expected = evidence.sample_index.map(manifest.set_index('sample_index').pid)
    if not expected.equals(evidence.pid):
        raise ValueError('Supporting evidence IDs disagree with the patient manifest.')
    evidence.to_csv(output / 'supporting_examples.csv', index=False)
    visual, summary = None, pd.DataFrame()
    if args.cached_only:
        cached = exact_cached_effect(directory, args.split, selected)
        cached.to_csv(output / 'exact_cached_intervention.csv', index=False)
        if not cached.empty:
            summary = pd.DataFrame([dict(strategy='displayed_subgraph', metric=metric, mean=float(cached.iloc[0][metric]), ci_low=np.nan, ci_high=np.nan) for metric in ['kl', 'flip']])
    else:
        images, overlays, pid, summary = refresh(args, directory, protocol, archive, manifest, selected, seeds, output)
        visual = (images, overlays, pid)
    draw(args, nodes, edges, names, visual, summary, output)
    retained = int((E[seeds, names.index(args.concept)] >= gamma).sum())
    report = [f'# {args.concept}: one-layer concept explanation', '', f'Model: {directory.parent.name}; layer: {directory.name}; evaluation split: {args.split}.', '', f'{len(seeds)} seeds selected by training IoU >= {gamma}; up to {args.neighbors} strongest training neighbours per seed at R >= {tau}. All eligible induced edges among the displayed nodes are retained. Selection never uses evaluation scores.', '', f'Seed channels: {seeds}. Displayed/ablated channel set: {selected}.', '', f'{retained}/{len(seeds)} seed associations remain above the fixed threshold on evaluation patients. {int(edges.eval_passes_threshold.sum())}/{len(edges)} displayed edges remain above the correlation threshold. These are descriptive threshold checks, not significance tests or proof of reliability.', '', '![Explanation](explanation.png)', '', '## Traceability', '', '- nodes.csv: complete training/evaluation concept profiles and fixed training communities.', '- edges.csv: exact channel pairs and training/evaluation correlations; undirected statistical associations.', '- supporting_examples.csv: saved highest-overlap examples per neuron/concept, not representative sampling. Empty evidence is not proof of absence.', '', '## Intervention evidence', '']
    if summary.empty:
        report.append('Not measured for this exact displayed set. Run without --cached-only. No existing result from another set has been substituted.')
    else:
        report.extend(['```', summary.to_string(index=False), '```', '', 'KL is prediction sensitivity, not accuracy loss. Flip is a fraction. Positive margin_drop means the original predicted-class versus original runner-up margin decreased; that class pair is fixed during intervention. Query-seed ablation uses fewer channels and is not an equal-budget control. Random draws are averaged within patient before bootstrap; pointwise intervals condition on this model, selection and sampled controls. No matched-activation or multiple-testing claim is made.'])
    report.extend(['', '## Generated description', '', f'The training query retrieves {len(seeds)} {args.concept}-associated seed channels in {nodes.loc[nodes.is_seed, "community"].nunique()} training communities. Their selected functional neighbourhood contains {len(selected)} channels and {len(edges)} coactivation connections.'])
    if pairs:
        strongest = edges.sort_values('train_r', ascending=False).iloc[0]
        report.append(f'The strongest displayed connection is n{int(strongest.source)}–n{int(strongest.target)}: training correlation {strongest.train_r:.3f}, evaluation correlation {strongest.eval_r:.3f}. Its endpoint concept profiles are available in nodes.csv; unlabelled endpoints are not assigned an anatomical meaning.')
    report.append(f'Only {retained} of the {len(seeds)} queried seed associations retain the fixed IoU threshold on evaluation patients. Consequently, the graph is a training-associated concept explanation with explicitly reported semantic limitations, not proof of a reliable anatomical mechanism.')
    if not summary.empty:
        effect = summary[(summary.strategy == 'displayed_subgraph') & (summary.metric == 'kl')].iloc[0]
        report.append(f'Removing exactly the {len(selected)} displayed channels produces mean prediction KL {effect["mean"]:.6f}. This measures intervention sensitivity; it does not validate the semantic labels or imply that the graph edges transmit information.')
    report.extend(['', '## Limits', '', 'Within-layer evidence tracing only. Correlation edges are not computational pathways. Threshold-passing labels are vocabulary-dependent associations, not proven selective detectors. Evaluation patients have been inspected previously: this is exploratory. Training communities are identifiers, not independently established semantic modules.', '', 'Full mode uses one seeded patient independent of overlap for exact-channel overlays, on the shared ED/ES grid. The figure does not show every selected channel; raw selected examples are explicit in overlay_selection.json. Cached-only mode has no refreshed overlays.', '', f'Source run: {directory}', ''])
    (output / 'EXPLANATION.md').write_text('\n'.join(report))
    provenance = dict(source=str(directory), arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, seeds=seeds, displayed_neurons=selected, tau=tau, gamma=gamma, representation_sha256=digest(directory / 'representations.npz'), script_sha256=digest(__file__), protocol=protocol)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'Saved explanation: {output}', flush=True)


def self_test():
    S = np.array([[.8, 0], [0, .9], [.6, 0], [0, 0]])
    R = np.array([[1, .7, .1, 0], [.7, 1, .6, 0], [.1, .6, 1, 0], [0, 0, 0, 1]])
    seeds, nodes, edges = select_graph(S, R, 0, .5, .3, 1, 1)
    assert seeds == [0] and nodes == [0, 1] and edges == [(0, 1)]
    assert select_graph(S, R, 0, .5, .9, 1, 1)[1:] == ([0], [])
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        pd.DataFrame([dict(neurons='[0, 2]', kl=1), dict(neurons='[1, 0]', kl=2)]).to_csv(p / 'ablation_selections_test.csv', index=False)
        assert exact_cached_effect(p, 'test', [0, 1]).iloc[0].kl == 2
        assert exact_cached_effect(p, 'test', [0, 1, 2]).empty
    print('Self-test passed: selection, isolation and exact intervention matching.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--concept', default='MYO')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--neighbors', type=int, default=2)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, help='Defaults to the saved run batch size for reproducible activation checks.')
    parser.add_argument('--random-repeats', type=int, default=20)
    parser.add_argument('--bootstraps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cached-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if min(args.seeds, args.random_repeats, args.bootstraps) < 1 or args.neighbors < 0 or (args.batch_size is not None and args.batch_size < 1):
            parser.error('Counts must be positive; neighbors may be zero.')
        run(args)
