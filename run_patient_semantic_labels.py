"""Patient-specific concept–neuron associations; functional graph stays frozen.

Subtract training spatial templates, then ask whether each channel's residual map
matches its own patient's anatomical residual better than other patients of the
same pathology. Labels are training-selected; validation/test only assess them.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scag.relations import fdr_bh
from scag.research import cache_patients, file_sha256, validate_manifests
from run_semantic_grounding import DEFAULT_RUN, spatial_ap


def residual_unit_maps(values, training_template):
    """Remove fixed location and patient-wide level; retain spatial deviations."""
    if values.ndim != 3 or training_template.shape != values.shape[1:] or not torch.isfinite(values).all():
        raise ValueError('Require finite patient x channel x pixel maps and a matching training template.')
    residual = values - training_template[None]
    residual = residual - residual.mean(dim=-1, keepdim=True)
    norm = residual.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-8, residual / norm.clamp_min(1e-8), 0)


def matched_evidence(activation, target, groups, permutations, seed):
    """Within-stratum patient permutation, keeping channel/concept pairs together."""
    if activation.ndim != 3 or target.ndim != 3 or activation.shape[0] != target.shape[0] or activation.shape[2] != target.shape[2]:
        raise ValueError('Activation and target maps must share patient and pixel axes.')
    if not torch.isfinite(activation).all() or not torch.isfinite(target).all():
        raise ValueError('Non-finite residual maps.')
    n = len(activation)
    groups = np.asarray(groups)
    if len(groups) != n or pd.isna(groups).any():
        raise ValueError('Require one nonmissing stratum per patient.')
    blocks = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    if min(map(len, blocks)) < 2:
        raise ValueError('Every pathology stratum needs at least two patients for exchangeable controls.')
    # K x C x patient x paired-patient; all spatial evidence is retained before permutations.
    similarity = torch.einsum('ikp,jcp->kcij', activation, target)
    index = torch.arange(n, device=activation.device)
    own = similarity[:, :, index, index]
    expected = torch.empty_like(own)
    for block in blocks:
        b = torch.as_tensor(block, device=activation.device)
        expected[:, :, b] = similarity[:, :, b[:, None], b[None, :]].mean(dim=-1)
    observed = own.mean(dim=-1)
    effect = (own - expected).mean(dim=-1)
    count = torch.zeros_like(observed)
    rng = np.random.default_rng(seed)
    for start in range(0, permutations, 64):
        assignments = np.tile(np.arange(n), (min(64, permutations - start), 1))
        for row in assignments:
            for block in blocks:
                row[block] = rng.permutation(block)
        assignments = torch.as_tensor(assignments, device=activation.device)
        null = similarity[:, :, index[None, :], assignments].mean(dim=-1)
        count += (null >= observed[:, :, None] - 1e-7).sum(dim=-1)
    return dict(effect=effect.cpu().numpy(), p=((count + 1) / (permutations + 1)).cpu().numpy(), patient_effect=(own - expected).permute(2, 0, 1).cpu().numpy(), matched_cosine=own.mean(dim=-1).cpu().numpy(), null_cosine=expected.mean(dim=-1).cpu().numpy())


def load_maps(args, output, protocol, manifests, archive):
    from scag.config import SCAGConfig
    from scag.data import load_datasets
    from scag.models import build_model
    from scag.activations import extract_activations
    from scag.semantics import concept_masks
    directory = args.layer_dir.resolve()
    cfg = SCAGConfig(dataset=protocol['dataset'], model_seed=protocol.get('model_seed', 0))
    checkpoint = json.loads((directory.parent / 'checkpoint.json').read_text())
    path = cfg.model_root / directory.parent.name / cfg.formulation / cfg.initialization / f'seed_{cfg.model_seed}' / 'best_val_mcc.pt'
    if path.resolve() != Path(checkpoint['path']).resolve() or file_sha256(path) != checkpoint['sha256']:
        raise ValueError('Checkpoint differs from the frozen source run.')
    model, _, _ = build_model(directory.parent.name, cfg)
    model.to(args.device).eval()
    result = {}
    for split, dataset in zip(['train', 'val', 'test'], load_datasets(cfg)[:3]):
        manifest = manifests[split]
        mapping = {str(pid): i for i, pid in enumerate(dataset.metadata.pid)}
        if len(mapping) != len(dataset):
            raise ValueError('Dataset has duplicate patient IDs.')
        loader = cache_patients(torch.utils.data.Subset(dataset, [mapping[pid] for pid in manifest.pid]), manifest, protocol['batch_size'])
        acts, segments = extract_activations(model, loader, model.get_submodule(directory.name), args.device)
        if not np.allclose(acts.mean((2, 3)).numpy(), archive[f'pooled_{split}'], rtol=1e-4, atol=1e-5):
            raise ValueError(f'{split}: refreshed activations disagree with source run.')
        masks, names = concept_masks(segments, protocol['semantic_level'])
        if names != archive['concepts'].tolist():
            raise ValueError('Concept ordering differs from source run.')
        result[split] = (acts, masks)
        manifest.to_csv(output / f'{split}_samples.csv', index=False)
        np.savez_compressed(output / f'spatial_maps_{split}.npz', activations=acts.numpy(), masks=masks.numpy(), pid=manifest.pid.to_numpy(dtype=str))
        print(f'{split}: verified and cached {len(acts)} patient maps', flush=True)
    return result, checkpoint


def analyse(args, output, maps, archive, manifests, protocol):
    names = archive['concepts'].tolist()
    h, w = maps['train'][1].shape[-2:]
    coordinates = np.random.default_rng(args.seed).choice(h * w, min(args.pixels, h * w), replace=False)
    sampled, targets, aps = {}, {}, {}
    for split, (acts, masks) in maps.items():
        chunks = []
        for i in range(len(acts)):
            up = torch.nn.functional.interpolate(acts[i:i + 1].to(args.device), size=(h, w), mode='bilinear', align_corners=False)[0]
            chunks.append(up.flatten(1)[:, coordinates])
        sampled[split] = torch.stack(chunks)
        targets[split] = masks.to(args.device).float().flatten(2)[:, :, coordinates]
        aps[split] = np.stack([spatial_ap(x, y).cpu().numpy() for x, y in zip(sampled[split], targets[split])])
    activation_template, mask_template = sampled['train'].mean(dim=0), targets['train'].mean(dim=0)
    np.savez_compressed(output / 'calibration.npz', sampled_pixels=coordinates, activation_template=activation_template.cpu().numpy(), mask_template=mask_template.cpu().numpy())
    results, selected, rows, coverage, comparisons = {}, None, [], [], []
    old = archive['S_train'] >= protocol['gamma']
    for split in ['train', 'val', 'test']:
        # Training leave-one-out residuals differ only by a positive scalar, cancelled by unit normalization.
        x = residual_unit_maps(sampled[split], activation_template)
        y = residual_unit_maps(targets[split], mask_template)
        print(f'{split}: testing patient-specific associations ({args.permutations} within-pathology permutations)', flush=True)
        result = matched_evidence(x, y, manifests[split].pathology.to_numpy(), args.permutations, args.seed)
        if split == 'train':
            labels = manifests[split].pathology.to_numpy()
            mismatch = np.arange(len(labels))
            rng = np.random.default_rng(args.seed + 991)
            for label in np.unique(labels):
                block = np.flatnonzero(labels == label)
                mismatch[block] = rng.permutation(block)
            negative = matched_evidence(x, y[mismatch], labels, args.permutations, args.seed + 1)
            negative_q = fdr_bh(negative['p'].ravel()).reshape(negative['p'].shape)
            negative_summary = dict(rule='one seeded random within-pathology reassignment; chance fixed points allowed', supported_edges=int(((negative_q <= args.fdr) & (negative['effect'] > 0)).sum()), total_pairs=int(negative_q.size), minimum_q=float(negative_q.min()))
            (output / 'negative_control.json').write_text(json.dumps(negative_summary, indent=2))
            print(f'Random patient-pairing control: {negative_summary}', flush=True)
        q = fdr_bh(result['p'].ravel()).reshape(result['p'].shape)
        supported = (q <= args.fdr) & (result['effect'] > 0)
        if split == 'train':
            selected = supported.copy()
        frozen_q = np.full_like(q, np.nan)
        if selected.any():
            frozen_q[selected] = fdr_bh(result['p'][selected])
        replicated = selected & (frozen_q <= args.fdr) & (result['effect'] > 0)
        result.update(q=q, frozen_q=frozen_q, ap=aps[split])
        results[split] = result
        np.savez_compressed(output / f'evidence_{split}.npz', **result, pid=manifests[split].pid.to_numpy(dtype=str), concepts=names)
        for i in range(len(old)):
            for c, name in enumerate(names):
                rows.append(dict(split=split, neuron=i, concept=name, effect=result['effect'][i, c], p=result['p'][i, c], q_all=q[i, c], q_training_family=frozen_q[i, c], training_edge=bool(selected[i, c]), replicated_training_edge=bool(replicated[i, c]), original_training_edge=bool(old[i, c]), mean_ap=np.nanmean(aps[split][:, i, c]), original_iou=archive[f'S_{split}'][i, c]))
        coverage.append(dict(split=split, original_training_channels=int(old.any(axis=1).sum()), new_training_channels=int(selected.any(axis=1).sum()), additional_training_channels=int((selected.any(axis=1) & ~old.any(axis=1)).sum()), original_training_edges=int(old.sum()), new_training_edges=int(selected.sum()), positive_frozen_edges=int((selected & (result['effect'] > 0)).sum()), corroborated_frozen_edges=int(replicated.sum()), corroborated_frozen_channels=int(replicated.any(axis=1).sum()), original_iou_retained_edges=int((old & (archive[f'S_{split}'] >= protocol['gamma'])).sum())))
        if split == 'train':
            continue
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(len(x), size=(args.bootstraps, len(x)))
        with np.load(args.layer_dir / f'patient_concepts_{split}.npz') as data:
            iou = data['per_patient_iou']
        assert iou.shape == aps[split].shape
        for c, name in enumerate(names):
            k = min(args.top_k, len(old))
            new = np.argsort(-results['train']['effect'][:, c], kind='stable')[:k]
            previous = np.argsort(-archive['S_train'][:, c], kind='stable')[:k]
            for metric, values in [('patient_specific_effect', result['patient_effect']), ('spatial_ap', aps[split]), ('original_iou', iou)]:
                a, b = np.nanmean(values[:, new, c], axis=1), np.nanmean(values[:, previous, c], axis=1)
                delta = a - b
                lo, hi = np.nanquantile(np.nanmean(delta[draws], axis=1), [.025, .975])
                comparisons.append(dict(split=split, concept=name, metric=metric, k=k, new_mean=np.nanmean(a), original_mean=np.nanmean(b), paired_difference=np.nanmean(delta), ci_low=lo, ci_high=hi, new_channels=json.dumps(new.tolist()), original_channels=json.dumps(previous.tolist())))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'semantic_edges.csv', index=False)
    coverage, comparisons = pd.DataFrame(coverage), pd.DataFrame(comparisons)
    coverage.to_csv(output / 'coverage.csv', index=False)
    comparisons.to_csv(output / 'ranking_comparison.csv', index=False)
    export_graph(output, table, archive, names, protocol)
    plot_results(output, table, results, maps, names, selected)
    report = '''# Patient-specific semantic labelling

## What changed

Concept–neuron links now test patient-specific spatial correspondence. The CNN and functional coactivation graph are unchanged. No community analysis or graph-based label propagation is used.

Remove the training mean spatial activation map and training mean concept mask. Remove each residual map's spatial mean and normalize its length. For each neuron–concept pair, compare the cosine similarity of correctly matched patients with the average similarity under within-pathology patient permutations. This asks whether responses track an individual's anatomy beyond fixed location, global response level and broad pathology. Training leave-one-out residuals are a positive multiple of ordinary training-centred residuals and therefore give identical normalized vectors. Zero residuals have zero similarity and cannot acquire an edge merely from a constant response.

A training edge requires positive matched-minus-permuted similarity and Benjamini–Hochberg q <= the configured FDR across all neuron–concept pairs. Weight is the positive training effect. Each concept–neuron pair can be present independently. All neurons remain in the graph; weak evidence remains unresolved. This is a patient-specific spatial association label, not a segmentation detector, causal role or exclusive concept identity. Signed residuals include above/below-template responses.

## Controls and limitations

Permutations retain spatial structure and match within pathology; validity assumes patients are exchangeable within those strata. Acquisition/site and other unrecorded confounders are not eliminated. BH control depends on its dependence assumptions. P-values have finite Monte Carlo resolution. Shared uniformly sampled pixels are independent of masks/activations and identical across patients. No threshold is tuned on validation/test. Validation/test only corroborate the frozen training edges, with BH within that frozen family; q_all is separately exploratory. Low validation sample size and few distinct permutations limit power.

The original IoU association criterion and the new residual criterion answer different questions, so higher coverage is not by itself an improvement. Top-k rankings are training-fixed, include unsupported pairs if needed, and are evaluated on both metrics. Paired bootstrap intervals are pointwise, conditional on templates/normalization/null reference, and unadjusted for multiple comparisons; they are descriptive, not a nested inferential test. Existing test data have already been inspected. No prediction ablations are run.

## Coverage

```
''' + coverage.to_string(index=False) + '\n```\n\n## Fixed-budget test comparison\n\n```\n' + comparisons.query('split == "test"').drop(columns=['new_channels', 'original_channels']).to_string(index=False) + '\n```\n\n## Outputs\n\n- joint_graph.graphml: typed concept/neuron nodes, unchanged functional edges, and training-selected semantic edges with validation/test evidence.\n- semantic_edges.csv: every candidate association, including unresolved pairs.\n- evidence_*.npz: patient effects and raw AP, with patient IDs.\n- spatial_maps_*.npz and calibration.npz: exact refreshed maps and sampled pixel/template evidence.\n- comparison.png/pdf: coverage and evaluation effects.\n- examples.png/pdf: training-selected channels on median and strongest test residual-match patients; these are selected examples, not an independent validation. Red/blue residuals show deviations from the training template, not ordinary activation heatmaps.\n'
    report += '\n## Mismatched-patient control\n\n```json\n' + (output / 'negative_control.json').read_text() + '\n```\nThis single random reassignment is a sanity check, not a complete empirical false-positive calibration.\n'
    by_concept = table.groupby(['split', 'concept'], sort=False).agg(training_edges=('training_edge', 'sum'), corroborated_edges=('replicated_training_edge', 'sum'))
    by_concept.to_csv(output / 'concept_coverage.csv')
    report += '\n## Coverage by concept\n\n```\n' + by_concept.to_string() + '\n```\n'
    (output / 'RESULTS.md').write_text(report)
    print(coverage.to_string(index=False), flush=True)
    print(comparisons.query('split == "test"').drop(columns=['new_channels', 'original_channels']).to_string(index=False), flush=True)


def export_graph(output, table, archive, names, protocol):
    import networkx as nx
    graph = nx.Graph(description='Frozen coactivation plus training-selected patient-specific spatial semantic associations; not causal pathways')
    graph.add_nodes_from([(f'n{i}', dict(kind='neuron', channel=int(i))) for i in range(len(archive['S_train']))])
    graph.add_nodes_from([(f'concept:{name}', dict(kind='concept', name=name)) for name in names])
    R = archive['R_train']
    for i, j in zip(*np.where(np.triu((R >= protocol['tau']) & (R > 0), 1))):
        graph.add_edge(f'n{i}', f'n{j}', kind='coactivation', weight=float(R[i, j]))
    for row in table.query('split == "train" and training_edge').itertuples():
        attributes = dict(kind='semantic_association', weight=float(row.effect), train_q=float(row.q_all), evidence='patient-specific residual spatial correspondence')
        for split in ['val', 'test']:
            evaluated = table[(table.split == split) & (table.neuron == row.neuron) & (table.concept == row.concept)].iloc[0]
            attributes.update({f'{split}_effect': float(evaluated.effect), f'{split}_q': float(evaluated.q_training_family)})
        graph.add_edge(f'concept:{row.concept}', f'n{row.neuron}', **attributes)
    nx.write_graphml(graph, output / 'joint_graph.graphml')


def plot_results(output, table, results, maps, names, selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    for c, (ax, name) in enumerate(zip(axes[0], names)):
        train, test = results['train']['effect'][:, c], results['test']['effect'][:, c]
        ax.scatter(train, test, c=np.where(selected[:, c], 'tab:blue', 'lightgray'), s=12)
        ax.axhline(0, color='black', linewidth=.6)
        ax.axvline(0, color='black', linewidth=.6)
        ax.set(title=f'{name}: {selected[:, c].sum()} training edges', xlabel='Training patient-specific effect', ylabel='Test patient-specific effect')
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'comparison.{extension}', dpi=180)
    plt.close(fig)
    train_acts, train_masks = maps['train']
    test_acts, test_masks = maps['test']
    shape = test_masks.shape[-2:]
    fig, axes = plt.subplots(len(names) * 2, 3, figsize=(10, 5 * len(names)), squeeze=False)
    for c, name in enumerate(names):
        eligible = np.flatnonzero(selected[:, c])
        neuron = int(eligible[np.argmax(results['train']['effect'][eligible, c])]) if len(eligible) else int(np.argmax(results['train']['effect'][:, c]))
        order = np.argsort(results['test']['patient_effect'][:, neuron, c], kind='stable')
        for offset, (sample, label) in enumerate([(order[len(order) // 2], 'median'), (order[-1], 'strongest')]):
            row = c * 2 + offset
            activation = torch.nn.functional.interpolate(test_acts[sample:sample + 1, neuron:neuron + 1], size=shape, mode='bilinear', align_corners=False)[0, 0].numpy()
            template = torch.nn.functional.interpolate(train_acts[:, neuron:neuron + 1].mean(dim=0, keepdim=True), size=shape, mode='bilinear', align_corners=False)[0, 0].numpy()
            residual = activation - template
            residual -= residual.mean()
            mask = test_masks[sample, c].numpy()
            mask_residual = mask.astype(float) - train_masks[:, c].float().mean(dim=0).numpy()
            mask_residual -= mask_residual.mean()
            limit = max(float(np.abs(residual).max()), 1e-8)
            axes[row, 0].imshow(activation, cmap='magma')
            axes[row, 1].imshow(residual, cmap='coolwarm', vmin=-limit, vmax=limit)
            axes[row, 2].imshow(mask_residual, cmap='coolwarm', vmin=-1, vmax=1)
            if mask.any() and not mask.all():
                for ax in axes[row]:
                    ax.contour(mask, levels=[.5], colors='black', linewidths=.7)
            for ax, title in zip(axes[row], ['Activation', 'Activation residual', 'Concept residual']):
                ax.set_title(f'{name} n{neuron}, {label} sample {sample}\n{title}', fontsize=9)
                ax.axis('off')
    fig.suptitle('Training-selected channels; selected test examples; colours independently scaled\nContours: actual concept mask. These are spatial maps, not MRI images.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'examples.{extension}', dpi=180)
    plt.close(fig)


def run(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use cinema with GPU access or explicitly --device cpu.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.layer_dir.resolve().parent.parent
    protocol = json.loads((root / 'protocol.json').read_text())
    manifests = {s: pd.read_csv(root / f'{s}_samples.csv', dtype={'pid': str}) for s in ['train', 'val', 'test']}
    validate_manifests(manifests)
    for frame in manifests.values():
        if 'pathology' not in frame or frame.pathology.isna().any():
            raise ValueError('Pathology strata are required; no silent unstratified fallback.')
    archive = dict(np.load(args.layer_dir / 'representations.npz'))
    output = args.layer_dir.resolve() / 'patient_semantic_labels' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True)
    maps, checkpoint = load_maps(args, output, protocol, manifests, archive)
    with torch.inference_mode():
        analyse(args, output, maps, archive, manifests, protocol)
    provenance = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, protocol=protocol, checkpoint=checkpoint, script_sha256=file_sha256(__file__), source_representation_sha256=file_sha256(args.layer_dir / 'representations.npz'))
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'Saved: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--pixels', type=int, default=2048)
    parser.add_argument('--permutations', type=int, default=9999)
    parser.add_argument('--bootstraps', type=int, default=2000)
    parser.add_argument('--fdr', type=float, default=.05)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.pixels, args.top_k) < 1 or min(args.permutations, args.bootstraps) < 100 or not 0 < args.fdr < 1:
        parser.error('Require positive pixels/top-k, at least 100 repetitions, and FDR in (0,1).')
    run(args)
