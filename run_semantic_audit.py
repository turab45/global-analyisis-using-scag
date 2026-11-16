"""Audit frozen SENCoG runs without refitting thresholds or intervention selections."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scag.semantic_audit import graph_trace, paired_tradeoff, reliability


def representative_indices(manifest, seed):
    rng = np.random.default_rng(seed)
    if 'pathology' in manifest:
        return [int(rng.choice(group.index)) for _, group in manifest.groupby('pathology', sort=True)]
    return sorted(rng.choice(len(manifest), min(5, len(manifest)), replace=False).tolist())


def refresh(directory, output, protocol, manifests, archive, device, seed):
    import torch
    from scag.config import SCAGConfig
    from scag.data import load_datasets
    from scag.models import build_model
    from scag.activations import extract_activations
    from scag.semantics import concept_masks, concept_profiles
    from scag.research import cache_patients, file_sha256
    from scag.relation_plots import plot_overlays
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use cinema outside a restricted sandbox.')
    torch.set_num_threads(4)
    cfg = SCAGConfig(dataset=protocol['dataset'])
    checkpoint = json.loads((directory.parent / 'checkpoint.json').read_text())
    if file_sha256(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Checkpoint changed since the original run.')
    model, _, _ = build_model(directory.parent.name, cfg)
    model.to(device).eval()
    datasets = dict(zip(['train', 'val', 'test'], load_datasets(cfg)[:3]))
    chosen = []
    for split, dataset in datasets.items():
        manifest = manifests[split]
        mapping = {str(pid): i for i, pid in enumerate(dataset.metadata.pid)}
        dataset = torch.utils.data.Subset(dataset, [mapping[pid] for pid in manifest.pid])
        loader = cache_patients(dataset, manifest, protocol['batch_size'])
        acts, segments = extract_activations(model, loader, model.get_submodule(directory.name), device)
        masks, names = concept_masks(segments, protocol['semantic_level'])
        profiles = concept_profiles(acts, masks, thresholds=archive['thresholds'], device=device)
        if not np.allclose(profiles['S'], archive[f'S_{split}'], atol=1e-6, rtol=1e-4):
            raise ValueError(f'{split}: refreshed semantics disagree with the saved run; investigate preprocessing before continuing.')
        np.savez_compressed(output / f'patient_concepts_{split}.npz', **profiles)
        for index in representative_indices(manifest, seed):
            chosen.append({'split': split, 'sample_index': index, 'pid': manifest.iloc[index].pid, 'sampling': 'one seeded random patient per pathology; not overlap-selected'})
            plot_overlays(acts, masks, profiles, archive['S'], names, loader, manifest, output, split, sample_index=index)
        del acts, segments, masks
    pd.DataFrame(chosen).to_csv(output / 'representative_patients.csv', index=False)


def plot_tradeoff(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    concepts, budgets = sorted(summary.concept.unique()), sorted(summary.budget.unique())
    for horizontal in ['iou', 'selectivity']:
        fig, axes = plt.subplots(len(concepts), len(budgets), figsize=(15, 10), squeeze=False, layout='constrained')
        for i, concept in enumerate(concepts):
            for j, budget in enumerate(budgets):
                ax = axes[i, j]
                group = summary[(summary.concept == concept) & (summary.budget == budget)]
                for strategy, color in [('functional', '#e58b39'), ('joint', '#148f80'), ('semantic', '#8c7adc'), ('random', '#89929b')]:
                    rows = group[group.strategy == strategy].set_index('metric')
                    x, y = rows.loc[horizontal], rows.loc['kl']
                    ax.plot([x.ci_low, x.ci_high], [y.estimate] * 2, color=color)
                    ax.plot([x.estimate] * 2, [y.ci_low, y.ci_high], color=color)
                    ax.scatter(x.estimate, y.estimate, color=color, s=30)
                    ax.annotate(strategy, (x.estimate, y.estimate), xytext=(4, 4), textcoords='offset points', fontsize=7)
                ax.set(title=f'{concept} | {budget:.0%}', xlabel='Concept association (IoU)' if horizontal == 'iou' else 'Vocabulary-relative selectivity margin', ylabel='Prediction KL')
                ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
                ax.grid(alpha=.15)
        fig.suptitle('Frozen selections: paired patient bootstrap, pointwise 95% intervals\nError bars are marginal intervals, not a joint confidence region')
        for extension in ['png', 'pdf']:
            fig.savefig(output / f'tradeoff_{horizontal}.{extension}', dpi=170)
        plt.close(fig)


def analyze(directory, repeats=2000, seed=42, refresh_evidence=False, device='cpu'):
    directory = directory.resolve()
    root = directory.parent.parent
    protocol = json.loads((root / 'protocol.json').read_text())
    output = directory / 'semantic_reliability'
    output.mkdir(exist_ok=True)
    manifests = {split: pd.read_csv(root / f'{split}_samples.csv', dtype={'pid': str}) for split in ['train', 'val', 'test']}
    with np.load(directory / 'representations.npz') as saved:
        archive = dict(saved)
    if refresh_evidence:
        refresh(directory, output, protocol, manifests, archive, device, seed)
    names = archive['concepts'].tolist()
    summaries, contrasts, diagnostics, coverage, missing = [], [], [], [], []
    for split, manifest in manifests.items():
        path = output / f'patient_concepts_{split}.npz'
        if not path.exists():
            path = directory / path.name
        with np.load(path) as saved:
            profiles = dict(saved)
        if len(profiles['intersection']) != len(manifest) or manifest.pid.duplicated().any():
            raise ValueError('Patient archive/manifest mismatch.')
        patient, intervals = reliability(profiles, manifest, protocol['gamma'], repeats, seed)
        if 'activation_area' not in profiles:
            missing.append(split)
        intervals['split'] = split
        diagnostics.append(patient)
        coverage.append(intervals)
        if split == 'train':
            continue
        selections = pd.read_csv(directory / f'ablation_selections_{split}.csv')
        patients = pd.read_csv(directory / f'ablation_patients_{split}.csv.gz', dtype={'pid': str})
        expected = patients.sample_index.map(manifest.set_index('sample_index').pid)
        if not expected.equals(patients.pid):
            raise ValueError('Ablation patient IDs disagree with manifest.')
        summary, delta = paired_tradeoff(selections, patients, profiles, names, repeats, seed)
        summary['split'], delta['split'] = split, split
        quality = selections.query("strategy == 'matched'").groupby(['concept', 'budget']).match_valid.mean().rename('matched_valid_fraction').reset_index()
        delta = delta.merge(quality, on=['concept', 'budget'], validate='many_to_one')
        summaries.append(summary)
        contrasts.append(delta)
        if split == 'test':
            nodes, edges = graph_trace(selections, archive['S'], archive['R_train'], names, pd.read_csv(directory / 'communities_train.csv'), protocol['tau'])
            for c, name in enumerate(names):
                nodes[f'test_{name}'] = archive['S_test'][nodes.neuron.to_numpy(), c]
            nodes.to_csv(output / 'selected_neurons.csv', index=False)
            edges.to_csv(output / 'ranking_edge_evidence.csv.gz', index=False)
            nodes.groupby(['concept', 'budget', 'community']).agg(selected_count=('neuron', 'size'), total_joint_score=('joint_score', 'sum')).to_csv(output / 'selected_communities.csv')
            plot_tradeoff(summary, output)
    pd.concat(summaries).to_csv(output / 'paired_summary.csv', index=False)
    pd.concat(contrasts).to_csv(output / 'paired_contrasts.csv', index=False)
    pd.concat(diagnostics).to_csv(output / 'patient_diagnostics.csv', index=False)
    pd.concat(coverage).to_csv(output / 'coverage_intervals.csv', index=False)
    pd.DataFrame({'neuron': range(len(archive['thresholds'])), 'training_threshold': archive['thresholds']}).to_csv(output / 'frozen_thresholds.csv', index=False)
    notes = f'''# Semantic reliability audit

Bootstrap repetitions: {repeats}; seed: {seed}. Selections, calibration and model are fixed.
IoU is recomputed as summed intersection / summed union in each patient bootstrap. Selectivity is query IoU minus the largest other-concept IoU, computed per neuron before averaging. KL and semantic metrics use identical resampled patients. Random/matched draws are averaged within patient/selection, never counted as extra patients. Intervals are pointwise, not multiplicity-adjusted; no new significance claim is generated. Matching failures remain in paired_contrasts.csv.

Patient mean maximum IoU is distinct from aggregate neuron IoU. Coverage intervals recompute the aggregate statistic after patient resampling. Train/val/test sample sizes remain different. Thresholds are frozen per neuron, not fitted per patient.

Graph evidence uses training communities and training edges. For each selected source neuron, the contributions S_ic A_ij S_jc sum to its joint score. Edge rows are directed score contributions even though the coactivation graph is undirected. Community -1 means an isolated neuron. No causal/anatomical relation is implied.

Missing area diagnostics for splits: {missing}. Use --refresh-evidence to recover them and seeded overlays; never infer those areas from existing IoU alone. The refresh verifies checkpoint hash, loaded patient order and agreement with saved semantic scores. Seeded overlays choose one patient per pathology independently of overlap; neurons are selected by training association. These are illustrative stratified examples, not a population-weighted image sample.

Original run: {root}
'''
    (output / 'README.md').write_text(notes)
    print(f'Saved audit: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layer_directory', type=Path)
    parser.add_argument('--bootstraps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--refresh-evidence', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.bootstraps < 1:
        parser.error('--bootstraps must be positive')
    with threadpool_limits(limits=4):
        analyze(args.layer_directory, args.bootstraps, args.seed, args.refresh_evidence, args.device)
