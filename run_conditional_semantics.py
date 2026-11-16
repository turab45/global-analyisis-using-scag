"""Test whether one neuron's activation adds predictive anatomy evidence to location.

Uses verified cached CNN maps, never retrains the CNN or changes coactivation.
60/30 development split for the default 90-patient cohort: fit on 60, freeze edge
selection using 30, evaluate unchanged probes on existing validation/test data.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from run_semantic_grounding import DEFAULT_RUN, spatial_ap
from scag.research import file_sha256, validate_manifests


def fit_probe(x, y, offset, ridge, steps=40):
    """Batched regularized offset-logistic probes; one slope and intercept per pair."""
    n, channels, pixels = x.shape
    concepts = y.shape[1]
    if y.shape != (n, concepts, pixels) or offset.shape != (concepts, pixels) or ridge <= 0:
        raise ValueError('Invalid probe shapes or regularization.')
    if not torch.isfinite(x).all() or not torch.isfinite(offset).all() or not torch.all((y == 0) | (y == 1)):
        raise ValueError('Require finite inputs and binary masks.')
    parameters = []
    max_gradient = 0.
    for start in range(0, channels, 16):
        features = x[:, start:start + 16, None, :]
        count = features.shape[1]
        b, w = torch.zeros((count, concepts), device=x.device), torch.zeros((count, concepts), device=x.device)
        for iteration in range(steps):
            logits = offset[None, None] + b[None, :, :, None] + w[None, :, :, None] * features
            probability = logits.sigmoid()
            error = probability - y[:, None]
            g0 = error.mean((0, 3)) + ridge * b
            g1 = (error * features).mean((0, 3)) + ridge * w
            gradient = max(g0.abs().max().item(), g1.abs().max().item())
            if gradient < 1e-6:
                break
            variance = probability * (1 - probability)
            h00 = variance.mean((0, 3)) + ridge
            h01 = (variance * features).mean((0, 3))
            h11 = (variance * features.square()).mean((0, 3)) + ridge
            determinant = (h00 * h11 - h01.square()).clamp_min(1e-12)
            b -= ((h11 * g0 - h01 * g1) / determinant).clamp(-2, 2)
            w -= ((h00 * g1 - h01 * g0) / determinant).clamp(-2, 2)
        max_gradient = max(max_gradient, gradient)
        parameters.append(torch.stack([b, w], dim=-1))
    if max_gradient > 1e-4:
        raise RuntimeError(f'Probe optimizer has not converged: maximum gradient {max_gradient:.3g}.')
    return torch.cat(parameters), max_gradient


def evaluate(x, y, offset, parameters, with_ap=True):
    n, channels, _ = x.shape
    losses = np.empty((n, channels, y.shape[1]))
    ap = np.full_like(losses, np.nan)
    for i in range(n):
        logits = offset[None] + parameters[:, :, 0, None] + parameters[:, :, 1, None] * x[i, :, None, :]
        loss = torch.nn.functional.softplus(logits) - y[i, None] * logits
        losses[i] = loss.mean(dim=-1).cpu().numpy()
        if with_ap:
            for c in range(y.shape[1]):
                ap[i, :, c] = spatial_ap(logits[:, c], y[i, c:c + 1])[:, 0].cpu().numpy()
    return losses, ap


def simultaneous_lower(differences, repeats, seed, family=None):
    """Approximate one-sided simultaneous patient-bootstrap bands, probes fixed."""
    if differences.ndim != 4 or not np.isfinite(differences).all() or len(differences) < 2:
        raise ValueError('Need finite patient x neuron x concept x control differences.')
    mean = differences.mean(axis=0)
    flat = differences.reshape(len(differences), -1)
    counts = np.random.default_rng(seed).multinomial(len(flat), np.full(len(flat), 1 / len(flat)), size=repeats)
    deviations = counts @ flat / len(flat) - flat.mean(axis=0)
    if family is not None:
        keep = np.repeat(family[:, :, None], differences.shape[-1], axis=-1).ravel()
        if not keep.any():
            return mean, np.full_like(mean, np.nan)
        deviations = deviations[:, keep]
    radius = max(0., np.quantile(deviations.max(axis=1), .95))
    return mean, mean - radius


def within_pathology_order(frame, seed):
    order = np.arange(len(frame))
    rng = np.random.default_rng(seed)
    for _, group in frame.reset_index(drop=True).groupby('pathology'):
        if len(group) < 2:
            raise ValueError('Shuffled control requires >=2 patients per pathology.')
        order[group.index] = rng.permutation(group.index.to_numpy())
    return order


def load_cache(args, output):
    source = args.cache_dir.resolve()
    directory = args.layer_dir.resolve()
    root = directory.parent.parent
    protocol = json.loads((root / 'protocol.json').read_text())
    old_provenance = json.loads((source / 'provenance.json').read_text())
    if old_provenance['source_representation_sha256'] != file_sha256(directory / 'representations.npz'):
        raise ValueError('Cached evidence is not from this representation archive.')
    archive = dict(np.load(directory / 'representations.npz'))
    manifests, maps, hashes = {}, {}, {}
    for split in ['train', 'val', 'test']:
        manifests[split] = pd.read_csv(source / f'{split}_samples.csv', dtype={'pid': str})
        original = pd.read_csv(root / f'{split}_samples.csv', dtype={'pid': str})
        if manifests[split].pid.tolist() != original.pid.tolist():
            raise ValueError('Cached manifest does not match original patient ordering.')
        path = source / f'spatial_maps_{split}.npz'
        hashes[split] = file_sha256(path)
        with np.load(path) as data:
            acts, masks = data['activations'], data['masks']
            if data['pid'].tolist() != original.pid.tolist() or len(acts) != len(masks):
                raise ValueError('Cached map patient IDs are inconsistent.')
        if not np.isfinite(acts).all() or not np.allclose(acts.mean((2, 3)), archive[f'pooled_{split}'], rtol=1e-4, atol=1e-5):
            raise ValueError('Cached activations disagree with saved representations.')
        if masks.shape[1] != len(archive['concepts']) or not np.isin(masks, [0, 1]).all():
            raise ValueError('Invalid concept masks.')
        maps[split] = (torch.from_numpy(acts), torch.from_numpy(masks))
    validate_manifests(manifests)
    return protocol, archive, manifests, maps, hashes


def run(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.layer_dir.resolve() / 'conditional_semantics' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True)
    protocol, archive, manifests, maps, hashes = load_cache(args, output)
    names = archive['concepts'].tolist()
    train_manifest = manifests['train']
    fit_ids, selection_ids = train_test_split(np.arange(len(train_manifest)), test_size=1 / 3, random_state=args.seed, stratify=train_manifest.pathology)
    fit_ids, selection_ids = np.sort(fit_ids), np.sort(selection_ids)
    assert not set(fit_ids) & set(selection_ids)
    split_manifest = train_manifest.copy()
    split_manifest['role'] = 'selection'
    split_manifest.loc[fit_ids, 'role'] = 'fit'
    split_manifest.to_csv(output / 'training_roles.csv', index=False)
    h, w = maps['train'][1].shape[-2:]
    coords = np.random.default_rng(args.seed).choice(h * w, min(args.pixels, h * w), replace=False)
    X, Y = {}, {}
    for split, (acts, masks) in maps.items():
        X[split] = torch.stack([torch.nn.functional.interpolate(a[None].to(args.device), size=(h, w), mode='bilinear', align_corners=False)[0].flatten(1)[:, coords] for a in acts])
        Y[split] = masks.to(args.device).float().flatten(2)[:, :, coords]
    template = X['train'][fit_ids].mean(dim=0)
    scale = (X['train'][fit_ids] - template).square().mean((0, 2)).sqrt().clamp_min(1e-6)
    X = {s: (x - template) / scale[None, :, None] for s, x in X.items()}
    probability = (Y['train'][fit_ids].sum(dim=0) + 1) / (len(fit_ids) + 2)
    offset = torch.logit(probability)
    print(f'Fit={len(fit_ids)}, selection={len(selection_ids)}; regularization fixed at {args.ridge}; device={args.device}', flush=True)
    base_parameters, base_gradient = fit_probe(torch.zeros_like(X['train'][fit_ids, :1]), Y['train'][fit_ids], offset, args.ridge)
    parameters, gradient = fit_probe(X['train'][fit_ids], Y['train'][fit_ids], offset, args.ridge)
    fit_frame = train_manifest.iloc[fit_ids].reset_index(drop=True)
    shuffle_fit = within_pathology_order(fit_frame, args.seed + 101)
    control_parameters, control_gradient = fit_probe(X['train'][fit_ids][shuffle_fit], Y['train'][fit_ids], offset, args.ridge)
    np.savez_compressed(output / 'frozen_probes.npz', parameters=parameters.cpu().numpy(), base_parameters=base_parameters.cpu().numpy(), shuffled_parameters=control_parameters.cpu().numpy(), offset=offset.cpu().numpy(), activation_template=template.cpu().numpy(), scale=scale.cpu().numpy(), sampled_pixels=coords)
    print('All probes fitted and converged. Evaluating unchanged models.', flush=True)
    rows, cover, comparisons, results = [], [], [], {}
    selected = None
    # Raw AP must use raw responses, not position-residualized scores used by the conditional model.
    raw_ap_train = np.stack([spatial_ap(x * scale[:, None] + template, y).cpu().numpy() for x, y in zip(X['train'], Y['train'])])
    for split in ['selection', 'val', 'test']:
        ids = selection_ids if split == 'selection' else np.arange(len(manifests[split]))
        original_split = 'train' if split == 'selection' else split
        frame = manifests[original_split].iloc[ids].reset_index(drop=True)
        x, y = X[original_split][ids], Y[original_split][ids]
        baseline_loss, baseline_ap = evaluate(torch.zeros_like(x[:, :1]), y, offset, base_parameters)
        loss, ap = evaluate(x, y, offset, parameters)
        permutation = within_pathology_order(frame, args.seed + 202 + ['selection', 'val', 'test'].index(split))
        control_loss, _ = evaluate(x[permutation], y, offset, control_parameters, with_ap=False)
        differences = np.stack([baseline_loss - loss, control_loss - loss], axis=-1)
        with threadpool_limits(limits=4):
            means, lower = simultaneous_lower(differences, args.bootstraps, args.seed, selected)
        if selected is None:
            selected = (lower > 0).all(axis=-1)
        corroborated = selected & (lower > 0).all(axis=-1)
        results[split] = dict(loss=loss, baseline_loss=baseline_loss, shuffled_loss=control_loss, gain=differences[..., 0], ap=ap, baseline_ap=baseline_ap, means=means, lower=lower)
        np.savez_compressed(output / f'patient_scores_{split}.npz', **results[split], pid=frame.pid.to_numpy(dtype=str), shuffled_patient=frame.pid.iloc[permutation].to_numpy(dtype=str))
        for i in range(len(parameters)):
            for c, name in enumerate(names):
                rows.append(dict(split=split, neuron=i, concept=name, gain=means[i, c, 0], lower_gain=lower[i, c, 0], gain_over_shuffled=means[i, c, 1], lower_gain_over_shuffled=lower[i, c, 1], training_selected=bool(selected[i, c]), corroborated=bool(corroborated[i, c]), patient_fraction_improved=float((differences[:, i, c, 0] > 0).mean()), mean_ap=np.nanmean(ap[:, i, c]), position_ap=np.nanmean(baseline_ap[:, 0, c]), slope=float(parameters[i, c, 1])))
        cover.append(dict(split=split, patients=len(frame), selected_channels=int(selected.any(axis=1).sum()), selected_edges=int(selected.sum()), positive_gain_edges=int((selected & (means[..., 0] > 0)).sum()), corroborated_edges=int(corroborated.sum()), corroborated_channels=int(corroborated.any(axis=1).sum())))
        print(f'{split}: selected={selected.sum()}, corroborated={corroborated.sum()}', flush=True)
        if split == 'selection':
            ranking = means[..., 0].copy()
            continue
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(len(frame), size=(args.bootstraps, len(frame)))
        with np.load(args.layer_dir / f'patient_concepts_{split}.npz') as data:
            original_iou = data['per_patient_iou']
        for c, name in enumerate(names):
            k = min(args.top_k, len(parameters))
            choices = {'conditional': np.argsort(-ranking[:, c], kind='stable')[:k], 'iou': np.argsort(-archive['S_train'][:, c], kind='stable')[:k], 'raw_ap': np.argsort(-np.nanmean(raw_ap_train[:, :, c], axis=0), kind='stable')[:k]}
            for method, channels in choices.items():
                for metric, values in [('log_loss_gain', differences[..., 0]), ('predictor_ap', ap), ('original_iou', original_iou)]:
                    patient = np.nanmean(values[:, channels, c], axis=1)
                    reference = np.nanmean(values[:, choices['conditional'], c], axis=1)
                    delta = reference - patient
                    lo, hi = np.nanquantile(np.nanmean(delta[draws], axis=1), [.025, .975])
                    comparisons.append(dict(split=split, concept=name, ranking=method, metric=metric, k=k, mean=np.nanmean(patient), conditional_minus_ranking=np.nanmean(delta), difference_ci_low=lo, difference_ci_high=hi, channels=json.dumps(channels.tolist())))
    edges, coverage, comparison = pd.DataFrame(rows), pd.DataFrame(cover), pd.DataFrame(comparisons)
    edges.to_csv(output / 'semantic_edges.csv', index=False)
    coverage.to_csv(output / 'coverage.csv', index=False)
    comparison.to_csv(output / 'ranking_comparison.csv', index=False)
    make_report(output, edges, coverage, comparison, args, len(fit_ids), len(selection_ids))
    provenance = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, cached_maps_sha256=hashes, script_sha256=file_sha256(__file__), representation_sha256=file_sha256(args.layer_dir / 'representations.npz'), optimizer_max_gradients=dict(position=base_gradient, augmented=gradient, shuffled=control_gradient), protocol=protocol)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(coverage.to_string(index=False), flush=True)
    print(comparison.query('split == "test" and metric == "log_loss_gain"').drop(columns='channels').to_string(index=False), flush=True)
    print(f'Saved: {output}', flush=True)


def make_report(output, edges, coverage, comparison, args, nfit, nselect):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, edges.concept.unique()):
        fit = edges[(edges.split == 'selection') & (edges.concept == name)]
        test = edges[(edges.split == 'test') & (edges.concept == name)]
        ax.scatter(fit.gain, test.gain, c=np.where(fit.training_selected, 'tab:blue', 'lightgray'), s=12)
        ax.axhline(0, color='black', linewidth=.7)
        ax.axvline(0, color='black', linewidth=.7)
        ax.set(title=name, xlabel='Selection log-loss improvement (nats/pixel)', ylabel='Test log-loss improvement (nats/pixel)')
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'comparison.{suffix}', dpi=180)
    plt.close(fig)
    text = f'''# Location-conditioned semantic association

## Design

Frozen CNN activations. {nfit} training patients fit all models/templates; {nselect} disjoint training patients select edges. No refitting or tuning on selection, validation or test patients. Existing validation/test cohorts have already been inspected: this experiment is exploratory.

Position baseline: per-pixel concept frequency with Beta(1,1) smoothing from fitting patients, converted to log odds, plus a regularized intercept. Augmented model: identical fixed location offset plus intercept and one standardized neuron-activation coefficient. The neuron's training spatial mean is subtracted, and its residual is scaled by its fitting-cohort RMS. This is a deliberately simple additive conditional logistic test, not a universally expressive estimate of conditional information. Negative slopes are permitted and do not mean an excitatory detector.

Both models use the same intercept penalty; augmentation adds one penalized coefficient. Ridge strength is fixed at {args.ridge}, not tuned using any evaluation cohort. A same-complexity control fits a model using activation maps randomly reassigned within pathology; a separate reassignment is used on each evaluation split. This single shuffle is a sanity comparator, not a full permutation null.

Primary score is position-only minus augmented mean binary log-loss in nats per sampled pixel, averaged equally over patients. Positive means incremental predictive performance under these models. Class weighting is not used. Secondary AP is AP of model predictions; it is not raw activation AP. Original IoU is kept separately. AP and log-loss have different numerical scales.

Edge selection requires positive approximate simultaneous 95% patient-bootstrap lower bounds versus both baseline and shuffled control across all neuron–concept–control comparisons on the training selection cohort. Validation/test corroboration uses the frozen selected family only. Bootstrap intervals condition on fitted probes and pixel samples, and do not include fitting uncertainty. Null gains and unresolved channels are retained in the tables. No coactivation-based label propagation, community analysis, graph modification, CNN training or prediction intervention occurs.

Uniform spatial coordinates are sampled once without consulting masks/activations and reused across patients/models. The baseline has the same spatial resolution as the augmented model. Constant/dead channels cannot add a slope contribution. Results depend on regularization, smoothing and sampling; this is one prespecified configuration, not a sensitivity sweep.

## Coverage

```
''' + coverage.to_string(index=False) + '\n```\n\n## Test ranking comparison\n\n```\n' + comparison.query('split == "test"').drop(columns='channels').to_string(index=False) + '\n```\n\nRankings are frozen from development data: IoU/raw-AP use the original training cohort, conditional uses the training selection cohort. All methods select the same number of channels, even if fewer pass the edge support rule. The table compares their conditional predictive gains, predictor AP and original IoU. Paired differences are conditional ranking minus comparator, with pointwise unadjusted patient-bootstrap intervals. Superior top-k performance alone does not imply statistically supported graph edges.\n\nFiles: frozen_probes.npz, training_roles.csv, patient_scores_*.npz, semantic_edges.csv, coverage.csv, ranking_comparison.csv, comparison.png/pdf and provenance.json. No existing labels have been replaced.\n'
    control_rows = []
    with np.load(output / 'patient_scores_test.npz') as saved:
        scores = dict(saved)
    development = edges[edges.split == 'selection']
    for c, name in enumerate(edges.concept.unique()):
        group = development[development.concept == name]
        chosen = group.sort_values(['gain', 'neuron'], ascending=[False, True]).head(args.top_k).neuron.to_numpy()
        augmented = scores['loss'][:, chosen, c].mean(axis=1)
        for control, baseline in [('position', scores['baseline_loss'][:, 0, c]), ('shuffled', scores['shuffled_loss'][:, chosen, c].mean(axis=1))]:
            difference = baseline - augmented
            draws = np.random.default_rng(args.seed).integers(len(difference), size=(args.bootstraps, len(difference)))
            low, high = np.quantile(difference[draws].mean(axis=1), [.025, .975])
            control_rows.append(dict(concept=name, control=control, k=len(chosen), mean_gain=difference.mean(), ci_low=low, ci_high=high, control_loss=baseline.mean(), augmented_loss=augmented.mean()))
    controls = pd.DataFrame(control_rows)
    controls.to_csv(output / 'topk_control_comparison.csv', index=False)
    text += '\n## Training-ranked top-k probes versus controls on test\n\n```\n' + controls.to_string(index=False) + '\n```\nThese are averages of individual-probe losses, not a multi-neuron decoder. Intervals are pointwise and unadjusted; these aggregate comparisons do not license individual neuron labels or change the frozen selection rule.\n'
    text += '\n## Interpretation\n\nThe conditional question is different from requiring a neuron alone to outperform a spatial prior. Positive held-out loss gains support complementary predictive information under this restricted probe. Zero selected edges means the prespecified individual-link support rule was not met; it does not imply zero information in the representation. Dependence on probe capacity, regularization and the chosen controls must be checked before using this as the main SCAG labelling method. Do not relax the threshold after seeing these test results.\n'
    (output / 'RESULTS.md').write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_RUN / 'patient_semantic_labels/20261111T141750_612642Z')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--pixels', type=int, default=2048)
    parser.add_argument('--ridge', type=float, default=.01)
    parser.add_argument('--bootstraps', type=int, default=2000)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.ridge <= 0 or min(args.pixels, args.top_k) < 1 or args.bootstraps < 100:
        parser.error('Require positive ridge/pixels/top-k and at least 100 bootstrap repetitions.')
    with torch.inference_mode():
        run(args)
