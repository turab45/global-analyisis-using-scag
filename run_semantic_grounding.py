"""Compare frozen IoU labels with controlled continuous concept–neuron evidence.

Does not alter the original graph or label files. Pixel AP is evaluated on a fixed
uniform sample per patient; uncertainty resamples patients, never pixels.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scag.research import cache_patients, file_sha256, validate_manifests

DEFAULT_RUN = Path('scag_outputs/research/acdc/anatomy/20261109T154512_822872Z/resnet18/encoder.layer4')


def spatial_ap(scores, targets):
    """Tie-correct non-interpolated AP: K x P scores and C x P binary labels."""
    if scores.ndim != 2 or targets.ndim != 2 or scores.shape[1] != targets.shape[1] or not torch.isfinite(scores).all():
        raise ValueError('Require finite K x P scores and C x P targets.')
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError('AP targets must be binary.')
    ordered, indices = scores.sort(dim=1, descending=True, stable=True)
    labels = targets[:, indices].permute(1, 0, 2).float()
    tp = labels.cumsum(dim=-1)
    ends = torch.cat([ordered[:, :-1] != ordered[:, 1:], torch.ones_like(ordered[:, :1], dtype=torch.bool)], dim=1)[:, None, :]
    previous = torch.cummax(torch.where(ends, tp, 0), dim=-1).values
    previous = torch.cat([torch.zeros_like(previous[..., :1]), previous[..., :-1]], dim=-1)
    rank = torch.arange(1, scores.shape[1] + 1, device=scores.device)
    positives = targets.sum(dim=1)[None, :]
    area = ((tp - previous) * (tp / rank) * ends).sum(dim=-1)
    return torch.where(positives > 0, area / positives.clamp_min(1), torch.nan)


def measure(acts, masks, train_mask_sum, train_count, split, args):
    n, channels = acts.shape[:2]
    concepts, h, w = masks.shape[1:]
    rng = np.random.default_rng(args.seed + ['train', 'val', 'test'].index(split))
    ap, shifted = np.empty((n, channels, concepts)), np.empty((n, channels, concepts))
    position, prevalence = np.empty((n, 1, concepts)), np.empty((n, 1, concepts))
    shifts, pixels = [], []
    for i in range(n):
        coords = rng.choice(h * w, min(args.pixels, h * w), replace=False)
        offsets = [(int(rng.integers(max(1, h // 4), max(2, 3 * h // 4))), int(rng.integers(max(1, w // 4), max(2, 3 * w // 4)))) for _ in range(args.shifts)]
        pixels.append(coords)
        shifts.append(offsets)
        target = masks[i].to(args.device).float()
        labels = [target.flatten(1)[:, coords]]
        labels.extend(torch.roll(target, (dy, dx), (-2, -1)).flatten(1)[:, coords] for dy, dx in offsets)
        labels = torch.cat(labels)
        prior = (train_mask_sum - masks[i].float()) / (train_count - 1) if split == 'train' else train_mask_sum / train_count
        position[i, 0] = spatial_ap(prior.to(args.device).flatten(1)[:, coords], labels[:concepts]).diag().cpu().numpy()
        prevalence[i, 0] = labels[:concepts].mean(dim=1).cpu().numpy()
        for start in range(0, channels, 32):
            score = torch.nn.functional.interpolate(acts[i:i + 1, start:start + 32].to(args.device), size=(h, w), mode='bilinear', align_corners=False)[0].flatten(1)[:, coords]
            values = spatial_ap(score, labels).cpu().numpy()
            ap[i, start:start + 32] = values[:, :concepts]
            shifted[i, start:start + 32] = np.nanmean(values[:, concepts:].reshape(len(score), args.shifts, concepts), axis=1)
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f'{split}: continuous evidence {i + 1}/{n}', flush=True)
    return dict(ap=ap, shifted=shifted, position=position, prevalence=prevalence, sampled_pixels=np.asarray(pixels), mask_shifts=np.asarray(shifts))


def bootstrap_evidence(result, repeats, seed):
    # Simultaneous lower bands across every channel, concept and all three controls.
    differences = np.stack([result['ap'] - result[key] for key in ['prevalence', 'position', 'shifted']], axis=-1)
    shape = differences.shape[1:]
    flat = differences.reshape(len(differences), -1)
    valid = np.isfinite(flat)
    if not valid.any(axis=0).all():
        raise ValueError('Some concepts have no evaluable sampled-positive patients; increase --pixels or audit masks.')
    mean = np.nansum(flat, axis=0) / valid.sum(axis=0)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(flat), np.full(len(flat), 1 / len(flat)), size=repeats)
    denominator = counts @ valid.astype(float)
    sampled = np.divide(counts @ np.nan_to_num(flat), denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
    radius = max(0., np.nanquantile(np.nanmax(sampled - mean, axis=1), .95))
    return mean.reshape(shape), (mean - radius).reshape(shape), np.nanmean(np.where(valid, flat > 0, np.nan), axis=0).reshape(shape)


def compare(directory, output, archive, evidence, manifests, args):
    names = archive['concepts'].tolist()
    rows, coverage, quality = [], [], []
    frozen = None
    baseline = archive['S_train'] >= args.gamma
    for split, result in evidence.items():
        mean, lower, consistency = bootstrap_evidence(result, args.bootstraps, args.seed)
        supported = (lower > 0).all(axis=-1)
        if split == 'train':
            frozen = supported.copy()
        S = archive[f'S_{split}']
        for i in range(len(S)):
            for c, concept in enumerate(names):
                row = dict(split=split, neuron=i, concept=concept, iou=S[i, c], mean_ap=np.nanmean(result['ap'][:, i, c]), mean_prevalence=np.nanmean(result['prevalence'][:, 0, c]), mean_position_ap=np.nanmean(result['position'][:, 0, c]), mean_shifted_ap=np.nanmean(result['shifted'][:, i, c]), weight=max(0., mean[i, c].min()), supported_here=bool(supported[i, c]), frozen_new_edge=bool(frozen[i, c]), frozen_iou_edge=bool(baseline[i, c]))
                for j, key in enumerate(['prevalence', 'position', 'shifted']):
                    row.update({f'gain_{key}': mean[i, c, j], f'lower_{key}': lower[i, c, j], f'patient_fraction_above_{key}': consistency[i, c, j]})
                rows.append(row)
        for method, selected, passes in [('iou', baseline, S >= args.gamma), ('controlled_ap', frozen, supported)]:
            coverage.append(dict(split=split, method=method, training_edges=int(selected.sum()), training_labelled_neurons=int(selected.any(axis=1).sum()), retained_edges=int((selected & passes).sum()), supported_neurons_here=int(passes.any(axis=1).sum()), supported_edges_here=int(passes.sum())))
        with np.load(directory / f'patient_concepts_{split}.npz') as data:
            patient_iou = data['per_patient_iou']
        assert patient_iou.shape == result['ap'].shape and len(patient_iou) == len(manifests[split])
        if split == 'train':
            train_weight = mean.min(axis=-1)
            continue
        draws = np.random.default_rng(args.seed).integers(len(patient_iou), size=(args.bootstraps, len(patient_iou)))
        for c, concept in enumerate(names):
            k = min(args.top_k, len(S))
            new = np.argsort(-train_weight[:, c], kind='stable')[:k]
            old = np.argsort(-archive['S_train'][:, c], kind='stable')[:k]
            for metric, values in [('ap', result['ap']), ('position_gain', result['ap'] - result['position']), ('iou', patient_iou)]:
                left, right = np.nanmean(values[:, new, c], axis=1), np.nanmean(values[:, old, c], axis=1)
                difference = left - right
                lo, hi = np.nanquantile(np.nanmean(difference[draws], axis=1), [.025, .975])
                quality.append(dict(split=split, concept=concept, k=k, metric=metric, controlled_ap_mean=np.nanmean(left), iou_ranking_mean=np.nanmean(right), paired_difference=np.nanmean(difference), ci_low=lo, ci_high=hi, new_neurons=json.dumps(new.tolist()), iou_neurons=json.dumps(old.tolist())))
    edges = pd.DataFrame(rows)
    coverage, quality = pd.DataFrame(coverage), pd.DataFrame(quality)
    diagnostics = edges.groupby(['split', 'concept'], sort=False).agg(position_ap=('mean_position_ap', 'first'), best_channel_position_gain=('gain_position', 'max'), channels_beating_position_mean=('gain_position', lambda x: int((x > 0).sum())), prevalence_supported=('lower_prevalence', lambda x: int((x > 0).sum())), shift_supported=('lower_shifted', lambda x: int((x > 0).sum()))).reset_index()
    diagnostics.to_csv(output / 'control_diagnostics.csv', index=False)
    edges.to_csv(output / 'semantic_edges.csv', index=False)
    coverage.to_csv(output / 'coverage.csv', index=False)
    quality.to_csv(output / 'ranking_comparison.csv', index=False)
    # Functional graph is copied from the frozen run, never used to manufacture semantic labels.
    R = archive['R_train']
    source, target = np.where(np.triu((R >= args.tau) & (R > 0), 1))
    pd.DataFrame(dict(source=source, target=target, correlation=R[source, target])).to_csv(output / 'functional_edges.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    for ax, concept in zip(axes[0], names):
        train = edges.query('split == "train" and concept == @concept').sort_values('neuron')
        test = edges.query('split == "test" and concept == @concept').sort_values('neuron')
        ax.scatter(test.iou, test.gain_position, s=12, alpha=.4, color='gray', label='All channels')
        chosen = train.frozen_new_edge.to_numpy()
        ax.scatter(test.iou.to_numpy()[chosen], test.gain_position.to_numpy()[chosen], s=22, label='Training-supported AP edge')
        ax.axhline(0, color='black', linewidth=.7)
        ax.axvline(args.gamma, color='black', linestyle='--', linewidth=.7)
        ax.set(title=concept, xlabel='Test IoU (original protocol)', ylabel='Test AP minus position-only AP')
        ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'comparison.{suffix}', dpi=180)
    plt.close(fig)
    report = '''# Controlled semantic grounding comparison

This experiment changes concept–neuron evidence only. Functional edges are unchanged. No community analysis, label propagation, CNN training or prediction intervention is performed.

## Method

Continuous spatial average precision (AP), with correct tied-score handling, is measured on uniform sampled pixels per patient. Pixels are sampled without looking at masks or responses. AP is undefined when a concept has no positive sampled pixels; those patient–concept observations are excluded, not assigned perfect or zero AP. All channels share sampled pixels and spatial controls within a patient.

Controls: concept prevalence; a training-derived pixel-location concept probability map (leave-one-patient-out for training); and mean AP under large circular translations of the mask. Circular shifts preserve mask area/shape on the torus but may create unrealistic positions, so they are a sensitivity control, not an anatomical null or a permutation p-value. Position maps do not remove every anatomical confound.

Edge weight = positive part of the smallest mean AP gain over the three controls. A supported edge requires positive approximate simultaneous 95% patient-bootstrap lower bounds for all three gains, across all channel–concept–control comparisons in that split. Training selects edges; evaluation never reselects the frozen graph. This is an exploratory evidence rule, not proof of a detector. Bootstrap inference conditions on the fitted position map and pixel/shift samples. Test retention is deliberately stricter than merely positive mean gain.

Coverage thresholds for AP and IoU differ in meaning: increased coverage is not evidence of better labels. The fixed-budget ranking comparison uses top-k training rankings (including unsupported edges if fewer than k pass), identical patients, AP and original per-patient IoU. Its paired intervals are pointwise and unadjusted. Do not compare raw AP numerically with IoU as if they were the same metric. This test cohort was previously inspected.

## Coverage

```
''' + coverage.to_string(index=False) + '\n```\n\n## Fixed-budget test comparison\n\n```\n' + quality.query('split == "test"').drop(columns=['new_neurons', 'iou_neurons']).to_string(index=False) + '\n```\n\nSee semantic_edges.csv for every neuron–concept profile, uncertainty and patient consistency; patient_evidence_*.npz and *_samples.csv for traceable patient evidence; functional_edges.csv for the unchanged coactivation graph; comparison.png/pdf for the comparison figure.\n'
    report += '\n## Which control limits support?\n\n```\n' + diagnostics.to_string(index=False) + '\n```\n'
    report += '\nA zero supported-edge count means no pair passed this particular conjunction of controls; it does not establish absent semantic information. Beating a strong location predictor is a conservative superiority criterion, not a necessary condition for anatomical association. Circular-shift and prevalence gains alone do not exclude positional confounding. Higher AP under the new ranking is evidence about spatial ranking, not proof of selectivity, prediction relevance, or stronger joint-graph explanations.\n'
    (output / 'RESULTS.md').write_text(report)
    print(coverage.to_string(index=False), flush=True)
    print(quality.query('split == "test"').drop(columns=['new_neurons', 'iou_neurons']).to_string(index=False), flush=True)


def run(args):
    from scag.config import SCAGConfig
    from scag.data import load_datasets
    from scag.models import build_model
    from scag.activations import extract_activations
    from scag.semantics import concept_masks
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; run cinema on the GPU host.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = args.layer_dir.resolve()
    root = directory.parent.parent
    protocol = json.loads((root / 'protocol.json').read_text())
    args.gamma, args.tau = protocol['gamma'], protocol['tau']
    manifests = {s: pd.read_csv(root / f'{s}_samples.csv', dtype={'pid': str}) for s in ['train', 'val', 'test']}
    validate_manifests(manifests)
    archive = dict(np.load(directory / 'representations.npz'))
    output = directory / 'semantic_grounding' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True)
    cfg = SCAGConfig(dataset=protocol['dataset'], model_seed=protocol.get('model_seed', 0))
    checkpoint = json.loads((directory.parent / 'checkpoint.json').read_text())
    actual = cfg.model_root / directory.parent.name / cfg.formulation / cfg.initialization / f'seed_{cfg.model_seed}' / 'best_val_mcc.pt'
    if actual.resolve() != Path(checkpoint['path']).resolve() or file_sha256(actual) != checkpoint['sha256']:
        raise ValueError('Checkpoint differs from original run.')
    model, _, _ = build_model(directory.parent.name, cfg)
    model.to(args.device).eval()
    datasets = dict(zip(['train', 'val', 'test'], load_datasets(cfg)[:3]))
    evidence = {}
    for split, dataset in datasets.items():
        manifest = manifests[split]
        mapping = {str(pid): i for i, pid in enumerate(dataset.metadata.pid)}
        if len(mapping) != len(dataset):
            raise ValueError('Duplicate patient IDs.')
        loader = cache_patients(torch.utils.data.Subset(dataset, [mapping[pid] for pid in manifest.pid]), manifest, protocol['batch_size'])
        acts, segments = extract_activations(model, loader, model.get_submodule(directory.name), args.device)
        if not np.allclose(acts.mean((2, 3)).numpy(), archive[f'pooled_{split}'], rtol=1e-4, atol=1e-5):
            raise ValueError(f'{split}: refreshed activations differ from saved run.')
        masks, names = concept_masks(segments, protocol['semantic_level'])
        if names != archive['concepts'].tolist():
            raise ValueError('Concept ordering differs.')
        if split == 'train':
            train_sum, train_count = masks.float().sum(dim=0), len(masks)
        with torch.inference_mode():
            evidence[split] = measure(acts, masks, train_sum, train_count, split, args)
        np.savez_compressed(output / f'patient_evidence_{split}.npz', **evidence[split], pid=manifest.pid.to_numpy(dtype=str), concepts=names)
        manifest.to_csv(output / f'{split}_samples.csv', index=False)
        del acts, segments, masks, loader
    with threadpool_limits(limits=4):
        compare(directory, output, archive, evidence, manifests, args)
    provenance = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, checkpoint=checkpoint, representation_sha256=file_sha256(directory / 'representations.npz'), script_sha256=file_sha256(__file__), protocol=protocol)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'Saved: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--pixels', type=int, default=4096)
    parser.add_argument('--shifts', type=int, default=8)
    parser.add_argument('--bootstraps', type=int, default=2000)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.pixels, args.shifts, args.top_k) < 1 or args.bootstraps < 100:
        parser.error('Require positive pixels/shifts/top-k and at least 100 bootstraps.')
    run(args)
