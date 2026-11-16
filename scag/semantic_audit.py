"""Frozen-selection semantic reliability, paired uncertainty and graph traceability."""
import json
import numpy as np
import pandas as pd
from .relations import adjacency


def aggregate_iou(intersection, union):
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)


def bootstrap_profiles(intersection, union, repeats, seed):
    if intersection.shape != union.shape or intersection.ndim != 3 or len(intersection) < 2 or repeats < 1:
        raise ValueError('Require matching patient/channel/concept arrays, >=2 patients and positive repeats.')
    if not np.isfinite(intersection).all() or not np.isfinite(union).all() or (intersection < 0).any() or (union < intersection).any():
        raise ValueError('Invalid intersection/union counts.')
    n = len(intersection)
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=repeats)
    shape = (repeats, *intersection.shape[1:])
    boot = aggregate_iou((counts @ intersection.reshape(n, -1)).reshape(shape), (counts @ union.reshape(n, -1)).reshape(shape))
    return aggregate_iou(intersection.sum(axis=0), union.sum(axis=0)), boot, counts


def paired_tradeoff(selections, patients, profiles, names, repeats=2000, seed=42):
    S, boot, counts = bootstrap_profiles(profiles['intersection'], profiles['union'], repeats, seed)
    if len(names) < 2:
        raise ValueError('Selectivity requires at least two concepts.')
    rows, contrasts = [], []
    n, channels = len(counts[0]), len(S)
    for (concept, budget), group in selections.groupby(['concept', 'budget']):
        c = names.index(concept)
        other = [d for d in range(len(names)) if d != c]
        selectivity = S[:, c] - S[:, other].max(axis=1)
        boot_selectivity = boot[:, :, c] - boot[:, :, other].max(axis=2)
        estimates, draws = {}, {}
        for strategy, plans in group.groupby('strategy'):
            weights = np.zeros(channels)
            for plan in plans.itertuples():
                chosen = np.array(json.loads(plan.neurons), dtype=int)
                if len(chosen) != plan.k or len(np.unique(chosen)) != plan.k or (chosen < 0).any() or (chosen >= channels).any():
                    raise ValueError('Invalid saved neuron selection.')
                weights[chosen] += 1 / (len(plans) * len(chosen))
            data = patients[(patients.concept == concept) & (patients.budget == budget) & (patients.strategy == strategy)]
            pivot = data.pivot(index='sample_index', columns='repeat', values='kl')
            if pivot.index.tolist() != list(range(n)) or set(pivot.columns) != set(plans.repeat) or pivot.isna().any().any():
                raise ValueError('Patient/repeat alignment is incomplete.')
            kl = pivot.mean(axis=1).to_numpy()
            if not np.isclose(kl.mean(), plans.kl.mean(), atol=1e-7):
                raise ValueError('Saved selection and patient KL summaries disagree.')
            estimates[strategy] = {'iou': S[:, c] @ weights, 'selectivity': selectivity @ weights, 'kl': kl.mean()}
            draws[strategy] = {'iou': boot[:, :, c] @ weights, 'selectivity': boot_selectivity @ weights, 'kl': counts @ kl / n}
            for metric in estimates[strategy]:
                lo, hi = np.quantile(draws[strategy][metric], [.025, .975])
                rows.append({'concept': concept, 'budget': budget, 'k': int(plans.k.iloc[0]), 'strategy': strategy, 'metric': metric, 'estimate': estimates[strategy][metric], 'ci_low': lo, 'ci_high': hi, 'n_patients': n})
        for strategy in estimates:
            if strategy == 'joint':
                continue
            for metric in estimates[strategy]:
                delta = draws['joint'][metric] - draws[strategy][metric]
                lo, hi = np.quantile(delta, [.025, .975])
                contrasts.append({'concept': concept, 'budget': budget, 'strategy': strategy, 'metric': metric, 'joint_minus_control': estimates['joint'][metric] - estimates[strategy][metric], 'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def reliability(profiles, manifest, gamma, repeats=2000, seed=42):
    S, boot, counts = bootstrap_profiles(profiles['intersection'], profiles['union'], repeats, seed)
    per = aggregate_iou(profiles['intersection'], profiles['union'])
    patient = manifest.copy()
    patient['mean_channel_max_iou'] = per.max(axis=2).mean(axis=1)
    patient['labelled_channel_fraction'] = (per.max(axis=2) >= gamma).mean(axis=1)
    if 'activation_area' in profiles:
        patient['mean_activation_coverage'] = profiles['activation_area'].mean(axis=1) / profiles['image_pixels']
        for c in range(profiles['mask_area'].shape[1]):
            patient[f'concept_{c}_area_fraction'] = profiles['mask_area'][:, c] / profiles['image_pixels']
    rows = []
    for metric, value, draws in [('aggregate_labelled_fraction', (S.max(axis=1) >= gamma).mean(), (boot.max(axis=2) >= gamma).mean(axis=1)), ('aggregate_mean_max_iou', S.max(axis=1).mean(), boot.max(axis=2).mean(axis=1))]:
        lo, hi = np.quantile(draws, [.025, .975])
        rows.append({'metric': metric, 'estimate': value, 'ci_low': lo, 'ci_high': hi, 'n_patients': len(patient)})
    for metric in patient.columns.difference(manifest.columns):
        x = patient[metric].to_numpy()
        lo, hi = np.quantile(counts @ x / len(x), [.025, .975])
        rows.append({'metric': metric, 'estimate': x.mean(), 'ci_low': lo, 'ci_high': hi, 'n_patients': len(x)})
    return patient, pd.DataFrame(rows)


def graph_trace(selections, S, R, names, partition, tau):
    A = adjacency(R, S, tau)
    communities = dict(zip(partition.neuron, partition.community))
    nodes, edges = [], []
    for plan in selections.query("strategy == 'joint'").itertuples():
        chosen = json.loads(plan.neurons)
        c = names.index(plan.concept)
        scores = S[:, c] * (A @ S[:, c])
        for i in chosen:
            contributions = S[i, c] * A[i] * S[:, c]
            if not np.isclose(contributions.sum(), scores[i]):
                raise AssertionError('Edge evidence does not sum to joint score.')
            nodes.append({'concept': plan.concept, 'budget': plan.budget, 'neuron': i, 'community': communities.get(i, -1), 'joint_score': scores[i], **{f'train_{name}': S[i, d] for d, name in enumerate(names)}})
            for j in np.flatnonzero(contributions):
                edges.append({'concept': plan.concept, 'budget': plan.budget, 'source': i, 'target': int(j), 'source_community': communities.get(i, -1), 'target_community': communities.get(j, -1), 'target_selected': int(j) in chosen, 'source_iou': S[i, c], 'target_iou': S[j, c], 'coactivation': A[i, j], 'contribution': contributions[j]})
    return pd.DataFrame(nodes), pd.DataFrame(edges, columns=['concept', 'budget', 'source', 'target', 'source_community', 'target_community', 'target_selected', 'source_iou', 'target_iou', 'coactivation', 'contribution'])
