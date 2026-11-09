# scag/ablation.py

from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler

from .relations import concept_rankings


def predict_ablation(model, hook_target, loader, num_filters, device, ablate=None):
    cache, handle = [], None
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("Ablation requires a sequential loader to pair identical patients.")
    if ablate is not None and len(ablate):
        channels = torch.zeros(num_filters, dtype=torch.bool, device=device)
        channels[list(ablate)] = True

        def zero_channels(module, inputs, output):
            output = output.clone()
            output[:, channels] = 0
            return output

        handle = hook_target.register_forward_hook(zero_channels)
    try:
        with torch.no_grad():
            for batch in loader:
                cache.append(torch.softmax(model(batch[0].to(device)), dim=1).cpu())
    finally:
        if handle is not None:
            handle.remove()
    return torch.cat(cache)


def matched_channels(selected, covariates, rng, caliper=0.1):
    """Randomized nearest matches in training covariate percentile space."""
    from scipy.stats import rankdata

    ranks = rankdata(covariates, axis=0) / len(covariates)
    available = np.ones(len(ranks), dtype=bool)
    if len(selected) * 2 <= len(ranks):
        available[selected] = False
    matches, gaps = [], []
    for target in rng.permutation(selected):
        candidates = np.flatnonzero(available)
        distances = np.max(np.abs(ranks[candidates] - ranks[target]), axis=1)
        close = candidates[distances <= caliper]
        chosen = rng.choice(close) if len(close) else candidates[np.argmin(distances)]
        matches.append(int(chosen))
        gaps.append(float(np.max(np.abs(ranks[chosen] - ranks[target]))))
        available[chosen] = False
    return np.array(matches), max(gaps, default=0), float(np.mean(np.isin(matches, selected)))


def predict_interventions(model, hook_target, loader, selections, device, intervention_batch=4):
    """Batch several channel interventions through the same cached patient images."""
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last or model.training:
        raise ValueError("Require eval mode and a sequential loader without dropped patients.")
    keys = list(selections)
    predictions = {key: [] for key in keys}
    for start in range(0, len(keys), intervention_batch):
        current = keys[start:start + intervention_batch]
        for batch in loader:
            images = batch[0].to(device)
            n = len(images)
            calls = 0

            def zero_channels(module, inputs, output):
                nonlocal calls
                calls += 1
                result = output.clone()
                for index, key in enumerate(current):
                    result[index * n:(index + 1) * n, list(key)] = 0
                return result

            handle = hook_target.register_forward_hook(zero_channels)
            try:
                with torch.inference_mode():
                    logits = model(images.repeat(len(current), 1, 1, 1))
                    probs = torch.softmax(logits, dim=1).cpu().reshape(len(current), n, -1)
                if calls != 1:
                    raise ValueError("Intervention layer must be called exactly once per forward pass.")
                for index, key in enumerate(current):
                    predictions[key].append(probs[index])
            finally:
                handle.remove()
    return {key: torch.cat(values) for key, values in predictions.items()}


def concept_ablation(model, hook_target, loader, A_train, S_train, names, device, budgets=(0.01, 0.05, 0.1), n_random=20, seed=42, activation_magnitude=None, intervention_batch=4):
    """Training-only selection; identical budgets, cached patients and interventions."""
    import json
    import networkx as nx

    if n_random < 1 or intervention_batch < 1 or not budgets or any(not 0 < fraction <= 1 for fraction in budgets):
        raise ValueError("Require positive repeats, intervention batch and budgets in (0, 1].")
    n = len(S_train)
    clean = predict_ablation(model, hook_target, loader, n, device)
    rng = np.random.default_rng(seed)
    plans = []
    if activation_magnitude is not None:
        if np.shape(activation_magnitude) != (n,) or not np.isfinite(activation_magnitude).all():
            raise ValueError("Require one finite training activation magnitude per channel.")
        pagerank = np.array(list(nx.pagerank(nx.from_numpy_array(A_train), weight="weight").values()))
    for c, concept in enumerate(names):
        rankings, scores = concept_rankings(A_train, S_train, c)
        if activation_magnitude is not None:
            scores.update({"pagerank": pagerank, "activation": activation_magnitude})
            rankings.update({key: np.argsort(-scores[key], kind="stable") for key in ("pagerank", "activation")})
        for fraction in budgets:
            k = max(1, int(np.ceil(n * fraction)))
            selections = [(strategy, 0, ranking[:k], np.nan, np.nan) for strategy, ranking in rankings.items()]
            selections.extend(("random", repeat, rng.choice(n, k, replace=False), np.nan, np.nan) for repeat in range(n_random))
            if activation_magnitude is not None:
                covariates = np.column_stack([activation_magnitude, A_train.sum(axis=1), S_train[:, c]])
                for repeat in range(n_random):
                    matched, gap, overlap = matched_channels(rankings["joint"][:k], covariates, rng)
                    selections.append(("matched", repeat, matched, gap, overlap))
            for strategy, repeat, selected, gap, overlap in selections:
                key = tuple(sorted(selected.tolist()))
                plans.append({
                    "concept": concept, "strategy": strategy, "budget": fraction, "k": k, "repeat": repeat,
                    "neurons": json.dumps(list(key)), "key": key,
                    "positive_score_selected": int((scores[strategy][selected] > 0).sum()) if strategy in scores else np.nan,
                    "mean_semantic_score": float(S_train[selected, c].mean()), "mean_strength": float(A_train[selected].sum(axis=1).mean()),
                    "mean_activation": float(np.mean(activation_magnitude[selected])) if activation_magnitude is not None else np.nan,
                    "match_max_percentile_gap": gap, "match_overlap_fraction": overlap,
                    "match_valid": bool(gap <= 0.1 and overlap == 0) if strategy == "matched" else None,
                })
    keys = dict.fromkeys(plan["key"] for plan in plans)
    print(f"  Ablation: {len(plans)} selections, {len(keys)} unique interventions, {len(loader.dataset)} patients", flush=True)
    predictions = predict_interventions(model, hook_target, loader, keys, device, intervention_batch)
    rows, patient_rows = [], []
    for plan in plans:
        key = plan.pop("key")
        ablated = predictions[key]
        kl = (clean * torch.log((clean + 1e-8) / (ablated + 1e-8))).sum(1).clamp_min(0).numpy()
        flip = (clean.argmax(1) != ablated.argmax(1)).numpy()
        rows.append({**plan, "kl": kl.mean(), "flip": flip.mean()})
        info = {field: plan[field] for field in ("concept", "strategy", "budget", "k", "repeat")}
        patient_rows.extend({**info, "sample_index": i, "kl": value, "flip": bool(flip[i])} for i, value in enumerate(kl))
    return pd.DataFrame(rows), pd.DataFrame(patient_rows)


def ablation_uncertainty(patients, n_bootstrap=2000, seed=42):
    """Patient bootstrap CIs; repeats are averaged within patient, never pooled as subjects."""
    from .relations import fdr_bh

    rng = np.random.default_rng(seed)
    summaries, contrasts = [], []
    for (concept, budget, k), group in patients.groupby(["concept", "budget", "k"]):
        averaged = group.groupby(["sample_index", "strategy"])[["kl", "flip"]].mean()
        for metric in ("kl", "flip"):
            values = averaged[metric].unstack("strategy")
            if values.isna().any().any() or len(values) < 2:
                raise ValueError("Every strategy requires the same two or more evaluation patients.")
            sample = rng.integers(len(values), size=(n_bootstrap, len(values)))
            signs = rng.choice([-1, 1], size=(n_bootstrap, len(values)))
            for strategy in values:
                x = values[strategy].to_numpy()
                ci = np.quantile(x[sample].mean(axis=1), [.025, .975])
                common = {"concept": concept, "budget": budget, "k": k, "metric": metric, "strategy": strategy, "n_patients": len(x)}
                summaries.append({**common, "mean": x.mean(), "ci_low": ci[0], "ci_high": ci[1]})
                if strategy != "joint":
                    delta = values["joint"].to_numpy() - x
                    bounds = np.quantile(delta[sample].mean(axis=1), [.025, .975])
                    p = (1 + (np.abs((signs * delta).mean(axis=1)) >= abs(delta.mean()) - 1e-12).sum()) / (n_bootstrap + 1)
                    contrasts.append({**common, "joint_minus_control": delta.mean(), "ci_low": bounds[0], "ci_high": bounds[1], "p": p})
    comparisons = pd.DataFrame(contrasts)
    comparisons["q"] = fdr_bh(comparisons.p.to_numpy())
    return pd.DataFrame(summaries), comparisons


def community_ablation(
    model, hook_target, loader, communities, num_filters, device,
    val_segs=None, node2label=None, concept_names=None,
    n_random=1000, seed=2,
):
    rng = np.random.default_rng(seed)

    def predict(ablate=None):
        return predict_ablation(model, hook_target, loader, num_filters, device, ablate)

    clean = predict()

    def kl_values(q):
        return (
            clean * torch.log((clean + 1e-8) / (q + 1e-8))
        ).sum(dim=1).numpy()

    rows = []

    for ci, community in enumerate(communities):
        if len(community) < 3 or ci >= 8:
            continue

        ablated = predict(community)
        delta = kl_values(ablated)

        random_delta = np.stack([
            kl_values(
                predict(rng.choice(num_filters, len(community), replace=False))
            )
            for _ in range(n_random)
        ])

        row = {
            "comm": ci,
            "size": len(community),
            "kl": float(delta.mean()),
            "rnd": float(random_delta.mean()),
            "rnd_sd": float(random_delta.std()),
            "flip": float((clean.argmax(1) != ablated.argmax(1)).float().mean()),
        }

        if val_segs is not None and node2label is not None:
            labeled_nodes = [n for n in community if n in node2label]

            if labeled_nodes:
                dominant = int(
                    Counter(node2label[n] for n in labeled_nodes)
                    .most_common(1)[0][0]
                )

                burden = (val_segs == dominant).float().mean(dim=(1, 2, 3)).numpy()
                high = burden > np.median(burden)

                if high.sum() >= 3 and (~high).sum() >= 3:
                    row.update({
                        "dom_concept": dominant,
                        "n_high": int(high.sum()),
                        "n_low": int((~high).sum()),
                        "kl_high": float(delta[high].mean()),
                        "kl_low": float(delta[~high].mean()),
                        "rnd_high": float(random_delta[:, high].mean()),
                        "rnd_low": float(random_delta[:, ~high].mean()),
                    })

                    row["specificity"] = row["kl_high"] - row["kl_low"]
                    row["spec_ratio"] = row["kl_high"] / (row["kl_low"] + 1e-9)

        rows.append(row)

        message = (
            f"  comm {ci:02d}: KL={row['kl']:.4f} "
            f"(random {row['rnd']:.4f}±{row['rnd_sd']:.4f})"
        )

        if "specificity" in row and concept_names is not None:
            message += (
                f" | [{concept_names[row['dom_concept']]}] "
                f"high/low={row['n_high']}/{row['n_low']} "
                f"KL={row['kl_high']:.3f}/{row['kl_low']:.3f} "
                f"spec={row['specificity']:+.3f}"
            )

        print(message)

    return pd.DataFrame(rows)
