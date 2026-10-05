# scag/pipeline.py

import json
import time
from collections import Counter
import numpy as np
import pandas as pd
from .models import build_model
from .activations import extract_activations
from .data import phase_encode_segments
from .semantics import compute_neuron_concept_matrix, make_semantic_views
from .matrices import compute_R, compute_Q, align_R_Q, compute_edge_reliability
from .graph import analyze_graph, density_threshold, seed_stability, community_cards, area_coupling
from .ablation import community_ablation

def compute_area_profiles(segments, num_concepts):
    seg = segments.numpy()
    profiles = np.zeros((seg.shape[0], num_concepts), dtype=np.float32)
    for concept_id in range(1, num_concepts):
        profiles[:, concept_id] = (seg == concept_id).mean(axis=(1, 2, 3))
    return profiles

def run_semantic_granularity(tag, R_train, R_eval, X, labeled, tau_dm, cfg):
    """

    Compare the same coactivation graph under:

      - anatomy

      - phase

      - anatomy × phase

    The labeled neuron set is fixed.

    """
    views = make_semantic_views(X)
    lab_idx = np.where(labeled)[0]
    R_train_lab = R_train[np.ix_(lab_idx, lab_idx)]
    R_eval_lab = R_eval[np.ix_(lab_idx, lab_idx)]
    rows = []
    summary = {}
    for level, view in views.items():
        X_view = view['X']
        names = view['names']
        categories = view['categories']
        dominant = X_view[lab_idx].argmax(axis=1)
        counts = Counter(dominant.tolist())
        readable_counts = {names[int(cid)]: int(count) for cid, count in sorted(counts.items())}
        print('\n' + '-' * 72)
        print(f'SEMANTIC LEVEL: {level.upper()}')
        print('-' * 72)
        print('Dominant concepts:', readable_counts)
        # Semantic similarity for the same labeled neurons
        Q = compute_Q(X_view[lab_idx])
        # Full-matrix R-Q alignment
        align_train = align_R_Q(R_train_lab, Q, n_perm=cfg.n_mantel_permutations)
        align_eval = align_R_Q(R_eval_lab, Q, n_perm=cfg.n_mantel_permutations)
        # Fixed-threshold graph
        fixed, artifacts = analyze_graph(R_eval, X_view, labeled, tau=cfg.tau, n_perm=cfg.n_battery_permutations)
        # Density-matched graph
        dm = {}
        if np.isfinite(tau_dm):
            dm, _ = analyze_graph(R_eval, X_view, labeled, tau=tau_dm, n_perm=cfg.n_battery_permutations)
        # Semantic enrichment for this interpretation
        community_cards(artifacts, names, categories)
        row = {
            'tag': tag,
            'dataset': cfg.dataset,
            'semantic_level': level,
            'n_labeled': len(lab_idx),
            'n_effective_classes': len(counts),
            'dominant_counts': json.dumps(readable_counts),
            'alignment_train_r': align_train[0],
            'alignment_train_z': align_train[1],
            'alignment_train_p': align_train[2],
            'alignment_eval_r': align_eval[0],
            'alignment_eval_z': align_eval[1],
            'alignment_eval_p': align_eval[2],
            'eval_purity': fixed['purity'],
            'eval_purity_z': fixed['purity_z'],
            'eval_ami': fixed['ami'],
            'eval_ami_z': fixed['ami_z'],
            'eval_assort': fixed['assort'],
            'eval_assort_z': fixed['assort_z'],
            'eval_soft_delta': fixed['soft_delta'],
            'eval_soft_z': fixed['soft_z'],
            'eval_alignment_tau_r': fixed['alignment_tau_r'],
            'eval_alignment_tau_z': fixed['alignment_tau_z'],
            'eval_modularity': fixed['modularity'],
            'dm_ami_z': dm.get('ami_z', np.nan),
            'dm_assort_z': dm.get('assort_z', np.nan),
            'dm_soft_z': dm.get('soft_z', np.nan),
            'dm_modularity': dm.get('modularity', np.nan),
        }
        rows.append(row)
        prefix = {'anatomy': 'sem_anatomy', 'phase': 'sem_phase', 'anatomy_phase': 'sem_joint'}[level]
        summary.update({
            f'{prefix}_n_classes': len(counts),
            f'{prefix}_alignment_r': round(float(align_eval[0]), 4),
            f'{prefix}_alignment_z': round(float(align_eval[1]), 4),
            f'{prefix}_ami_z': round(float(fixed['ami_z']), 4),
            f'{prefix}_assort_z': round(float(fixed['assort_z']), 4),
            f'{prefix}_soft_z': round(float(fixed['soft_z']), 4),
            f'{prefix}_dm_ami_z': round(float(dm.get('ami_z', np.nan)), 4),
            f'{prefix}_dm_assort_z': round(float(dm.get('assort_z', np.nan)), 4),
            f'{prefix}_dm_soft_z': round(float(dm.get('soft_z', np.nan)), 4),
        })
    return (pd.DataFrame(rows), summary, views)

def run_model(tag, train_loader, eval_loader, cfg):
    """

    Run the complete SCAG analysis for one trained architecture.

    """
    start = time.perf_counter()
    # ============================================================
    # 1. MODEL
    # ============================================================
    model, hook_target, layer_name = build_model(tag, cfg)
    model = model.to(cfg.device).eval()
    # ============================================================
    # 2. ACTIVATION EXTRACTION
    # ============================================================
    train_acts, train_segs = extract_activations(model, train_loader, hook_target, cfg.device)
    eval_acts, eval_segs = extract_activations(model, eval_loader, hook_target, cfg.device)
    if train_acts is None or train_segs is None:
        raise RuntimeError('Training extraction did not return activations and segmentations.')
    if eval_acts is None or eval_segs is None:
        raise RuntimeError('Evaluation extraction did not return activations and segmentations.')
    train_segs = phase_encode_segments(train_segs)
    eval_segs = phase_encode_segments(eval_segs)
    n_filters = train_acts.shape[1]
    print(f'[{tag}] train activations={tuple(train_acts.shape)}, eval activations={tuple(eval_acts.shape)}')
    # ============================================================
    # 3. NEURON-CONCEPT ASSOCIATION X
    # ============================================================
    X, labeled = compute_neuron_concept_matrix(train_acts, train_segs, num_concepts=cfg.num_concepts, alpha=cfg.alpha, eta=cfg.eta)
    lab_idx = np.where(labeled)[0]
    label_counts = Counter(X[lab_idx].argmax(axis=1).tolist())
    readable_counts = {cfg.concept_name_map[int(cid)]: int(count) for cid, count in sorted(label_counts.items())}
    print(f'[{tag}] labeled neurons: {len(lab_idx)}/{n_filters} ({100 * labeled.mean():.1f}%)')
    print(f'[{tag}] dominant concepts:', readable_counts)
    if len(lab_idx) < cfg.min_labeled_neurons or len(label_counts) < 2:
        raise RuntimeError(f'Semantic eligibility failed: {len(lab_idx)} labeled neurons, {len(label_counts)} concepts.')
    # ============================================================
    # 4. COACTIVATION MATRICES R
    # ============================================================
    R_train = compute_R(train_acts)
    R_eval = compute_R(eval_acts)
    # ============================================================
    # 5. SEMANTIC SIMILARITY Q
    # ============================================================
    Q = compute_Q(X[lab_idx])
    # ============================================================
    # 6. R-Q ALIGNMENT
    # ============================================================
    R_train_lab = R_train[np.ix_(lab_idx, lab_idx)]
    R_eval_lab = R_eval[np.ix_(lab_idx, lab_idx)]
    alignment_train = align_R_Q(R_train_lab, Q, n_perm=cfg.n_mantel_permutations)
    alignment_eval = align_R_Q(R_eval_lab, Q, n_perm=cfg.n_mantel_permutations)
    print(f'[{tag}] R-Q alignment train r={alignment_train[0]:.4f}, eval r={alignment_eval[0]:.4f}')
    # ============================================================
    # 7. TRAIN-EVALUATION EDGE RELIABILITY
    # ============================================================
    edge_reliability = compute_edge_reliability(R_train, R_eval, lab_idx)
    print(f'[{tag}] edge reliability: rho={edge_reliability:.4f}')
    # ============================================================
    # 8. FIXED-THRESHOLD GRAPH
    # ============================================================
    graph_train, artifacts_train = analyze_graph(R_train, X, labeled, tau=cfg.tau, n_perm=cfg.n_battery_permutations)
    graph_eval, artifacts_eval = analyze_graph(R_eval, X, labeled, tau=cfg.tau, n_perm=cfg.n_battery_permutations)
    # ============================================================
    # 9. DENSITY-MATCHED GRAPH
    # ============================================================
    tau_dm = density_threshold(R_eval, cfg.graph_density)
    graph_dm, artifacts_dm = analyze_graph(R_eval, X, labeled, tau=tau_dm, n_perm=cfg.n_battery_permutations)
    print(f'[{tag}] density-matched tau*={tau_dm:.4f}')
    # ============================================================
    # 10. SEMANTIC GRANULARITY
    # ============================================================
    granularity_df, granularity_summary, semantic_views = run_semantic_granularity(tag=tag, R_train=R_train, R_eval=R_eval, X=X, labeled=labeled, tau_dm=tau_dm, cfg=cfg)
    # ============================================================
    # 11. LOUVAIN STABILITY
    # ============================================================
    ami_mean, ami_sd, partition_stability = seed_stability(artifacts_eval['G'], artifacts_eval['lab_nodes'], artifacts_eval['labels'], n_seeds=cfg.n_louvain_seeds)
    # ============================================================
    # 12. COMMUNITY ENRICHMENT
    # ============================================================
    enrichment_df = community_cards(artifacts_eval, cfg.concept_name_map, cfg.concept_category_map)
    # ============================================================
    # 13. ANATOMY-AREA COUPLING
    # ============================================================
    area_profiles = compute_area_profiles(train_segs, cfg.num_concepts)
    area_stats = area_coupling(R_eval, X, labeled, area_profiles, n_perm=cfg.n_cooccurrence_permutations)
    # ============================================================
    # 14. CAUSAL COMMUNITY ABLATION
    # ============================================================
    ablation_df = None
    if cfg.do_ablation:
        node2label = dict(zip(artifacts_eval['lab_nodes'], artifacts_eval['labels'].tolist()))
        ablation_df = community_ablation(model=model, hook_target=hook_target, loader=eval_loader, communities=artifacts_eval['communities'], num_filters=n_filters, device=cfg.device, val_segs=eval_segs, node2label=node2label, concept_names=cfg.concept_name_map, n_random=cfg.n_ablation_samples)
    # ============================================================
    # 15. ABLATION SUMMARY
    # ============================================================
    specificity_mean = np.nan
    specificity_ratio = np.nan
    if ablation_df is not None and 'specificity' in ablation_df:
        specificity_mean = float(ablation_df['specificity'].mean())
    if ablation_df is not None and 'spec_ratio' in ablation_df:
        specificity_ratio = float(ablation_df['spec_ratio'].mean())
    # ============================================================
    # 16. SUMMARY
    # ============================================================
    summary = {
        'dataset': cfg.dataset,
        'tag': tag,
        'layer': layer_name,
        'validation_source': cfg.validation_source,
        'n_train': train_acts.shape[0],
        'n_eval': eval_acts.shape[0],
        'n_filters': n_filters,
        'n_labeled': len(lab_idx),
        'edge_reliability': round(float(edge_reliability), 4),
        'alignment_train_r': round(float(alignment_train[0]), 4),
        'alignment_train_z': round(float(alignment_train[1]), 4),
        'alignment_train_p': float(alignment_train[2]),
        'alignment_eval_r': round(float(alignment_eval[0]), 4),
        'alignment_eval_z': round(float(alignment_eval[1]), 4),
        'alignment_eval_p': float(alignment_eval[2]),
        'eval_purity': round(graph_eval['purity'], 4),
        'eval_purity_z': round(graph_eval['purity_z'], 4),
        'eval_ami': round(graph_eval['ami'], 4),
        'eval_ami_z': round(graph_eval['ami_z'], 4),
        'eval_assort': round(graph_eval['assort'], 4),
        'eval_assort_z': round(graph_eval['assort_z'], 4),
        'eval_soft_delta': round(graph_eval['soft_delta'], 4),
        'eval_soft_z': round(graph_eval['soft_z'], 4),
        'eval_alignment_tau_r': round(graph_eval['alignment_tau_r'], 4),
        'eval_modularity': round(graph_eval['modularity'], 4),
        'eval_n_comm': graph_eval['n_comm'],
        'tau_dm': round(float(tau_dm), 4),
        'dm_n_edges': graph_dm['n_edges'],
        'dm_ami_z': round(graph_dm['ami_z'], 4),
        'dm_assort_z': round(graph_dm['assort_z'], 4),
        'dm_soft_z': round(graph_dm['soft_z'], 4),
        'dm_modularity': round(graph_dm['modularity'], 4),
        'ami_seed_mean': round(ami_mean, 4),
        'ami_seed_sd': round(ami_sd, 4),
        'partition_stability': round(partition_stability, 4),
        'pos_area_z': area_stats.get('positive', (0, 0, np.nan, 0))[2],
        'anti_area_z': area_stats.get('anti', (0, 0, np.nan, 0))[2],
        'abl_kl_ratio': round(float((ablation_df['kl'] / ablation_df['rnd']).mean()), 2) if ablation_df is not None and len(ablation_df) else np.nan,
        'abl_spec_mean': round(specificity_mean, 4) if np.isfinite(specificity_mean) else np.nan,
        'abl_spec_ratio': round(specificity_ratio, 2) if np.isfinite(specificity_ratio) else np.nan,
        'minutes': round((time.perf_counter() - start) / 60, 1),
    }
    summary.update(granularity_summary)
    # ============================================================
    # 17. RETURN ALL USEFUL ARTIFACTS
    # ============================================================
    return {
        'summary': summary,
        'granularity': granularity_df,
        'X': X,
        'labeled': labeled,
        'R_train': R_train,
        'R_eval': R_eval,
        'Q': Q,
        'semantic_views': semantic_views,
        'alignment_train': alignment_train,
        'alignment_eval': alignment_eval,
        'enrichment': enrichment_df,
        'area_stats': area_stats,
        'ablation': ablation_df,
        'graph_train': graph_train,
        'graph_eval': graph_eval,
        'graph_dm': graph_dm,
        'artifacts_train': artifacts_train,
        'artifacts_eval': artifacts_eval,
        'artifacts_dm': artifacts_dm,
    }
