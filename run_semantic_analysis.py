"""Reproducible SENCoG research experiments with separate train/val/test cohorts."""

import argparse
import gc
import json
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

from scag.ablation import ablation_uncertainty, concept_ablation
from scag.activations import extract_activations
from scag.config import SCAGConfig
from scag.matrices import compute_R
from scag.relations import adjacency, community_organisation, concept_relations, concept_subgraph, fdr_bh, semantic_functional_pairs
from scag.research import bootstrap_relations, cache_patients, concept_evidence, file_sha256, relation_stability, validate_manifests
from scag.semantics import concept_masks, concept_profiles


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["acdc", "mnms2"], default="acdc")
    parser.add_argument("--models", nargs="+", default=["resnet18"], help="Model names or all.")
    parser.add_argument("--layers", nargs="+", default=["last"], help="last, all checkpoint feature stages, or explicit spatial module names.")
    parser.add_argument("--semantic-level", choices=["anatomy", "phase", "anatomy_phase"], default="anatomy")
    parser.add_argument("--tau", type=float, default=0.30, help="Prespecified primary threshold; never selected on test results.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.30, 0.35, 0.40], help="Descriptive graph sensitivity thresholds; ablation uses tau only.")
    parser.add_argument("--gamma", type=float, default=0.015)
    parser.add_argument("--similarity", type=float, default=0.8)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--bootstraps", type=int, default=2000, help="Patient bootstrap and paired sign-flip repeats for ablation.")
    parser.add_argument("--relation-bootstraps", type=int, default=200, help="Patient bootstrap correlations for primary-threshold relation CIs.")
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--intervention-batch", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Four patients per split and reduced repetitions; execution validation only.")
    parser.add_argument("--output", type=Path, default=Path("scag_outputs/research"))
    args = parser.parse_args()
    if not 0 <= args.tau <= 1 or not 0 < args.gamma <= 1 or not 0 <= args.similarity <= 1 or any(not 0 <= t <= 1 for t in args.thresholds):
        parser.error("Thresholds/similarity must be in [0,1], gamma in (0,1].")
    if min(args.permutations, args.random_repeats, args.bootstraps, args.relation_bootstraps, args.batch_size, args.intervention_batch) < 1 or args.neighbors < 0 or any(not 0 < b <= 1 for b in args.budgets):
        parser.error("Counts must be positive, neighbors nonnegative, budgets in (0,1].")
    if len(args.layers) > 1 and any(name in {"all", "last"} for name in args.layers):
        parser.error("Use all/last alone or explicit module names.")
    if "all" in args.models and args.models != ["all"]:
        parser.error("Use --models all alone.")
    args.thresholds = sorted(set([args.tau, *args.thresholds]))
    args.budgets = sorted(set(args.budgets))
    if args.smoke:
        args.permutations, args.random_repeats, args.bootstraps, args.relation_bootstraps = 19, 3, 99, 19
        args.thresholds, args.budgets = [args.tau], [0.1]
    return args


def layer_names(model, requested):
    candidates = model.feature_layer_candidates
    names = candidates if requested == ["all"] else candidates[-1:] if requested == ["last"] else requested
    names = list(dict.fromkeys(names))
    if not names or any(name not in dict(model.named_modules()) for name in names):
        raise ValueError(f"Unknown or empty layer selection: {names}")
    return names


def sample_manifest(dataset, split):
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        frame = pd.concat([sample_manifest(part, getattr(part, "split", split)) for part in dataset.datasets], ignore_index=True)
    elif isinstance(dataset, torch.utils.data.Subset):
        frame = sample_manifest(dataset.dataset, split).iloc[dataset.indices].reset_index(drop=True)
    else:
        if not hasattr(dataset, "metadata") or "pid" not in dataset.metadata:
            raise ValueError("Patient metadata is required for evidence and split validation.")
        columns = [name for name in ("pid", "pathology") if name in dataset.metadata]
        frame = dataset.metadata[columns].astype(str).reset_index(drop=True)
        frame["split"] = getattr(dataset, "split", split)
        frame["data_dir"] = str(dataset.data_dir)
    frame["sample_index"] = np.arange(len(frame))
    return frame


def analyze_layer(model, layer, loaders, cfg, args, directory, manifests):
    from scag.relation_plots import plot_ablation, plot_layer, plot_overlays

    directory.mkdir(parents=True, exist_ok=False)
    target = model.get_submodule(layer)
    train_S, thresholds, train_A, train_partition, activation_magnitude = None, None, None, None, None
    tables, enrichments, ablation_tables, contrast_tables, stability_tables, sve_tables = [], [], [], [], [], []
    arrays, selections = {}, None
    for split, loader in loaders.items():
        print(f"  {split}: extraction and calibrated concept IoUs", flush=True)
        acts, segments = extract_activations(model, loader, target, args.device)
        masks, names = concept_masks(segments, args.semantic_level)
        profiles = concept_profiles(acts, masks, cfg.alpha, thresholds, args.device)
        S_split = profiles["S"]
        R = compute_R(acts)
        pooled = acts.mean(dim=(2, 3)).numpy()
        if split == "train":
            train_S, thresholds = S_split, profiles["thresholds"]
            activation_magnitude = acts.abs().mean(dim=(0, 2, 3)).numpy()
            arrays.update(S=train_S, thresholds=thresholds, concepts=np.array(names), activation_magnitude=activation_magnitude)
        arrays.update({f"R_{split}": R, f"S_{split}": S_split, f"pooled_{split}": pooled})
        concept_evidence(profiles, train_S, names, manifests[split]).to_csv(directory / f"concept_evidence_{split}.csv", index=False)
        plot_overlays(acts, masks, profiles, train_S, names, loader, manifests[split], directory, split)
        np.savez_compressed(directory / f"patient_concepts_{split}.npz", per_patient_iou=profiles["per_patient_iou"], intersection=profiles["intersection"], union=profiles["union"])
        del acts, segments, masks
        gc.collect()
        for tau in args.thresholds:
            print(f"  {split}: graph/null analysis tau={tau:g}", flush=True)
            A = adjacency(R, train_S, tau)
            relations = concept_relations(A, train_S, names, args.permutations, args.seed, rewire=True)
            relations.insert(0, "split", split)
            relations.insert(0, "tau", tau)
            if tau == args.tau:
                intervals = bootstrap_relations(pooled, train_S, names, tau, args.relation_bootstraps, args.seed)
                relations = relations.merge(intervals, on=["concept_c", "concept_d"], validate="one_to_one")
            tables.append(relations)
        A = adjacency(R, train_S, args.tau)
        if split == "train":
            train_A = A
        primary = tables[-len(args.thresholds) + args.thresholds.index(args.tau)]
        pairs, pair_summary = semantic_functional_pairs(R, train_S, args.tau, args.similarity, args.permutations, args.seed)
        pairs.to_csv(directory / f"pairs_{split}.csv.gz", index=False)
        pair_summary.to_csv(directory / f"pair_categories_{split}.csv", index=False)
        enrichment, sve, partition = community_organisation(A, S_split, names, args.gamma, args.permutations, args.seed)
        enrichment.insert(0, "split", split)
        enrichment.insert(0, "partition_source", split)
        enrichments.append(enrichment)
        sve.insert(0, "split", split)
        sve.insert(0, "partition_source", split)
        sve_tables.append(sve)
        partition.to_csv(directory / f"communities_{split}.csv", index=False)
        if split == "train":
            train_partition = partition
        else:
            fixed_enrichment, fixed_sve, _ = community_organisation(A, S_split, names, args.gamma, args.permutations, args.seed, partition=train_partition)
            for frame in (fixed_enrichment, fixed_sve):
                frame.insert(0, "split", split)
                frame.insert(0, "partition_source", "train")
            enrichments.append(fixed_enrichment)
            sve_tables.append(fixed_sve)
            stability = relation_stability(train_A, A, train_S, S_split)
            common = train_partition.merge(partition, on="neuron", suffixes=("_train", "_split"))
            stability["partition_ami"] = adjusted_mutual_info_score(common.community_train, common.community_split) if len(common) >= 2 else np.nan
            stability["partition_common_neurons"] = len(common)
            stability["split"] = split
            stability_tables.append(stability)
        for concept in names:
            nodes, edges = concept_subgraph(A, train_S, names, concept, args.gamma, args.neighbors)
            nodes.to_csv(directory / f"query_{concept}_{split}_nodes.csv", index=False)
            edges.to_csv(directory / f"query_{concept}_{split}_edges.csv", index=False)
        plot_layer(primary, pairs, A, train_S, names, args, directory, split)
        if split != "train" and not args.skip_ablation:
            ablation, patients = concept_ablation(model, target, loader, train_A, train_S, names, args.device, tuple(args.budgets), args.random_repeats, args.seed, activation_magnitude, args.intervention_batch)
            identity = ablation[["concept", "strategy", "budget", "repeat", "neurons"]]
            if selections is not None and not identity.equals(selections):
                raise RuntimeError("Validation and test intervention selections differ.")
            selections = identity
            ablation["heldout_mean_concept_iou"] = [float(S_split[json.loads(row.neurons), names.index(row.concept)].mean()) for row in ablation.itertuples()]
            summary, contrasts = ablation_uncertainty(patients, args.bootstraps, args.seed)
            summary["split"], contrasts["split"] = split, split
            ablation_tables.append(summary)
            diagnostics = ablation[ablation.strategy == "matched"].groupby(["concept", "budget", "k"]).agg(matched_valid_fraction=("match_valid", "mean"), matched_max_gap=("match_max_percentile_gap", "max"), matched_mean_overlap=("match_overlap_fraction", "mean")).reset_index()
            contrasts = contrasts.merge(diagnostics, on=["concept", "budget", "k"], validate="many_to_one")
            contrast_tables.append(contrasts)
            ablation.to_csv(directory / f"ablation_selections_{split}.csv", index=False)
            patients.merge(manifests[split], on="sample_index", validate="many_to_one").to_csv(directory / f"ablation_patients_{split}.csv.gz", index=False)
            plot_ablation(summary, directory, split)
    np.savez_compressed(directory / "representations.npz", **arrays)
    results = {"relations": pd.concat(tables, ignore_index=True), "enrichment": pd.concat(enrichments, ignore_index=True), "community_sve": pd.concat(sve_tables, ignore_index=True), "stability": pd.concat(stability_tables, ignore_index=True)}
    if ablation_tables:
        results.update(ablation_summary=pd.concat(ablation_tables, ignore_index=True), ablation_contrasts=pd.concat(contrast_tables, ignore_index=True))
    for name, frame in results.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    return results


def write_results(collected, output, args):
    from scag.relation_plots import plot_comparison

    combined = {name: pd.concat(tables, ignore_index=True) for name, tables in collected.items()}
    for name, table in combined.items():
        for p_column in [column for column in table if column == "p" or column.endswith("_p")]:
            table[p_column.removesuffix("p") + "q_run"] = fdr_bh(table[p_column].to_numpy())
        table.to_csv(output / f"{name}_all.csv", index=False)
    if "relations" in combined:
        plot_comparison(combined["relations"].query("tau == @args.tau"), output)
    lines = ["# SENCoG experiment results", "", f"Execution-only smoke run: {args.smoke}.", "", "Train fits thresholds, semantics, graph rankings and control matches. Validation and test are reported separately. Test patients have been inspected previously in this project; this run does not restore an untouched confirmatory test set.", "", f"Primary threshold: {args.tau}. Threshold sensitivity: {args.thresholds}. Interventions use the primary threshold only.", "", "Read ablation_summary_all.csv for patient-bootstrap KL/flip intervals and ablation_contrasts_all.csv for paired joint-minus-control effects and two-sided sign-flip p values. q_run corrects across the requested run for each test family. CIs are pointwise and conditional on the trained model/rankings and sampled control sets.", "", "Matched comparisons require matched_valid_fraction = 1; otherwise the output documents insufficient matches and cannot support a fully matched-control claim. Positive disruption is sensitivity, not proof of concept-specific causality.", "", "Relation nulls: complete-profile permutation; degree-stratified permutation; uniform edge-position rewiring (preserves edge count/weight multiset, not degree). Relation bootstrap intervals condition on training semantics and resample patients to rebuild correlations. SVE fixed-partition results use new split-specific IoUs with frozen training thresholds.", ""]
    if "ablation_contrasts" in combined:
        final = combined["ablation_contrasts"].query("split == 'test'")
        columns = ["model", "layer", "concept", "budget", "metric", "strategy", "joint_minus_control", "ci_low", "ci_high", "q_run", "matched_valid_fraction"]
        lines.extend(["Test comparisons (see CSV for full precision):", "", "```text", final[columns].round(5).to_string(index=False), "```", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    from scag.data import load_datasets
    from scag.models import build_model

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    requested_device = args.device
    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Activate cinema and run outside a GPU-restricted sandbox.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = SCAGConfig(dataset=args.dataset, batch_size=args.batch_size)
    tags = cfg.architectures if args.models == ["all"] else args.models
    if any(tag not in cfg.architectures for tag in tags):
        raise ValueError(f"Models must be chosen from {cfg.architectures}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output / cfg.dataset / args.semantic_level / run_id
    output.mkdir(parents=True, exist_ok=False)
    print(f"Device: {args.device}" + (f" ({torch.cuda.get_device_name(0)})" if args.device == "cuda" else ""), flush=True)
    print(f"Results: {output}", flush=True)
    source_paths = [Path(__file__), *Path(__file__).parent.joinpath("scag").glob("*.py")]
    protocol = {
        **vars(args), "output": str(output), "requested_device": requested_device,
        "python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__, "pandas": pd.__version__, "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "source_sha256": {str(path.relative_to(Path(__file__).parent)): file_sha256(path) for path in source_paths},
        "semantic_definition": "Direct spatial IoU, strict activation > training percentile; anatomy uses ED/ES mask union, phase uses annotated foreground.",
        "split_policy": "No pooled validation/test. All selections and thresholds fixed before test; prior test exploration acknowledged.",
        "primary_budget": 0.1, "alpha": cfg.alpha, "model_seed": cfg.model_seed,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    train, val, test, _ = load_datasets(cfg)
    datasets = {"train": train, "val": val, "test": test}
    if args.smoke:
        datasets = {split: torch.utils.data.Subset(dataset, range(min(4, len(dataset)))) for split, dataset in datasets.items()}
    manifests = {split: sample_manifest(dataset, split) for split, dataset in datasets.items()}
    validate_manifests(manifests)
    loaders = {}
    for split, dataset in datasets.items():
        print(f"Caching {split}: {len(dataset)} patients and checking IDs", flush=True)
        loaders[split] = cache_patients(dataset, manifests[split], args.batch_size)
        manifests[split].to_csv(output / f"{split}_samples.csv", index=False)
    collected, failures = {}, []
    for tag in tags:
        try:
            model, _, _ = build_model(tag, cfg)
            model.to(args.device).eval()
            checkpoint = cfg.model_root / tag / cfg.formulation / cfg.initialization / f"seed_{cfg.model_seed}" / "best_val_mcc.pt"
            model_directory = output / tag
            model_directory.mkdir()
            (model_directory / "checkpoint.json").write_text(json.dumps({"path": str(checkpoint), "sha256": file_sha256(checkpoint)}, indent=2) + "\n")
            for layer in layer_names(model, args.layers):
                print(f"[{tag}/{layer}] Starting", flush=True)
                try:
                    with threadpool_limits(limits=4):
                        results = analyze_layer(model, layer, loaders, cfg, args, model_directory / layer, manifests)
                    for name, frame in results.items():
                        frame.insert(0, "layer", layer)
                        frame.insert(0, "model", tag)
                        frame.insert(0, "dataset", cfg.dataset)
                        collected.setdefault(name, []).append(frame)
                    write_results(collected, output, args)
                except Exception:
                    failures.append({"model": tag, "layer": layer, "traceback": traceback.format_exc()})
                    print(failures[-1]["traceback"], flush=True)
                finally:
                    gc.collect()
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
            del model
        except Exception:
            failures.append({"model": tag, "traceback": traceback.format_exc()})
            print(failures[-1]["traceback"], flush=True)
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} failures; see {output / 'failures.json'}")
    print(f"Complete. Open {output / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
