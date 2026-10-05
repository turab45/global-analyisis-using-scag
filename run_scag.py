import gc
import traceback

import numpy as np
import pandas as pd
import torch

from scag.config import SCAGConfig
from scag.data import build_graph_analysis_cohort, build_loaders, load_mnms2_datasets, make_split_safe
from scag.pipeline import run_model


def main():
    cfg = SCAGConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, _ = load_mnms2_datasets(cfg)
    eval_dataset = build_graph_analysis_cohort(val_dataset, test_dataset)
    eval_dataset = make_split_safe(train_dataset, eval_dataset)
    train_loader, eval_loader = build_loaders(
        train_dataset, eval_dataset, cfg.batch_size, cfg.num_workers
    )

    all_results, all_granularity = [], []

    for tag in cfg.architectures:
        print(f"\n{'=' * 72}\n[{tag}] SCAG\n{'=' * 72}")
        try:
            result = run_model(tag, train_loader, eval_loader, cfg)
            all_results.append(result["summary"])
            all_granularity.append(result["granularity"])

            np.savez_compressed(
                cfg.output_dir / f"{tag}_semantics.npz",
                X=result["X"], labeled=result["labeled"],
                R_train=result["R_train"].astype(np.float32),
                R_eval=result["R_eval"].astype(np.float32),
                X_anatomy=result["semantic_views"]["anatomy"]["X"],
                X_phase=result["semantic_views"]["phase"]["X"],
                X_anatomy_phase=result["semantic_views"]["anatomy_phase"]["X"],
            )
            result["granularity"].to_csv(
                cfg.output_dir / f"{tag}_semantic_granularity.csv", index=False
            )
            if result["enrichment"] is not None:
                result["enrichment"].to_csv(cfg.output_dir / f"{tag}_enrichment.csv", index=False)
            if result["ablation"] is not None:
                result["ablation"].to_csv(cfg.output_dir / f"{tag}_ablation.csv", index=False)

            pd.DataFrame(all_results).to_csv("big_run_results_mnms2_v4.csv", index=False)
            pd.concat(all_granularity, ignore_index=True).to_csv(
                "semantic_granularity_all_mnms2_v4.csv", index=False
            )
            print(f"[{tag}] DONE in {result['summary']['minutes']} min")

        except Exception:
            print(f"[{tag}] SKIPPED — failed during run:\n{traceback.format_exc()}")
        finally:
            gc.collect()
            if cfg.device.type == "cuda":
                torch.cuda.empty_cache()
            elif cfg.device.type == "mps":
                torch.mps.empty_cache()

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\nFINAL SUMMARY")
        print(results_df.to_string(index=False))
        results_df.to_csv("big_run_results_mnms2_v4.csv", index=False)

    if all_granularity:
        granularity_df = pd.concat(all_granularity, ignore_index=True)
        cols = [
            "tag", "semantic_level", "n_labeled", "n_effective_classes",
            "alignment_eval_r", "alignment_eval_z", "eval_ami", "eval_ami_z",
            "eval_assort", "eval_assort_z", "eval_soft_delta", "eval_soft_z",
            "dm_ami_z", "dm_assort_z", "dm_soft_z",
        ]
        print("\nANATOMY vs PHASE vs ANATOMY×PHASE")
        print(granularity_df[cols].round(4).to_string(index=False))
        granularity_df.to_csv("semantic_granularity_all_mnms2_v4.csv", index=False)


if __name__ == "__main__":
    main()
