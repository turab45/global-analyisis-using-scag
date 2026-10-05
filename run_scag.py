# run_scag.py

import gc
import traceback

import numpy as np
import pandas as pd
import torch

from scag.config import SCAGConfig
from scag.data import build_graph_analysis_cohort, make_split_safe, build_loaders
from scag.pipeline import run_model

# Import these from wherever your existing Block-0 code is moved.
from project_setup import (
    MNMS2SemanticDataset,
    EDES2DClassifier,
    MODEL_ROOT,
    SPEC,
    splits,
    train_dataset,
    val_dataset,
    eval_transform,
)


def main():
    cfg = SCAGConfig()
    cfg.output_dir.mkdir(exist_ok=True)

    # Graph-analysis cohort: official val + test
    eval_dataset = build_graph_analysis_cohort(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset_class=MNMS2SemanticDataset,
        splits=splits,
        eval_transform=eval_transform,
    )

    eval_dataset = make_split_safe(train_dataset, eval_dataset)

    train_loader, eval_loader = build_loaders(
        train_dataset,
        eval_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    all_results = []
    all_granularity = []

    for tag in cfg.architectures:
        print(f"\n{'=' * 72}")
        print(f"[{tag}] SCAG")
        print("=" * 72)

        try:
            result = run_model(
                tag=tag,
                train_loader=train_loader,
                val_loader=eval_loader,
                model_root=MODEL_ROOT,
                model_class=EDES2DClassifier,
                classes=SPEC.classes,
                cfg=cfg,
            )

            summary = result["summary"]
            all_results.append(summary)
            all_granularity.append(result["granularity"])

            # Semantic representations + R matrices
            np.savez_compressed(
                cfg.output_dir / f"{tag}_semantics.npz",
                X=result["X"],
                labeled=result["labeled"],
                R_train=result["R_train"].astype(np.float32),
                R_eval=result["R_eval"].astype(np.float32),
                X_anatomy=result["semantic_views"]["anatomy"]["X"],
                X_phase=result["semantic_views"]["phase"]["X"],
                X_anatomy_phase=result["semantic_views"]["anatomy_phase"]["X"],
            )

            # Semantic granularity
            result["granularity"].to_csv(
                cfg.output_dir / f"{tag}_semantic_granularity.csv",
                index=False,
            )

            # Community enrichment
            if result["enrichment"] is not None:
                result["enrichment"].to_csv(
                    cfg.output_dir / f"{tag}_enrichment.csv",
                    index=False,
                )

            # Community ablation
            if result["ablation"] is not None:
                result["ablation"].to_csv(
                    cfg.output_dir / f"{tag}_ablation.csv",
                    index=False,
                )

            # Continuously update summaries
            pd.DataFrame(all_results).to_csv(
                "big_run_results_mnms2_v4.csv",
                index=False,
            )

            pd.concat(all_granularity, ignore_index=True).to_csv(
                "semantic_granularity_all_mnms2_v4.csv",
                index=False,
            )

            print(f"[{tag}] DONE in {summary['minutes']} min")

        except Exception:
            print(
                f"[{tag}] SKIPPED — failed during run:\n"
                f"{traceback.format_exc()}"
            )

        finally:
            gc.collect()

            if cfg.device.type == "cuda":
                torch.cuda.empty_cache()

            elif cfg.device.type == "mps":
                torch.mps.empty_cache()

    # Final summaries
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        results_df.to_csv(
            "big_run_results_mnms2_v4.csv",
            index=False,
        )

    if all_granularity:
        granularity_df = pd.concat(
            all_granularity,
            ignore_index=True,
        )

        print("\n" + "=" * 72)
        print("ANATOMY vs PHASE vs ANATOMY×PHASE")
        print("=" * 72)

        cols = [
            "tag",
            "semantic_level",
            "n_labeled",
            "n_effective_classes",
            "alignment_eval_r",
            "alignment_eval_z",
            "eval_ami",
            "eval_ami_z",
            "eval_assort",
            "eval_assort_z",
            "eval_soft_delta",
            "eval_soft_z",
            "dm_ami_z",
            "dm_assort_z",
            "dm_soft_z",
        ]

        print(
            granularity_df[cols]
            .round(4)
            .to_string(index=False)
        )

        granularity_df.to_csv(
            "semantic_granularity_all_mnms2_v4.csv",
            index=False,
        )


if __name__ == "__main__":
    main()