# run_scag.py

import argparse
import gc
import traceback

import numpy as np
import pandas as pd
import torch

from scag.config import SCAGConfig
from scag.data import (
    load_datasets,
    build_graph_analysis_cohort,
    make_split_safe,
    build_loaders,
)
from scag.pipeline import run_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Semantic Coactivation Graph analysis."
    )

    parser.add_argument(
        "--dataset",
        choices=["mnms2", "acdc"],
        default="mnms2",
        help="Dataset to analyze.",
    )

    return parser.parse_args()


def save_model_outputs(tag, result, cfg):
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main semantic / functional representations
    np.savez_compressed(
        output_dir / f"{tag}_scag.npz",
        X=result["X"],
        labeled=result["labeled"],
        R_train=result["R_train"].astype(np.float32),
        R_eval=result["R_eval"].astype(np.float32),
        Q=result["Q"].astype(np.float32),
        X_anatomy=result["semantic_views"]["anatomy"]["X"],
        X_phase=result["semantic_views"]["phase"]["X"],
        X_anatomy_phase=result["semantic_views"]["anatomy_phase"]["X"],
    )

    # Semantic granularity
    result["granularity"].to_csv(
        output_dir / f"{tag}_semantic_granularity.csv",
        index=False,
    )

    # Community enrichment
    if result["enrichment"] is not None:
        result["enrichment"].to_csv(
            output_dir / f"{tag}_enrichment.csv",
            index=False,
        )

    # Community ablation
    if result["ablation"] is not None:
        result["ablation"].to_csv(
            output_dir / f"{tag}_ablation.csv",
            index=False,
        )


def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    args = parse_args()

    cfg = SCAGConfig(
        dataset=args.dataset
    )

    cfg.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SEMANTIC COACTIVATION GRAPH")
    print("=" * 72)
    print(f"Dataset: {cfg.dataset}")
    print(f"Device: {cfg.device}")
    print(f"Data: {cfg.data_root}")
    print(f"Models: {cfg.model_root}")
    print(f"Output: {cfg.output_dir}")
    print("=" * 72)

    # ============================================================
    # DATA
    # ============================================================

    train_dataset, val_dataset, test_dataset, _ = load_datasets(
        cfg
    )

    eval_dataset = build_graph_analysis_cohort(
        val_dataset,
        test_dataset,
        cfg,
    )

    eval_dataset = make_split_safe(
        train_dataset,
        eval_dataset,
    )

    train_loader, eval_loader = build_loaders(
        train_dataset,
        eval_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # ============================================================
    # MODEL LOOP
    # ============================================================

    summaries = []
    granularity_tables = []

    for tag in cfg.architectures:
        print("\n" + "=" * 72)
        print(
            f"[{cfg.dataset.upper()}] "
            f"{tag}"
        )
        print("=" * 72)

        try:
            result = run_model(
                tag=tag,
                train_loader=train_loader,
                eval_loader=eval_loader,
                cfg=cfg,
            )

            summaries.append(
                result["summary"]
            )

            granularity_tables.append(
                result["granularity"]
            )

            save_model_outputs(
                tag,
                result,
                cfg,
            )

            # Save progress after every architecture
            pd.DataFrame(
                summaries
            ).to_csv(
                cfg.output_dir / "summary.csv",
                index=False,
            )

            pd.concat(
                granularity_tables,
                ignore_index=True,
            ).to_csv(
                cfg.output_dir
                / "semantic_granularity_all.csv",
                index=False,
            )

            print(
                f"[{tag}] DONE — "
                f"{result['summary']['minutes']} min"
            )

        except Exception:
            print(
                f"[{tag}] FAILED\n"
                f"{traceback.format_exc()}"
            )

        finally:
            cleanup()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print("\n" + "=" * 72)
    print(
        f"FINAL SUMMARY — "
        f"{cfg.dataset.upper()}"
    )
    print("=" * 72)

    if summaries:
        summary_df = pd.DataFrame(
            summaries
        )

        summary_df.to_csv(
            cfg.output_dir / "summary.csv",
            index=False,
        )

        print(
            summary_df.to_string(
                index=False
            )
        )

    if granularity_tables:
        granularity_df = pd.concat(
            granularity_tables,
            ignore_index=True,
        )

        granularity_df.to_csv(
            cfg.output_dir
            / "semantic_granularity_all.csv",
            index=False,
        )

        print("\n" + "=" * 72)
        print(
            "ANATOMY vs PHASE vs "
            "ANATOMY × PHASE"
        )
        print("=" * 72)

        columns = [
            "dataset",
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

        available = [
            column
            for column in columns
            if column in granularity_df.columns
        ]

        print(
            granularity_df[
                available
            ]
            .round(4)
            .to_string(index=False)
        )

    print("\nSaved to:")
    print(cfg.output_dir)


if __name__ == "__main__":
    main()