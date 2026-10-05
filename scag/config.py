"""
Configuration for Semantic Coactivation Graph (SCAG) experiments.

This file contains only experiment settings and constants.
No dataset construction, model loading, or analysis should happen here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import torch


# ================================================================
# DEFAULT ARCHITECTURES
# ================================================================

DEFAULT_ARCHITECTURES = (
    "mobilenet_v3_small",
    "efficientnet_b0",
    "mobilenet_v3_large",
    "densenet121",
    "efficientnet_b3",
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet161",
    "convnext_tiny",
    "convnext_small",
)


# ================================================================
# DEFAULT FEATURE LAYERS
#
# These are fallback values.
# The actual layer may later be recovered from the checkpoint
# metadata in models.py, as your current code already does.
# ================================================================

DEFAULT_LAYER_NAMES = {
    "resnet18": "encoder.layer4",
    "densenet121": "encoder.features.denseblock4",
    "efficientnet_b0": "encoder.features.7",
    "convnext_tiny": "encoder.features.7",
}


# ================================================================
# PHASE-AWARE CONCEPTS
#
# 0 = background
#
# ED:
#   1 = RV_ED
#   2 = MYO_ED
#   3 = LV_ED
#
# ES:
#   4 = RV_ES
#   5 = MYO_ES
#   6 = LV_ES
# ================================================================

CONCEPT_NAMES = (
    "background",
    "RV_ED",
    "MYO_ED",
    "LV_ED",
    "RV_ES",
    "MYO_ES",
    "LV_ES",
)


# ================================================================
# CONFIGURATION
# ================================================================

@dataclass
class SCAGConfig:
    """
    Main configuration for SCAG experiments.

    The defaults reproduce the current M&Ms-2 LAX-4C V4 setup.
    """

    # ------------------------------------------------------------
    # Neuron-concept association
    # ------------------------------------------------------------

    # Fraction of strongest activations retained when constructing
    # neuron activation masks.
    alpha: float = 0.005

    # Minimum neuron-concept association score required for a neuron
    # to be considered semantically labelled.
    eta: float = 0.015

    # Minimum number of labelled neurons required to continue.
    min_labeled_neurons: int = 10

    # ------------------------------------------------------------
    # Coactivation graph
    # ------------------------------------------------------------

    # Fixed Spearman correlation threshold.
    tau: float = 0.35

    # Density used for the density-matched graph.
    graph_density: float = 0.02

    # ------------------------------------------------------------
    # Permutation tests
    # ------------------------------------------------------------

    # Graph-semantic battery permutations.
    n_battery_permutations: int = 500

    # P-Q / Mantel-style alignment permutations.
    n_mantel_permutations: int = 1000

    # Co-occurrence / area-coupling permutations.
    n_cooccurrence_permutations: int = 1000

    # ------------------------------------------------------------
    # Community analysis
    # ------------------------------------------------------------

    # Number of Louvain seeds used for stability analysis.
    n_louvain_seeds: int = 10

    # ------------------------------------------------------------
    # Causal ablation
    # ------------------------------------------------------------

    n_ablation_samples: int = 1000

    do_ablation: bool = True

    # Currently disabled in V4.
    do_depth_analysis: bool = False

    # ------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------

    batch_size: int = 16

    num_workers: int = 0

    # ------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------

    model_seed: int = 0

    formulation: str = "stacked"

    initialization: str = "imagenet"

    architectures: Tuple[str, ...] = DEFAULT_ARCHITECTURES

    layer_names: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_LAYER_NAMES)
    )

    # ------------------------------------------------------------
    # Semantic concepts
    # ------------------------------------------------------------

    concept_names: Tuple[str, ...] = CONCEPT_NAMES

    # ------------------------------------------------------------
    # Current M&Ms-2 graph-analysis cohort
    #
    # train           = 160
    # validation      = 30
    # test            = 110
    # graph evaluation = validation + test = 140
    #
    # NOTE:
    # Because val + test are combined for graph analysis,
    # the official test set is not untouched in this experiment.
    # ------------------------------------------------------------

    validation_source: str = "official_val_plus_test"

    expected_train_size: int = 160

    expected_graph_eval_size: int = 140

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    random_seed: int = 42

    mantel_seed: int = 7

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    output_dir: Path = Path("scag_cache_mnms2_v4")

    # ------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------

    @property
    def num_concepts(self) -> int:
        """Number of concepts including background."""
        return len(self.concept_names)

    @property
    def concept_name_map(self) -> Dict[int, str]:
        """Map concept ID -> human-readable concept name."""
        return dict(enumerate(self.concept_names))

    @property
    def concept_category_map(self) -> Dict[int, str]:
        """
        Map concept ID -> semantic category.

        Current representation is anatomy × phase.
        """
        return {
            i: "background" if i == 0 else "anatomy_phase"
            for i in range(self.num_concepts)
        }

    @property
    def device(self) -> torch.device:
        """
        Automatically select the best available device.

        Priority:
            1. Apple MPS
            2. CUDA
            3. CPU
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")