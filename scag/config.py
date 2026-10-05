# scag/config.py

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class SCAGConfig:
    dataset: str = "mnms2"

    # SCAG hyperparameters
    alpha: float = 0.005
    eta: float = 0.015
    tau: float = 0.35
    graph_density: float = 0.02
    min_labeled_neurons: int = 10

    # Permutations / stability
    n_battery_permutations: int = 500
    n_mantel_permutations: int = 1000
    n_cooccurrence_permutations: int = 1000
    n_louvain_seeds: int = 10
    n_ablation_samples: int = 1000

    # Experiment
    do_ablation: bool = True
    batch_size: int = 16
    num_workers: int = 0

    # Model checkpoint settings
    model_seed: int = 0
    formulation: str = "stacked"
    initialization: str = "imagenet"

    def __post_init__(self):
        self.dataset = self.dataset.lower()

        if self.dataset not in {"mnms2", "acdc"}:
            raise ValueError(
                f"Unknown dataset {self.dataset!r}. "
                "Choose 'mnms2' or 'acdc'."
            )

    @property
    def results_root(self):
        return Path(
            os.environ.get(
                "CINEMA_RESULTS_ROOT",
                "/media/kislay/New Volume/Turab/results/cinema_reproduction",
            )
        ).expanduser().resolve()

    @property
    def data_root(self):
        if self.dataset == "mnms2":
            return self.results_root / "processed" / "mnms2"

        return self.results_root / "processed" / "acdc"

    @property
    def model_root(self):
        if self.dataset == "mnms2":
            return (
                self.results_root
                / "architecture_bank"
                / "2d"
                / "mnms2_lax_4c"
            )

        return (
            self.results_root
            / "architecture_bank"
            / "2d"
            / "acdc_sax_mid_2d"
        )

    @property
    def task_name(self):
        if self.dataset == "mnms2":
            return "mnms2_lax_4c"

        return "acdc_sax_mid_2d"

    @property
    def model_family(self):
        if self.dataset == "mnms2":
            return "architecture_bank"

        return "acdc_2d_architecture_bank"

    @property
    def classes(self):
        if self.dataset == "mnms2":
            return (
                "ARR",
                "CIA",
                "FALL",
                "HCM",
                "LV",
                "NOR",
            )

        return (
            "DCM",
            "HCM",
            "MINF",
            "NOR",
            "RV",
        )

    @property
    def view(self):
        if self.dataset == "mnms2":
            return "lax_4c"

        return "sax"

    @property
    def expected_sizes(self):
        if self.dataset == "mnms2":
            return {
                "train": 160,
                "val": 30,
                "test": 110,
                "eval": 140,
            }

        return {
            "train": 90,
            "val": 10,
            "test": 50,
            "eval": 60,
        }

    @property
    def architectures(self):
        if self.dataset == "mnms2":
            return (
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

        return (
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
        )

    # Shared semantic vocabulary
    @property
    def concept_names(self):
        return (
            "background",
            "RV_ED",
            "MYO_ED",
            "LV_ED",
            "RV_ES",
            "MYO_ES",
            "LV_ES",
        )

    @property
    def concept_name_map(self):
        return dict(enumerate(self.concept_names))

    @property
    def concept_category_map(self):
        return {
            0: "background",
            1: "anatomy_phase",
            2: "anatomy_phase",
            3: "anatomy_phase",
            4: "anatomy_phase",
            5: "anatomy_phase",
            6: "anatomy_phase",
        }

    @property
    def num_concepts(self):
        return len(self.concept_names)

    @property
    def validation_source(self):
        return "official_val_plus_test"

    @property
    def output_dir(self):
        return Path("scag_outputs") / self.dataset

    @property
    def device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")