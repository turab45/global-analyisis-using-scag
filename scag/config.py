import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class SCAGConfig:
    results_root: Path = field(default_factory=lambda: Path(
        os.environ.get("CINEMA_RESULTS_ROOT", "/media/kislay/New Volume/Turab/results/cinema_reproduction")
    ).expanduser().resolve())

    task: str = "mnms2_lax_4c"
    view: str = "lax_4c"
    classes: tuple = ("ARR", "CIA", "FALL", "HCM", "LV", "NOR")
    expected_train: int = 160
    expected_val: int = 30
    expected_test: int = 110

    alpha: float = 0.005
    eta: float = 0.015
    min_labeled_neurons: int = 10
    tau: float = 0.35
    graph_density: float = 0.02

    n_battery_permutations: int = 500
    n_mantel_permutations: int = 1000
    n_cooccurrence_permutations: int = 1000
    n_louvain_seeds: int = 10
    n_ablation_samples: int = 1000
    do_ablation: bool = True

    batch_size: int = 16
    num_workers: int = 0
    model_seed: int = 0
    formulation: str = "stacked"
    initialization: str = "imagenet"
    validation_source: str = "official_val_plus_test"

    architectures: tuple = (
        "mobilenet_v3_small", "efficientnet_b0", "mobilenet_v3_large",
        "densenet121", "efficientnet_b3", "resnet18", "resnet34",
        "resnet50", "densenet161", "convnext_tiny", "convnext_small",
    )

    concept_names: tuple = (
        "background", "RV_ED", "MYO_ED", "LV_ED", "RV_ES", "MYO_ES", "LV_ES"
    )
    output_dir: Path = field(default_factory=lambda: Path("scag_cache_mnms2_v4"))

    @property
    def data_root(self):
        return self.results_root / "processed" / "mnms2"

    @property
    def model_root(self):
        return self.results_root / "architecture_bank" / "2d" / self.task

    @property
    def num_concepts(self):
        return len(self.concept_names)

    @property
    def concept_name_map(self):
        return dict(enumerate(self.concept_names))

    @property
    def concept_category_map(self):
        return {i: ("background" if i == 0 else "anatomy_phase") for i in range(self.num_concepts)}

    @property
    def device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")