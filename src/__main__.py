"""YAML-configured global SCAG analysis."""

import hydra
from omegaconf import DictConfig

from .pipeline import run_analysis


@hydra.main(version_base=None, config_path="configs", config_name="acdc")
def main(config: DictConfig):
    run_analysis(config)


if __name__ == "__main__":
    main()
