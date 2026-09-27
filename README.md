# Global SCAG analysis

An analysis-only extraction of the ACDC and M&Ms2 SCAG notebooks. It loads frozen
2-D classifiers or existing SCAG matrices, measures neuron–concept associations,
and evaluates semantic organization in their coactivation graphs.

There are no training loops, optimizers, augmentation, or imports from
`training_code`. Local neuron interventions and causal ablations remain outside
this global-analysis package.

## Structure

```text
scag_global_analysis/
├── pyproject.toml
├── .gitignore
└── scag/
    ├── __main__.py       # Hydra command-line entry point
    ├── data.py           # Patient cohorts, splits, paired image/mask loading
    ├── preprocessing.py  # Deterministic normalization and padding
    ├── models.py         # Checkpoint-compatible inference models
    ├── activations.py    # Feature hooks and patient-level pooling
    ├── concepts.py       # Phase-aware masks and network dissection
    ├── coactivation.py   # Correlation matrices and edge selection
    ├── alignment.py      # Semantic cosine similarity and R–Q alignment
    ├── graph.py          # NetworkX communities, enrichment, stability
    ├── evaluation.py     # Global graph/semantic metrics and null models
    ├── visualization.py  # Matrix and graph figure exports
    ├── pipeline.py       # Complete analysis workflow
    ├── utils.py
    └── configs/
        ├── default.yaml
        ├── acdc.yaml
        ├── mnms2.yaml
        └── mnms2_sax.yaml
```

This follows the proposed module layout, with preprocessing separated explicitly.
YAML files are bundled inside the package so installed commands can find them.
Data, masks, checkpoints, and outputs are referenced by path; empty placeholder
folders and a test directory are not included.

## Run

From this directory, using the existing CineMA environment:

```bash
python -m pip install -e . --no-deps
python -m scag --cfg job
```

For a new environment, install the dependencies in `pyproject.toml` and a
PyTorch/torchvision build suitable for the machine. The `scag-global` command is
also available after installation.

Analyze an existing notebook cache without loading a model or patient data:

```bash
python -m scag \
  input.cache=../scag_cache_v4/resnet18_semantics.npz \
  output.dir=outputs/acdc/resnet18
```

Run from a checkpoint and CineMA-preprocessed NIfTI data:

```bash
python -m scag --config-name acdc \
  data.processed_dir=/path/to/processed/acdc \
  model.checkpoint=/path/to/best_val_mcc.pt \
  output.dir=outputs/acdc/resnet18
```

Use `--config-name mnms2` for LAX-4C or `--config-name mnms2_sax` for central SAX.
Set `model.layer=encoder.layer4` to select a layer explicitly; otherwise the last
feature-layer candidate in the checkpoint is used. Checkpoint class order, task,
and dimensionality must match the selected dataset configuration.

Images and masks use the existing layout:

```text
<processed-dir>/<split>/<pid>/<pid>_sax_ed.nii.gz
<processed-dir>/<split>/<pid>/<pid>_sax_ed_gt.nii.gz
```

ES files use `_es`; LAX-4C uses `_lax_4c` instead of `_sax`. Masks contain
0=background, 1=RV, 2=MYO, 3=LV. A separate `data.masks_root` may mirror this layout.
SAX uses the central acquired slice; LAX-4C uses its single slice. ED/ES images
are min/max scaled together and end-padded to at least 192×192, matching the
existing evaluation preprocessing. Larger images retain their geometry.

## Core functions

```python
from scag import compute_correlation, compute_semantic_similarity, compute_alignment

R = compute_correlation(pooled_activations)   # patients × neurons → neurons × neurons
Q = compute_semantic_similarity(profiles)    # neurons × concepts → neurons × neurons
result = compute_alignment(R, Q, n_permutations=1000, seed=42)
```

`compute_correaltion` is also available as an alias for the spelling in the
original request. The three functions can also be imported from `scag.alignment`.

- **Correlation R:** Spearman correlation of spatially averaged neuron responses
  across patients. Pearson is optional. Self-edges and constant-unit correlations
  are zero.
- **Network dissection:** per-neuron activation thresholds use the reference
  cohort's 99.5th percentile (`alpha=0.005`). Bilinearly upsampled salient maps are
  compared with six anatomy×phase masks by global IoU. A neuron is labeled when
  its best IoU reaches `eta=0.015`. Constant units receive zero profiles.
- **Semantic similarity Q:** cosine similarity between rows of the neuron–concept
  association matrix. Zero profiles have zero similarity. Centered and IDF-weighted
  variants are included as optional notebook sensitivity analyses.
- **Alignment:** Spearman correlation between matching off-diagonal upper
  triangles of R and Q. The default clips negative R values, matching the notebooks.
  Set `alignment.positive_only=false` for signed correlations. Permutation tests
  relabel Q's rows and columns together and use a one-sided, plus-one-corrected
  p-value. R and Q must have identical neuron order.

The joint anatomy×phase labeled-neuron set stays fixed when profiles are
marginalized to anatomy-only and phase-only views. Full-matrix alignment includes
all labeled neurons; thresholded-graph alignment uses labeled graph nodes.

## Graph analysis

NetworkX builds positive weighted graphs at `graph.threshold` and at the requested
`graph.density`. Density selection retains the requested number of pairs, up to
available positive edges; ties are resolved by neuron index. This avoids the
notebooks' quantile-tie ambiguity. `n_graph` counts nodes after optional isolate
removal, while total neurons and original isolates are reported separately.

Outputs include Louvain communities, weighted modularity, component counts,
community purity, adjusted mutual information, unweighted label assortativity,
and within-minus-between-community semantic similarity. Label-permutation nulls
preserve concept frequencies; soft-metric nulls preserve community sizes.
Community enrichment uses hypergeometric tests with BH correction over all
reported community/concept tests, including zero-count combinations.
Partition stability reports pairwise AMI across configured Louvain seeds.

The default reference cohort is the original training split, used only to fit
activation thresholds and concept profiles. The graph cohort is validation plus
test, matching the existing notebooks; the test split is therefore used for
analysis. Patient-ID overlap is rejected. Legacy caches retain their supplied
labeled mask and matrices; patient separation cannot be reverified from them.

Shared ED/ES encoders contribute the mean of their two feature maps per patient.
Undefined metrics (constant matrices, empty graphs, insufficient pairs) are
reported as missing values, with a status where applicable.

## Outputs

Each run writes the resolved YAML, provenance, profiles and correlation matrices,
`alignment.csv`, `graph_metrics.csv`, cross-cohort edge reproducibility, and a JSON
summary. Each semantic view also contains Q matrices with their neuron indices,
GraphML graphs, community enrichment/stability tables, and optional PNG figures.

Existing output directories are protected unless `output.overwrite=true` is set.
`output.figures=false` disables plots. Permutation counts and community seeds can
be reduced for a quick check. Original notebooks, checkpoints, and cached results
are left in place.
