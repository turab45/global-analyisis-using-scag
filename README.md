# Semantic Coactivation Graph (SCAG)

SCAG extends a neuron coactivation graph with semantic information. The same analysis pipeline is applied to trained cardiac-MRI CNNs from **M&Ms-2** and **ACDC**.

The core representation is:

```text
activations A
    ├── neuron-concept association X ──> semantic similarity Q
    └── spatial pooling ───────────────> coactivation R

R <── alignment ──> Q
        |
        v
coactivation graph
        |
        +── Louvain communities
        +── semantic enrichment / homophily
        +── anatomy-area coupling
        └── causal community ablation
```


## Current multilayer experiments and checkpoints

Start with **`scag_resnet18_from_scratch.ipynb`** (the filename is historical; it also supports DenseNet-121 and DenseNet-161). All model, dataset, stage, query and intervention settings are in its top configuration cell. Restart the kernel and run from the top when switching models. Fixed-budget and variable-size traversal comparisons are separate experiments. Current notebook graphs use training patients; validation is the default evaluation split. The pooled graph-cohort description below applies to the original pipeline, not this notebook.

- Model architectures/checkpoint loading: `scag/models.py`.
- Paths, model seeds, dataset configuration: `scag/config.py`.
- Checkpoint index, exact paths, sizes, SHA-256 and download URLs: [`checkpoints/manifest.json`](checkpoints/manifest.json).
- Weights: [GitHub release `checkpoints-v1`](https://github.com/turab45/global-analyisis-using-scag/releases/tag/checkpoints-v1), kept outside Git history.
- Download helper: `scripts/download_checkpoints.py` (Python 3.11+, GitHub CLI; authenticate with `gh auth login` for private repository access).
- Research plans and methodological notes: `ideas/`.

The release contains **63 existing checkpoints, approximately 3.65 GiB**: ACDC has 10 architectures × 3 seeds; M&Ms-2 has 11 architectures × 3 seeds. Download only the models you need:

```bash
export CINEMA_RESULTS_ROOT="$PWD/cinema_results"
python scripts/download_checkpoints.py --dataset acdc --model densenet121 --seed 0 --list
python scripts/download_checkpoints.py --dataset acdc --model densenet121 --seed 0
# Download all checkpoints (approximately 3.65 GiB):
python scripts/download_checkpoints.py
```

The helper verifies SHA-256, skips verified files, and refuses to overwrite different existing checkpoints. It restores these paths automatically:

```text
$CINEMA_RESULTS_ROOT/
  architecture_bank/2d/
    acdc_sax_mid_2d/<model>/stacked/imagenet/seed_<n>/best_val_mcc.pt
    mnms2_lax_4c/<model>/stacked/imagenet/seed_<n>/best_val_mcc.pt
  processed/
    acdc/     # separately prepared dataset; not distributed here
    mnms2/    # separately prepared dataset; not distributed here
```

Set `CINEMA_RESULTS_ROOT` **before starting Jupyter**. The original workstation root is `/media/kislay/New Volume/Turab/results/cinema_reproduction`; it is not a required path on another machine. Trained checkpoints do not include MRI datasets or the preprocessing artifacts required by `scag/data.py`. Obtain and prepare those separately under the dataset's access terms. The original training pipeline is not recreated by downloading the weights.

Notebook outputs are cleared for publication. Running experiments recreates figures, patient-level evidence, graphs and CSV files under `scag_outputs/from_scratch/<dataset>_<model>/<timestamp>/`. These generated files, MRI patches, activation caches and notebook backups are excluded from Git. Graph correlations are associations; intervention results describe sensitivity under the chosen baseline, not clinical causality.

## Supported datasets

| Dataset | Network input | Pathology classes | Train | Val | Test | Graph cohort |
|---|---|---|---:|---:|---:|---:|
| M&Ms-2 | LAX-4C ED + ES | ARR, CIA, FALL, HCM, LV, NOR | 160 | 30 | 110 | 140 |
| ACDC | central SAX ED + ES | DCM, HCM, MINF, NOR, RV | 90 | 10 | 50 | 60 |

The graph cohort is **official validation + official test** for both datasets. Therefore, after SCAG analysis, the official test split is **not an untouched held-out test set**.

## Semantic concepts

Both datasets use the same cardiac-anatomy masks and the same phase-aware semantic vocabulary:

```text
0  background
1  RV_ED
2  MYO_ED
3  LV_ED
4  RV_ES
5  MYO_ES
6  LV_ES
```

The same labeled neurons are also evaluated under three semantic views:

- **anatomy:** RV, MYO, LV
- **phase:** ED, ES
- **anatomy × phase:** RV_ED, MYO_ED, LV_ED, RV_ES, MYO_ES, LV_ES

## Repository structure

```text
SCAG/
├── scag/
│   ├── __init__.py
│   ├── config.py          # dataset selection + SCAG configuration
│   ├── data.py            # M&Ms-2 and ACDC dataset adapters
│   ├── models.py          # shared 2-D ED/ES architecture + checkpoint loading
│   ├── activations.py     # feature extraction
│   ├── semantics.py       # neuron-concept X + semantic views
│   ├── matrices.py        # R, Q, R-Q alignment, edge reproducibility
│   ├── graph.py           # graph construction and semantic graph analyses
│   ├── ablation.py        # causal community ablation
│   └── pipeline.py        # complete per-model SCAG pipeline
│
├── demo_mnms2.ipynb       # step-by-step M&Ms-2 example
├── demo_acdc.ipynb        # step-by-step ACDC example
├── cmri-concept-profiles.ipynb # SAX/LAX image and class concept profiles
├── run_scag.py            # multi-architecture runner
├── threshold_analysis.py  # threshold sweep with printed lists + CSV
├── threshold_analysis.ipynb # all-model DataFrame + inline threshold plots
├── requirements.txt
└── README.md
```

## Installation

Create or activate your Python environment and install the requirements:

```bash
pip install -r requirements.txt
```

The code expects PyTorch/torchvision versions compatible with the checkpoints that were used to train the architecture banks.

### M&Ms-2 CineMA config

For M&Ms-2, the same CineMA package/check-out used during training must be importable because `scag/data.py` reads:

```text
cinema.classification.mnms2/config.yaml
```

to reproduce the original evaluation preprocessing configuration.

The ACDC SCAG adapter contains the required central-SAX selection and evaluation preprocessing extracted from the supplied ACDC training code/config.

## Data and checkpoint root

By default SCAG uses:

```text
/media/kislay/New Volume/Turab/results/cinema_reproduction
```

Override this with:

```bash
export CINEMA_RESULTS_ROOT=/path/to/cinema_reproduction
```

Expected layout:

```text
$CINEMA_RESULTS_ROOT/
├── processed/
│   ├── mnms2/
│   └── acdc/
│
└── architecture_bank/
    └── 2d/
        ├── mnms2_lax_4c/
        └── acdc_sax_mid_2d/
```

Each trained architecture is expected under:

```text
<model_root>/<backbone>/<formulation>/<initialization>/seed_0/best_val_mcc.pt
```

For example:

```text
architecture_bank/2d/acdc_sax_mid_2d/resnet18/stacked/imagenet/seed_0/best_val_mcc.pt
```

## Step-by-step demos

Use the notebooks when you want to inspect every SCAG stage separately.

For image-level interpretation, open `cmri-concept-profiles.ipynb` (renamed from `cmri-gradcma.ipynb`). It restores both SAX/LAX ResNet50 checkpoints, shows original ED/ES images, calibrates Network Dissection on training-split masks, and plots concept profiles as horizontal stacked bars using paired red, blue, and yellow colors. Side-by-side SAX/LAX figures show both percentage shares of named concept strength and raw signed contributions for the predicted or selected class. It uses the saved preprocessing configuration without the training checkout. Originals, activation arrays, concept associations, CSV profiles, and figures are saved under `outputs/concept_profiles/`; comparison figures are also exported as PNG/SVG under `comparison/<split>/`.

Within each `<view>/<split>/`, exports use `original_<CLASS>/`, `masks_<CLASS>/`, and `concept_<CLASS>/percentage/` or `raw/`, grouped by the image's ground-truth diagnosis. Original images and separate anatomy overlays share patient/phase filenames; concept figures place masked ED/ES images above one profile bar. CSV/NPZ files, including integer segmentation labels, live in `data/`.

### M&Ms-2

Open:

```text
demo_mnms2.ipynb
```

It starts with:

```python
cfg = SCAGConfig(dataset="mnms2")
```

### ACDC

Open:

```text
demo_acdc.ipynb
```

It starts with:

```python
cfg = SCAGConfig(dataset="acdc")
```

After dataset/model loading, both notebooks intentionally execute the same SCAG sequence:

```text
activations
    -> phase-aware masks
    -> X
    -> R
    -> Q
    -> R-Q alignment
    -> fixed-threshold graph
    -> density-matched graph
    -> semantic granularity
    -> Louvain stability
    -> community enrichment
    -> anatomy-area coupling
    -> causal ablation
```

The demo uses `resnet18` by default. Change:

```python
tag = "resnet18"
```

to inspect another available checkpoint.

## Run all architectures

### M&Ms-2

```bash
python run_scag.py --dataset mnms2
```

### ACDC

```bash
python run_scag.py --dataset acdc
```

Dataset selection changes only the data/model adapter. The SCAG analysis code is shared.

## Main configuration

The central configuration is `SCAGConfig`:

```python
from scag.config import SCAGConfig

cfg = SCAGConfig(dataset="acdc")
```

Shared default SCAG settings include:

```text
alpha                         0.005
eta                           0.015
tau                           0.35
graph density                 0.02
minimum labeled neurons       10
graph permutations            500
R-Q alignment permutations    1000
area-coupling permutations    1000
Louvain seeds                 10
random ablations              1000
```

These parameters are shared across ACDC and M&Ms-2 so that dataset comparisons use the same SCAG procedure.

## Core matrices

### X — neuron-concept association

`X` has shape:

```text
number of neurons × number of semantic concepts
```

It is derived from activation maps and segmentation masks using the SCAG/Network-Dissection-style IoU association procedure.

### R — functional coactivation

Activation maps are spatially pooled and neuron-neuron Spearman correlation is computed:

```text
R[i, j] = functional coactivation between neurons i and j
```

### Q — semantic similarity

Cosine similarity is computed between neuron concept profiles:

```text
Q[i, j] = semantic similarity between neurons i and j
```

The current implementation uses **cosine similarity only**.

### R-Q alignment

The upper triangles of `R` and `Q` are compared with Spearman correlation and a neuron-label permutation null distribution.

## Graph construction

The fixed-threshold graph uses:

```text
tau = 0.35
```

A second graph is constructed at a matched edge density:

```text
density = 2%
```

Louvain communities are then evaluated for:

- semantic purity
- adjusted mutual information (AMI)
- semantic assortativity
- soft semantic-profile similarity
- thresholded R-Q alignment
- partition stability across Louvain seeds
- concept enrichment

### Threshold sensitivity

Open `threshold_analysis.ipynb` to run all configured architectures for both datasets (11 M&Ms-2 models and 10 ACDC models). It reuses saved SCAG matrices when available; otherwise it extracts activations and calibrates concept labels from the trained checkpoints using the existing data/model helpers. Results are combined in `df`, with one row per dataset/model/threshold, and six threshold curves are displayed for each dataset. The notebook does not export CSVs, matrices, or figures; any model failures are reported in `failures_df`.

Sweep the coactivation correlation cutoff using an existing SCAG archive:

```bash
python threshold_analysis.py scag_outputs/acdc/resnet18_scag.npz
python threshold_analysis.py scag_outputs/mnms2/resnet18_scag.npz --thresholds 0.2 0.3 0.35 0.4 0.5
```

The default thresholds are `[0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`, including the current `tau = 0.35`. The script prints a Python list for each metric and saves one CSV row per threshold beside the input, e.g. `resnet18_scag_thresholds.csv`. Use `--output path/to/results.csv` to choose a destination. Saved archives use `R_eval`; model inference is not repeated.

You can also call it directly after computing `R_eval`, `X`, and `labeled` in either demo notebook:

```python
from threshold_analysis import analyze_thresholds

results = analyze_thresholds(R_eval, X, labeled, thresholds=[0.2, 0.3, 0.35, 0.4, 0.5])
thresholds = results["threshold"]
edge_counts = results["n_edges"]
modularities = results["modularity"]
semantic_amis = results["ami"]
```

Results also include retained node and isolate counts, density, mean degree, connected-component count, largest-component fraction, community count, and retained labeled-neuron count. Edges use the existing `R[i, j] >= tau` rule, exclude zero correlations, and retain their correlation weights. Isolated neurons are removed as in the main pipeline. Density and largest-component fraction use **all original neurons** as the denominator; mean degree uses retained neurons. Semantic AMI compares communities with the dominant concept of retained labeled neurons. Empty graphs have zero size/connectivity statistics and `NaN` modularity/AMI; AMI is also `NaN` with fewer than two labeled neurons.

Louvain uses seed 42 for reproducibility. This is a descriptive sensitivity sweep: it does not run permutation tests, repeated-seed stability checks, ablation, or plotting.

## Causal ablation

For each eligible Louvain community, SCAG zeros its feature channels at the selected network layer and measures the change in model predictions.

The principal outputs include:

- KL divergence after community ablation
- size-matched random-neuron KL baseline
- prediction flip rate
- anatomy-burden conditional specificity when available

The default `n_ablation_samples=1000` is intended for final experiments and can be computationally expensive.

## Outputs

Results are separated automatically by dataset:

```text
scag_outputs/
├── mnms2/
└── acdc/
```

For each architecture:

```text
<tag>_scag.npz
<tag>_semantic_granularity.csv
<tag>_enrichment.csv
<tag>_ablation.csv
```

The `.npz` file stores the central SCAG representations:

```text
X
labeled
R_train
R_eval
Q
X_anatomy
X_phase
X_anatomy_phase
```

Dataset-level summaries are saved as:

```text
summary.csv
semantic_granularity_all.csv
```

## Recommended workflow

For a new environment or code change:

1. Run `demo_mnms2.ipynb` with `resnet18`.
2. Run `demo_acdc.ipynb` with `resnet18`.
3. Confirm dataset sizes, feature layer, `X`, `R`, `Q`, graph statistics, and ablation output.
4. Run the full architecture bank with `run_scag.py`.
5. Compare ACDC and M&Ms-2 results using the shared `dataset` and `tag` columns.

## SemanticLens-inspired analyses across layers

`run_semantic_analysis.py` adds concept-centred analysis without changing the original runner. It uses all spatial feature layers listed in each checkpoint by default. Explicit module names allow earlier layers too; each layer is analysed independently. These are within-layer coactivation graphs, not cross-layer attribution or causal-flow graphs.

```bash
# All designated layers of ResNet18, including concept-guided ablation
python run_semantic_analysis.py --dataset acdc --models resnet18 --layers all

# Specific early and late layers, descriptive analyses only
python run_semantic_analysis.py --dataset mnms2 --models resnet18 --layers encoder.layer1 encoder.layer4 --skip-ablation

# All architectures, anatomy-by-phase vocabulary
python run_semantic_analysis.py --dataset acdc --semantic-level anatomy_phase --skip-ablation
```

The default vocabulary is anatomy (RV, MYO, LV); `--semantic-level phase` and `anatomy_phase` reuse the existing semantic views. Anatomy/phase scores are sums of the existing phase-specific IoUs, not newly calibrated union-mask IoUs. Background is excluded because the existing dissection code assigns it zero. Neurons with nonzero profiles are included even below the existing hard-label threshold; query and enrichment membership use `--gamma` (default 0.015).

Implemented analyses:

- **Concept relationships:** `W = S.T @ A @ S`, with positive `R >= tau` edges, zero diagonal, raw weights, weights per total adjacency mass, and the semantic-only `S.T @ S` baseline. Within-concept entries count both orientations of undirected edges. Train/evaluation graphs use the same training-derived `S`.
- **Null validation:** complete neuron-profile shuffling with the graph fixed, plus degree-quartile-stratified profile shuffling. Tied degrees stay together. Profile shuffling is algebraically equivalent to randomly relabelling graph nodes while holding `S` fixed; it is not independent-edge rewiring. Exports include null mean, excess weight, z, one-sided enrichment p and BH-adjusted q. Degenerate nulls have undefined z. BH correction is within each layer/split/null across unique concept pairs including the diagonal, not across the entire architecture bank. Null tests assess organisation conditional on these patients, not uncertainty under patient resampling.
- **Traceable concept queries:** all neurons above gamma, all edges between those seeds, and up to `--neighbors 3` strongest outside neighbours per seed. Node CSVs retain every concept score; edge CSVs retain neuron IDs and correlation weights. `supporting_examples.csv` links each neuron to its five highest mean-activation training patients, their split, data directory and sample index. These illustrate node responses; they are not per-image causal evidence for an edge.
- **Community organisation:** multi-label concept enrichment with BH correction and semantic variance explained (SVE) against size-preserving partition permutations. The population is nonisolated neurons with nonzero profiles; isolated neurons do not inflate SVE through singleton communities.
- **Semantic/functional pairs:** signed correlation versus cosine similarity, four categories using prespecified `--tau 0.35` and `--similarity 0.8`, profile-permutation count tests, and all neuron pairs in compressed CSV. Zero-profile neurons have undefined semantics and are excluded. High coactivation alone does not establish functional redundancy.
- **Concept-guided intervention:** training-derived joint score `S[:, c] * (A_train @ S[:, c])` versus semantic-only, weighted-degree-only and random selection. All strategies use the same layer, evaluation patients, channel pool and channel count. Default budgets are 1%, 5%, 10%, rounded up; random selection repeats 20 times. Rankings break ties by neuron ID and report how many selected neurons have positive scores. Exports include individual selections, aggregate KL/flip rates and patient-level effects. Repeat SD describes random-selection variability, not a patient confidence interval.

Results go to `scag_outputs/semantic_analysis/<dataset>/<semantic-level>/<model>/<layer>/`: representations, provenance, relation/null tables, enrichment/SVE, query nodes/edges, pair categories, ablations and PNG figures. Parent-level `relations_all.csv` and comparison heatmaps align the same concept pairs across models, layers and train/evaluation splits. Raw weights depend on layer width and graph density; inspect null-adjusted and edge-mass-normalised values alongside them. Layer names from different architectures do not imply equivalent depth. The evidence figure displays only the 12 strongest query edges per concept; CSVs retain the complete query. A rerun with the same output path replaces that run's files; choose `--output` for separate configurations.

The runner verifies disjoint training/evaluation patient IDs. Rankings are never fitted to evaluation activations, but evaluation remains the existing official validation-plus-test cohort, not a newly untouched test set. Prediction sensitivity does not by itself establish concept-specific causality or anatomical relations. Activation-matched control selection, patient bootstrap uncertainty, broader artefact auditing and SemanticLens-style example-embedding clarity/polysemanticity are not implemented.

Use `--permutations` and `--random-repeats` for exploratory versus final experiments. Early layers require more activation memory; layers are processed sequentially and training maps are released before evaluation extraction. `--skip-ablation` avoids repeated inference. Failures are recorded in `failures.json`, and partial runs exit nonzero.

```bash
python -m pytest tests/test_semantic_analysis.py tests/test_threshold_analysis.py -q
```
