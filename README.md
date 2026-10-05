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
├── run_scag.py            # multi-architecture runner
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
