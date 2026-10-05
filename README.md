# SCAG

Refactored Semantic Coactivation Graph code for M&Ms-2 LAX-4C.

## Structure

```text
SCAG/
├── scag/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── activations.py
│   ├── semantics.py
│   ├── matrices.py
│   ├── graph.py
│   ├── ablation.py
│   └── pipeline.py
├── demo.ipynb
├── run_scag.py
└── requirements.txt
```

## Run

Use `demo.ipynb` for the step-by-step single-model walkthrough.

Use:

```bash
python run_scag.py
```

for the full multi-architecture experiment.

## Paths

By default the code uses:

```text
/media/kislay/New Volume/Turab/results/cinema_reproduction
```

Override it with:

```bash
export CINEMA_RESULTS_ROOT=/path/to/cinema_reproduction
```

The same CineMA installation/check-out used by the original training code must be importable because `scag/data.py` reads `cinema.classification.mnms2/config.yaml` to reproduce the exact evaluation preprocessing configuration.
