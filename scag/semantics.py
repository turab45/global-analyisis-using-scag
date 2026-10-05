# scag/semantics.py

import numpy as np
import torch


def compute_neuron_concept_matrix(activations, segments, num_concepts, alpha, eta):
    N, F, _, _ = activations.shape
    _, _, H, W = segments.shape

    seg = segments.numpy().astype(np.int64, copy=False)
    counts = np.bincount(seg.reshape(-1), minlength=num_concepts)[:num_concepts].astype(np.float64)
    counts[0] = 0.0

    thresholds = np.percentile(
        activations.numpy().reshape(N, F, -1),
        (1 - alpha) * 100,
        axis=(0, 2),
    )

    X = np.zeros((F, num_concepts), dtype=np.float32)

    for k in range(F):
        up = torch.nn.functional.interpolate(
            activations[:, k:k + 1], size=(H, W),
            mode="bilinear", align_corners=False
        ).squeeze(1)

        mask = up.numpy() >= thresholds[k]
        ii, yy, xx = np.nonzero(mask)

        if ii.size == 0:
            continue

        labels = seg[ii, :, yy, xx].reshape(-1)
        intersection = np.bincount(labels, minlength=num_concepts)[:num_concepts].astype(np.float64)
        intersection[0] = 0.0

        union = float(mask.sum()) + counts - intersection
        X[k] = np.divide(
            intersection, union,
            out=np.zeros(num_concepts, dtype=np.float64),
            where=union > 0,
        )

        if (k + 1) % 128 == 0 or k + 1 == F:
            print(f"  dissect {k + 1}/{F}")

    labeled = X.max(axis=1) >= eta
    return X, labeled


def make_semantic_views(X):
    if X.ndim != 2:
        raise ValueError("X must be a 2-D neuron × concept matrix.")
    if X.shape[1] != 7:
        raise ValueError(f"Expected 7 columns, got {X.shape[1]}.")

    F = X.shape[0]

    # Anatomy: RV_ED+RV_ES, MYO_ED+MYO_ES, LV_ED+LV_ES
    X_anatomy = np.zeros((F, 4), dtype=np.float32)
    X_anatomy[:, 0] = X[:, 0]
    X_anatomy[:, 1] = X[:, 1] + X[:, 4]
    X_anatomy[:, 2] = X[:, 2] + X[:, 5]
    X_anatomy[:, 3] = X[:, 3] + X[:, 6]

    # Phase: all ED anatomy, all ES anatomy
    X_phase = np.zeros((F, 3), dtype=np.float32)
    X_phase[:, 0] = X[:, 0]
    X_phase[:, 1] = X[:, 1] + X[:, 2] + X[:, 3]
    X_phase[:, 2] = X[:, 4] + X[:, 5] + X[:, 6]

    return {
        "anatomy": {
            "X": X_anatomy,
            "names": {0: "background", 1: "RV", 2: "MYO", 3: "LV"},
            "categories": {0: "background", 1: "anatomy", 2: "anatomy", 3: "anatomy"},
        },
        "phase": {
            "X": X_phase,
            "names": {0: "background", 1: "ED", 2: "ES"},
            "categories": {0: "background", 1: "phase", 2: "phase"},
        },
        "anatomy_phase": {
            "X": X.astype(np.float32, copy=True),
            "names": {
                0: "background", 1: "RV_ED", 2: "MYO_ED", 3: "LV_ED",
                4: "RV_ES", 5: "MYO_ES", 6: "LV_ES",
            },
            "categories": {
                0: "background",
                1: "anatomy_phase", 2: "anatomy_phase", 3: "anatomy_phase",
                4: "anatomy_phase", 5: "anatomy_phase", 6: "anatomy_phase",
            },
        },
    }