# scag/semantics.py

import numpy as np
import torch


def concept_masks(segments, level):
    """Masks on the shared ED/ES image grid; anatomy is the union across phases."""
    if segments.ndim != 4 or segments.shape[1] != 2 or not torch.all((segments >= 0) & (segments <= 3)):
        raise ValueError("Require N x 2 x H x W raw ED/ES labels in {0,1,2,3}.")
    if level == "anatomy":
        return torch.stack([(segments == c).any(dim=1) for c in (1, 2, 3)], dim=1), ["RV", "MYO", "LV"]
    if level == "phase":
        return segments > 0, ["ED_foreground", "ES_foreground"]
    if level == "anatomy_phase":
        return torch.stack([segments[:, phase] == c for phase in (0, 1) for c in (1, 2, 3)], dim=1), [f"{name}_{phase}" for phase in ("ED", "ES") for name in ("RV", "MYO", "LV")]
    raise ValueError(f"Unknown semantic level: {level}")


def concept_profiles(activations, masks, alpha=0.005, thresholds=None, device="cpu", batch_size=8, channel_batch=16):
    """Exact aggregate and per-patient IoU using training-calibrated thresholds."""
    if activations.ndim != 4 or masks.ndim != 4 or len(activations) != len(masks) or not len(masks):
        raise ValueError("Activations and concept masks must have matching nonempty sample axes.")
    if not 0 < alpha < 1 or not torch.isfinite(activations).all() or min(batch_size, channel_batch) < 1:
        raise ValueError("Require finite activations, alpha in (0,1), and positive batch sizes.")
    n, channels = activations.shape[:2]
    if thresholds is None:
        thresholds = np.percentile(activations.numpy().reshape(n, channels, -1), 100 * (1 - alpha), axis=(0, 2))
    thresholds = np.asarray(thresholds)
    if thresholds.shape != (channels,) or not np.isfinite(thresholds).all():
        raise ValueError("One finite training threshold per channel is required.")
    intersection = np.zeros((n, channels, masks.shape[1]), dtype=np.float64)
    union = np.zeros_like(intersection)
    cutoffs = torch.as_tensor(thresholds, dtype=activations.dtype, device=device)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            targets = masks[start:end].to(device=device, dtype=torch.float32).flatten(2)
            target_area = targets.sum(dim=2)
            for k in range(0, channels, channel_batch):
                stop = min(channels, k + channel_batch)
                up = torch.nn.functional.interpolate(activations[start:end, k:stop].to(device), size=masks.shape[-2:], mode="bilinear", align_corners=False)
                # Strict > avoids assigning concepts to dead or constant channels.
                active = (up > cutoffs[k:stop][None, :, None, None]).float().flatten(2)
                overlap = torch.bmm(active, targets.transpose(1, 2))
                area = active.sum(dim=2)[:, :, None] + target_area[:, None, :] - overlap
                intersection[start:end, k:stop] = overlap.cpu().numpy()
                union[start:end, k:stop] = area.cpu().numpy()
    summed_union = union.sum(axis=0)
    S = np.divide(intersection.sum(axis=0), summed_union, out=np.zeros_like(summed_union), where=summed_union > 0)
    per_patient = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return {"S": S, "per_patient_iou": per_patient, "intersection": intersection, "union": union, "thresholds": thresholds}


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
