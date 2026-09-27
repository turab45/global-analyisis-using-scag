"""Phase-aware anatomy concepts and network dissection."""

import numpy as np
import torch
import torch.nn.functional as F

from .utils import finite_matrix

CONCEPT_NAMES = ("background", "RV_ED", "MYO_ED", "LV_ED", "RV_ES", "MYO_ES", "LV_ES")


def encode_concepts(masks):
    """Encode paired anatomy masks as six anatomy×phase concepts."""
    result = torch.as_tensor(masks)
    if result.ndim != 3 or result.shape[0] != 2:
        raise ValueError("Masks must have shape (2, H, W)")
    if not torch.all((result >= 0) & (result <= 3) & (result == result.round())):
        raise ValueError("Masks must have anatomy IDs 0..3")
    result = result.long().clone()
    result[1] = torch.where(result[1] > 0, result[1] + 3, result[1])
    return result


def network_dissection(activations, masks, *, alpha=0.005, eta=0.015, channel_batch_size=32):
    """Fit unit thresholds and global IoU profiles on the reference cohort only.

    Thresholds use each neuron's (1-alpha) spatial activation quantile. Salient
    maps are bilinearly aligned with masks. Constant units are uninformative and
    receive zero profiles. The background column is retained as zero for caches.
    """
    if not 0 < alpha < 1 or not 0 <= eta <= 1 or channel_batch_size < 1:
        raise ValueError("Invalid dissection alpha, eta, or channel batch size")
    if not len(activations) or len(activations) != len(masks):
        raise ValueError("Provide matching, nonempty activation and mask sequences")
    maps = [torch.as_tensor(a).float().cpu() for a in activations]
    channels = maps[0].shape[0]
    if any(a.ndim != 3 or a.shape[0] != channels or not torch.isfinite(a).all() for a in maps):
        raise ValueError("Activation maps must be finite (C, H, W) tensors with the same channels")
    flat = torch.cat([a.flatten(1) for a in maps], dim=1).numpy()
    thresholds = np.quantile(flat, 1 - alpha, axis=1)
    variable = np.ptp(flat, axis=1) > 0
    intersections = np.zeros((channels, 7), dtype=np.float64)
    active_counts, concept_counts = np.zeros(channels), np.zeros(7)
    for feature, anatomy in zip(maps, masks):
        encoded = encode_concepts(anatomy)
        height, width = encoded.shape[-2:]
        targets = [(encoded == c).any(dim=0) for c in range(1, 7)]
        concept_counts[1:] += [target.sum().item() for target in targets]
        for start in range(0, channels, channel_batch_size):
            stop = min(start + channel_batch_size, channels)
            upsampled = F.interpolate(
                feature[None, start:stop], size=(height, width), mode="bilinear", align_corners=False
            )[0]
            active = upsampled >= torch.from_numpy(thresholds[start:stop])[:, None, None]
            active &= torch.from_numpy(variable[start:stop])[:, None, None]
            active_counts[start:stop] += active.sum(dim=(1, 2)).numpy()
            for concept, target in enumerate(targets, 1):
                intersections[start:stop, concept] += (active & target).sum(dim=(1, 2)).numpy()
    union = active_counts[:, None] + concept_counts[None, :] - intersections
    profiles = np.divide(intersections, union, out=np.zeros_like(intersections), where=union > 0)
    labeled = variable & (profiles[:, 1:].max(axis=1) >= eta)
    return profiles.astype(np.float32), labeled, thresholds


def semantic_views(profiles):
    """Marginalize the same joint profiles without changing the labeled neurons."""
    profiles = finite_matrix(profiles, "profiles")
    if profiles.shape[1] != 7 or (profiles < 0).any():
        raise ValueError("Expected seven nonnegative columns: background plus six phase-aware concepts")
    anatomy = np.column_stack((profiles[:, 0], profiles[:, 1:4] + profiles[:, 4:7]))
    phase = np.column_stack((profiles[:, 0], profiles[:, 1:4].sum(axis=1), profiles[:, 4:7].sum(axis=1)))
    return {
        "anatomy_phase": (profiles, CONCEPT_NAMES),
        "anatomy": (anatomy, ("background", "RV", "MYO", "LV")),
        "phase": (phase, ("background", "ED", "ES")),
    }
