"""Hook-based feature extraction from frozen models."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ActivationSet:
    maps: list
    masks: list
    pooled: np.ndarray
    patient_ids: list[str]


@torch.inference_mode()
def extract_activations(model, layer, loader, device="cpu", keep_maps=True):
    """Extract one neuron response per patient with hooks removed on failure.

    Shared ED/ES encoders invoke the layer twice. Their feature maps are averaged
    within a patient, rather than treating the two phases as separate patients.
    Different input sizes are forwarded individually to preserve preprocessing.
    """
    model.eval()
    captured, maps, masks, pooled, patients = [], [], [], [], []

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output) or output.ndim != 4:
            raise ValueError("Select a spatial 2-D feature layer with output (N, C, H, W)")
        captured.append(output.detach().cpu())

    handle = model.get_submodule(layer).register_forward_hook(hook)
    try:
        for samples in loader:
            groups = [samples] if len({tuple(s["image"].shape) for s in samples}) == 1 else [[s] for s in samples]
            for group in groups:
                captured.clear()
                model(torch.stack([s["image"] for s in group]).to(device))
                expected = 2 if getattr(model, "formulation", "stacked") == "shared" else 1
                if len(captured) != expected:
                    raise ValueError(f"Expected {expected} feature-layer calls, got {len(captured)}")
                features = torch.stack(captured).mean(dim=0)
                if not torch.isfinite(features).all():
                    raise ValueError("Non-finite model activations")
                pooled.extend(features.mean(dim=(-2, -1)).numpy())
                patients.extend(s["pid"] for s in group)
                if keep_maps:
                    maps.extend(features.unbind(0))
                    masks.extend(s["masks"] for s in group)
    finally:
        handle.remove()
    if not pooled:
        raise ValueError("Analysis cohort is empty")
    return ActivationSet(maps, masks, np.asarray(pooled), patients)
