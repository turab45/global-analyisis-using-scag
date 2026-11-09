# scag/activations.py

import torch


def extract_activations(model, loader, hook_target, device):
    activations, segments = [], []

    def hook(module, inputs, output):
        if not torch.is_tensor(output) or output.ndim != 4:
            raise ValueError("SCAG requires a spatial NCHW layer output.")
        # Later in-place activations must not change the captured layer output.
        activations.append(output.detach().to("cpu", copy=True))

    handle = hook_target.register_forward_hook(hook)

    try:
        with torch.no_grad():
            for batch in loader:
                before = len(activations)
                model(batch[0].to(device))
                if len(activations) != before + 1:
                    raise ValueError("SCAG requires a layer called once per batch (stacked ED/ES formulation).")

                if torch.is_tensor(batch[1]):
                    segments.append(batch[1].cpu().to(torch.int16))
    finally:
        handle.remove()

    activations = torch.cat(activations, dim=0)
    segments = torch.cat(segments, dim=0) if segments else None

    return activations, segments
