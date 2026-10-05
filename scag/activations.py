# scag/activations.py

import torch


def extract_activations(model, loader, hook_target, device):
    activations, segments = [], []

    def hook(module, inputs, output):
        activations.append(output.detach().cpu())

    handle = hook_target.register_forward_hook(hook)

    try:
        with torch.no_grad():
            for batch in loader:
                model(batch[0].to(device))

                if torch.is_tensor(batch[1]):
                    segments.append(batch[1].cpu().to(torch.int16))
    finally:
        handle.remove()

    activations = torch.cat(activations, dim=0)
    segments = torch.cat(segments, dim=0) if segments else None

    return activations, segments