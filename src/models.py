"""Inference-only reconstruction of the existing 2-D classification checkpoints."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models


def make_encoder(backbone, in_channels):
    if backbone.startswith(("resnet", "resnext", "wide_resnet")):
        first, last = "conv1", "fc"
    elif backbone.startswith("densenet"):
        first, last = "features.conv0", "classifier"
    elif backbone.startswith(("efficientnet", "convnext", "mobilenet_v3")):
        first, last = "features.0.0", "classifier"
    else:
        raise ValueError(f"Unsupported 2-D backbone: {backbone}")
    # State dictionaries supply all weights; never download or initialize a pretrained model.
    encoder = getattr(models, backbone)(weights=None)
    head = encoder.get_submodule(last)
    if isinstance(head, nn.Sequential):
        last += f".{len(head) - 1}"
        head = head[-1]
    dimension = head.in_features
    parent, _, name = last.rpartition(".")
    setattr(encoder.get_submodule(parent) if parent else encoder, name, nn.Identity())
    conv = encoder.get_submodule(first)
    replacement = nn.Conv2d(
        in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        conv.bias is not None,
        conv.padding_mode,
    )
    parent, _, name = first.rpartition(".")
    setattr(encoder.get_submodule(parent) if parent else encoder, name, replacement)
    return encoder, dimension


class EDESClassifier(nn.Module):
    """Checkpoint-compatible ED/ES fusion; used only in evaluation mode."""

    def __init__(self, backbone, formulation, initialization, n_classes):
        super().__init__()
        if formulation not in ("stacked", "shared") or initialization not in ("randinit", "imagenet"):
            raise ValueError("Unsupported fusion or initialization metadata")
        self.formulation, self.initialization = formulation, initialization
        self.encoder, self.feature_dim = make_encoder(backbone, 2 if formulation == "stacked" else 1)
        self.classifier = nn.Linear(self.feature_dim * (1 if formulation == "stacked" else 3), n_classes)

    def forward(self, image):
        if self.initialization == "imagenet":
            image = (image - 0.449) / 0.226
        if self.formulation == "stacked":
            features = self.encoder(image)
        else:
            ed, es = self.encoder(image[:, :1]), self.encoder(image[:, 1:2])
            features = torch.cat((ed, es, es - ed), dim=1)
        return self.classifier(features)


def load_model(checkpoint, *, classes, task, layer=None, device="cpu"):
    """Load a state-dictionary checkpoint and verify its cohort metadata."""
    path = Path(checkpoint).expanduser()
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"classes": list(classes), "task": task, "dimensionality": 2}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Checkpoint {key}={state.get(key)!r}; expected {value!r}")
    model = EDESClassifier(state["backbone"], state["formulation"], state["initialization"], len(classes))
    model.load_state_dict(state["model"], strict=True)
    if layer is None:
        candidates = state.get("feature_layer_candidates", [])
        if not candidates:
            raise ValueError("No feature layer in checkpoint; set model.layer in YAML")
        layer = candidates[-1]
    model.get_submodule(layer)
    model.requires_grad_(False).eval().to(device)
    metadata = {
        key: state.get(key)
        for key in ("family", "task", "backbone", "formulation", "initialization", "classes", "seed", "epoch")
    }
    metadata.update(checkpoint=str(path.resolve()), layer=layer)
    return model, layer, metadata
