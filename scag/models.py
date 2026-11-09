# scag/models.py

import torch
import torch.nn as nn
from torchvision import models


def torchvision_weights(name, initialization):
    if initialization == "randinit":
        return None

    weights = {
        "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.DEFAULT,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "efficientnet_b3": models.EfficientNet_B3_Weights.DEFAULT,
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "densenet161": models.DenseNet161_Weights.DEFAULT,
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "convnext_small": models.ConvNeXt_Small_Weights.DEFAULT,
    }

    if name not in weights:
        raise ValueError(f"Unsupported backbone: {name}")

    return weights[name]


def adapt_first_conv(conv, in_channels, pretrained):
    replacement = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        if pretrained and conv.in_channels == 3:
            grayscale = conv.weight.sum(dim=1, keepdim=True)

            if in_channels == 1:
                replacement.weight.copy_(grayscale)
            else:
                replacement.weight.copy_(
                    grayscale.repeat(1, in_channels, 1, 1)
                    / in_channels
                )
        else:
            nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu",)

        if replacement.bias is not None:
            if conv.bias is None:
                replacement.bias.zero_()
            else:
                replacement.bias.copy_(conv.bias)

    return replacement


def make_2d_encoder(name, in_channels, initialization):
    pretrained = initialization == "imagenet"
    weights = torchvision_weights(name, initialization)

    model = getattr(models, name)(weights=weights)

    if name.startswith("resnet"):
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        model.conv1 = adapt_first_conv(model.conv1, in_channels, pretrained,)

    elif name.startswith("densenet"):
        feature_dim = int(model.classifier.in_features)
        model.classifier = nn.Identity()
        model.features.conv0 = adapt_first_conv(model.features.conv0, in_channels,pretrained,)

    elif (
        name.startswith("efficientnet")
        or name.startswith("convnext")
        or name.startswith("mobilenet_v3")
    ):
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels, pretrained,)

    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return model, feature_dim


class EDES2DClassifier(nn.Module):
    """2-D ED/ES classifier used by both M&Ms-2 and ACDC."""

    def __init__(self, backbone, formulation, initialization, n_classes,):
        super().__init__()

        self.formulation = formulation
        self.initialization = initialization

        if formulation == "stacked":
            self.encoder, feature_dim = make_2d_encoder(backbone, in_channels=2, initialization=initialization,)
            classifier_dim = feature_dim

        elif formulation == "shared":
            self.encoder, feature_dim = make_2d_encoder(backbone,in_channels=1, initialization=initialization,)
            classifier_dim = feature_dim * 3

        else:
            raise ValueError(f"Unknown formulation: {formulation}")

        self.classifier = nn.Linear(classifier_dim, n_classes,)
        self.feature_dim = feature_dim

    def normalize(self, image):
        if self.initialization != "imagenet":
            return image

        return (image - 0.449) / 0.226

    def forward_features(self, image):
        image = self.normalize(image)

        if self.formulation == "stacked":
            return self.encoder(image)

        ed = self.encoder(image[:, 0:1])
        es = self.encoder(image[:, 1:2])

        return {"ed": ed, "es": es, "delta": es - ed,}

    def forward(self, image):
        features = self.forward_features(image)

        if isinstance(features, dict):
            features = torch.cat([features["ed"], features["es"], features["delta"],], dim=1,)

        return self.classifier(features)


def _expected_checkpoint_metadata(tag, cfg):
    if cfg.dataset == "mnms2":
        return {
            "family": "architecture_bank",
            "task": cfg.task_name,
            "dimensionality": 2,
            "backbone": tag,
            "formulation": cfg.formulation,
            "initialization": cfg.initialization,
            "classes": list(cfg.classes),
        }

    if cfg.dataset == "acdc":
        return {
            "family": "acdc_2d_architecture_bank",
            "backbone": tag,
            "formulation": cfg.formulation,
            "initialization": cfg.initialization,
        }

    raise ValueError(f"Unsupported dataset: {cfg.dataset}")


def build_model(tag, cfg):
    checkpoint_path = (
        cfg.model_root
        / tag
        / cfg.formulation
        / cfg.initialization
        / f"seed_{cfg.model_seed}"
        / "best_val_mcc.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path,map_location="cpu",weights_only=False,)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu",)

    expected = _expected_checkpoint_metadata(tag, cfg,)

    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"Checkpoint {key}={state.get(key)!r}, expected {value!r}")

    model = EDES2DClassifier(backbone=tag, formulation=cfg.formulation, initialization="randinit", n_classes=len(cfg.classes),)

    model.load_state_dict(state["model"],strict=True,)

    # We construct without downloading ImageNet weights because
    # checkpoint weights are loaded immediately afterwards.
    # This restores the normalization used during training.
    model.initialization = cfg.initialization

    model.feature_layer_candidates = tuple(state["feature_layer_candidates"])
    layer_name = model.feature_layer_candidates[-1]

    modules = dict(model.named_modules())

    if layer_name not in modules:
        raise RuntimeError(f"Feature layer {layer_name!r} not found in {tag}.")

    hook_target = modules[layer_name]

    print(f"[{cfg.dataset}/{tag}] loaded checkpoint: {checkpoint_path}")
    print(f"[{cfg.dataset}/{tag}] SCAG layer: {layer_name}")

    return model, hook_target, layer_name
