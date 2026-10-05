# scag/models.py

import torch


def build_model(tag, model_root, model_class, classes, cfg):
    checkpoint_path = (
        model_root / tag / cfg.formulation / cfg.initialization
        / f"seed_{cfg.model_seed}" / "best_val_mcc.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")

    expected = {
        "family": "architecture_bank",
        "task": "mnms2_lax_4c",
        "dimensionality": 2,
        "backbone": tag,
        "formulation": cfg.formulation,
        "initialization": cfg.initialization,
        "classes": list(classes),
    }

    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(
                f"Checkpoint {key}={state.get(key)!r}, expected {value!r}"
            )

    model = model_class(
        tag,
        cfg.formulation,
        "randinit",
        n_classes=len(classes),
    )

    model.load_state_dict(state["model"], strict=True)
    model.initialization = cfg.initialization

    layer_name = state["feature_layer_candidates"][-1]
    hook_target = dict(model.named_modules())[layer_name]

    print(f"[{tag}] loaded checkpoint: {checkpoint_path}")

    return model, hook_target, layer_name