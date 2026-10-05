import torch


def build_graph_analysis_cohort(train_dataset, val_dataset, dataset_class, splits, eval_transform):
    test_dataset = dataset_class("test", splits["test"], eval_transform)
    val_dataset_v4 = torch.utils.data.ConcatDataset([val_dataset, test_dataset])
    val_dataset_v4.images = val_dataset.images + test_dataset.images

    sizes = {"train": len(train_dataset), "val": len(val_dataset_v4)}
    assert sizes == {"train": 160, "val": 140}, sizes

    print("[v4] Graph cohort: val (30) + test (110) = 140 patients.")
    print("[v4] Official test set is NOT held out in this analysis.")
    return val_dataset_v4


def _dataset_paths(dataset):
    for attr in ["images", "image_files", "filenames", "files", "paths"]:
        values = getattr(dataset, attr, None)
        try:
            if values is not None and len(values) == len(dataset):
                return [str(v) for v in values], attr
        except Exception:
            pass

    samples = getattr(dataset, "samples", None)
    try:
        if samples is not None and len(samples) == len(dataset):
            paths = [str(s[0]) if isinstance(s, (tuple, list)) and len(s) else str(s) for s in samples]
            return paths, "samples"
    except Exception:
        pass

    return None, None


def make_split_safe(train_dataset, val_dataset):
    train_paths, train_source = _dataset_paths(train_dataset)
    val_paths, val_source = _dataset_paths(val_dataset)

    if train_paths is None or val_paths is None:
        print("[split-safe] path metadata not found; using full graph cohort.")
        return val_dataset

    train_paths = set(train_paths)
    keep = [i for i, path in enumerate(val_paths) if path not in train_paths]

    print(f"[split-safe] train source={train_source}, val source={val_source}")
    print(f"[split-safe] {len(val_dataset)} -> {len(keep)} "
          f"(dropped {len(val_dataset) - len(keep)} train-overlapping)")

    return torch.utils.data.Subset(val_dataset, keep)


def build_loaders(train_dataset, val_dataset, batch_size=16, num_workers=0):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def phase_encode_segments(segments):
    encoded = segments.clone().long()
    encoded[:, 1] = torch.where(encoded[:, 1] > 0, encoded[:, 1] + 3, encoded[:, 1])
    return encoded