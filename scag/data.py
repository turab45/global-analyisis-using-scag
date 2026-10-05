# scag/data.py

import os

os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from cinema.classification.dataset import (
    EndDiastoleEndSystoleDataset,
    get_image_transforms,
)

from training_code.classification.dataset import (
    local_config,
    split_metadata,
)

from training_code.classification.protocol import TASKS

from training_code.classification.acdc_2d.dataset import (
    ACDCMidSAX2DDataset,
    get_2d_transforms,
    load_acdc_config,
    load_official_splits,
    split_directory,
)


# ============================================================
# M&Ms-2
# ============================================================

class MNMS2SemanticDataset(torch.utils.data.Dataset):
    labels = ["background", "RV", "MYO", "LV"]
    label_names = dict(enumerate(labels))
    num_labels = len(labels)

    def __init__(self, split, metadata, transform, data_root, config, spec):
        self.split = split
        self.data_dir = data_root / split

        self.base = EndDiastoleEndSystoleDataset(
            data_dir=self.data_dir,
            meta_df=metadata.reset_index(drop=True),
            class_col=config.data.class_column,
            classes=list(config.data[config.data.class_column]),
            views=spec.view,
            transform=transform,
        )

        self.metadata = self.base.meta_df
        self.images = [
            f"{split}/{pid}"
            for pid in self.metadata["pid"].astype(str)
        ]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        pid = str(sample["pid"])

        segments = []

        for phase in ("ed", "es"):
            path = (
                self.data_dir
                / pid
                / f"{pid}_lax_4c_{phase}_gt.nii.gz"
            )

            volume = np.transpose(
                sitk.GetArrayFromImage(
                    sitk.ReadImage(str(path))
                )
            )

            segments.append(
                torch.from_numpy(
                    volume[..., 0].astype(np.int64)
                )
            )

        segments = torch.stack(segments)
        image = sample["lax_4c_image"]

        segments = _pad_masks_to_image(
            segments,
            image,
            pid,
        )

        return image, segments, {
            "pid": pid,
            "pathology": sample["class"],
        }


def load_mnms2_datasets(cfg):
    spec = TASKS["mnms2_lax_4c"]

    config = local_config(
        spec,
        cfg.data_root,
        seed=cfg.model_seed,
    )

    splits = split_metadata(
        spec,
        cfg.data_root,
    )

    _, eval_transform = get_image_transforms(config)

    train = MNMS2SemanticDataset(
        "train",
        splits["train"],
        eval_transform,
        cfg.data_root,
        config,
        spec,
    )

    val = MNMS2SemanticDataset(
        "val",
        splits["val"],
        eval_transform,
        cfg.data_root,
        config,
        spec,
    )

    test = MNMS2SemanticDataset(
        "test",
        splits["test"],
        eval_transform,
        cfg.data_root,
        config,
        spec,
    )

    return train, val, test, splits


# ============================================================
# ACDC
# ============================================================

class ACDCSemanticDataset(torch.utils.data.Dataset):
    labels = ["background", "RV", "MYO", "LV"]
    label_names = dict(enumerate(labels))
    num_labels = len(labels)

    def __init__(self, split, metadata, transform, data_root):
        self.split = split
        self.data_dir = split_directory(
            data_root,
            split,
        )

        self.base = ACDCMidSAX2DDataset(
            self.data_dir,
            metadata,
            transform,
        )

        self.metadata = self.base.metadata
        self.images = [
            f"{split}/{pid}"
            for pid in self.metadata["pid"].astype(str)
        ]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]

        pid = str(sample["pid"])
        slice_index = int(sample["slice_index"])

        segments = []

        for phase in ("ed", "es"):
            path = (
                self.data_dir
                / pid
                / f"{pid}_sax_{phase}_gt.nii.gz"
            )

            volume = np.transpose(
                sitk.GetArrayFromImage(
                    sitk.ReadImage(str(path))
                )
            )

            segments.append(
                torch.from_numpy(
                    volume[..., slice_index].astype(np.int64)
                )
            )

        segments = torch.stack(segments)
        image = sample["sax_image"]

        segments = _pad_masks_to_image(
            segments,
            image,
            pid,
        )

        return image, segments, {
            "pid": pid,
            "pathology": sample["class"],
            "slice_index": slice_index,
        }


def load_acdc_datasets(cfg):
    splits = load_official_splits(
        cfg.data_root
    )

    config = load_acdc_config(
        cfg.data_root,
        seed=cfg.model_seed,
    )

    _, eval_transform = get_2d_transforms(config)

    train = ACDCSemanticDataset(
        "train",
        splits["train"],
        eval_transform,
        cfg.data_root,
    )

    val = ACDCSemanticDataset(
        "val",
        splits["val"],
        eval_transform,
        cfg.data_root,
    )

    test = ACDCSemanticDataset(
        "test",
        splits["test"],
        eval_transform,
        cfg.data_root,
    )

    return train, val, test, splits


# ============================================================
# COMMON DATASET INTERFACE
# ============================================================

def load_datasets(cfg):
    if not cfg.data_root.is_dir():
        raise FileNotFoundError(
            f"{cfg.dataset.upper()} dataset not found: "
            f"{cfg.data_root}"
        )

    if cfg.dataset == "mnms2":
        datasets = load_mnms2_datasets(cfg)

    elif cfg.dataset == "acdc":
        datasets = load_acdc_datasets(cfg)

    else:
        raise ValueError(
            f"Unsupported dataset: {cfg.dataset}"
        )

    train, val, test, splits = datasets

    expected = cfg.expected_sizes

    actual = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }

    expected_raw = {
        "train": expected["train"],
        "val": expected["val"],
        "test": expected["test"],
    }

    if actual != expected_raw:
        raise RuntimeError(
            f"Unexpected {cfg.dataset} split sizes.\n"
            f"Expected: {expected_raw}\n"
            f"Found:    {actual}"
        )

    print(
        f"Loaded {cfg.dataset.upper()}: "
        f"train={len(train)}, "
        f"val={len(val)}, "
        f"test={len(test)}"
    )

    return train, val, test, splits


# ============================================================
# GRAPH-ANALYSIS COHORT
# ============================================================

def build_graph_analysis_cohort(
    val_dataset,
    test_dataset,
    cfg,
):
    dataset = torch.utils.data.ConcatDataset(
        [val_dataset, test_dataset]
    )

    dataset.images = (
        val_dataset.images
        + test_dataset.images
    )

    expected = cfg.expected_sizes["eval"]

    if len(dataset) != expected:
        raise RuntimeError(
            f"Expected graph-analysis cohort of "
            f"{expected}, got {len(dataset)}."
        )

    print(
        f"[{cfg.dataset}] graph cohort: "
        f"val ({len(val_dataset)}) + "
        f"test ({len(test_dataset)}) "
        f"= {len(dataset)} patients."
    )

    print(
        f"[{cfg.dataset}] official test set is "
        "NOT held out in this graph analysis."
    )

    return dataset


# ============================================================
# SPLIT SAFETY
# ============================================================

def _dataset_paths(dataset):
    for attr in [
        "images",
        "image_files",
        "filenames",
        "files",
        "paths",
    ]:
        values = getattr(dataset, attr, None)

        try:
            if values is not None and len(values) == len(dataset):
                return [str(v) for v in values], attr
        except Exception:
            pass

    samples = getattr(dataset, "samples", None)

    try:
        if samples is not None and len(samples) == len(dataset):
            paths = [
                str(sample[0])
                if isinstance(sample, (tuple, list))
                and len(sample)
                else str(sample)
                for sample in samples
            ]

            return paths, "samples"

    except Exception:
        pass

    return None, None


def make_split_safe(
    train_dataset,
    eval_dataset,
):
    train_paths, train_source = _dataset_paths(
        train_dataset
    )

    eval_paths, eval_source = _dataset_paths(
        eval_dataset
    )

    if train_paths is None or eval_paths is None:
        print(
            "[split-safe] path metadata not found; "
            "using full graph cohort."
        )

        return eval_dataset

    train_paths = set(train_paths)

    keep = [
        i
        for i, path in enumerate(eval_paths)
        if path not in train_paths
    ]

    print(
        f"[split-safe] train source={train_source}, "
        f"eval source={eval_source}"
    )

    print(
        f"[split-safe] {len(eval_dataset)} -> "
        f"{len(keep)} "
        f"(dropped {len(eval_dataset) - len(keep)} "
        "train-overlapping)"
    )

    return torch.utils.data.Subset(
        eval_dataset,
        keep,
    )


# ============================================================
# LOADERS
# ============================================================

def build_loaders(
    train_dataset,
    eval_dataset,
    batch_size=16,
    num_workers=0,
):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, eval_loader


# ============================================================
# SHARED MASK PROCESSING
# ============================================================

def _pad_masks_to_image(
    segments,
    image,
    pid,
):
    if segments.shape[-2:] == image.shape[-2:]:
        return segments

    dh = image.shape[-2] - segments.shape[-2]
    dw = image.shape[-1] - segments.shape[-1]

    if dh < 0 or dw < 0:
        raise RuntimeError(
            f"Mask larger than image for {pid}: "
            f"{segments.shape} vs {image.shape}"
        )

    return F.pad(
        segments,
        (0, dw, 0, dh),
    )


def phase_encode_segments(segments):
    encoded = segments.clone().long()

    encoded[:, 1] = torch.where(
        encoded[:, 1] > 0,
        encoded[:, 1] + 3,
        encoded[:, 1],
    )

    return encoded