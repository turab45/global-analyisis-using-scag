# scag/data.py

import os
import importlib
import inspect
from pathlib import Path

os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from omegaconf import OmegaConf

from cinema.classification.dataset import EndDiastoleEndSystoleDataset, get_image_transforms


# ============================================================
# COMMON CONFIG / SPLITS
# ============================================================

def _load_cinema_config(cfg):
    package = importlib.import_module(f"cinema.classification.{cfg.dataset}")
    config_path = Path(inspect.getfile(package)).resolve().parent / "config.yaml"
    config = OmegaConf.load(config_path)
    config.data.dir = str(cfg.data_root)
    config.model.views = cfg.view
    config.seed = cfg.model_seed
    classes = tuple(config.data[config.data.class_column])
    if classes != tuple(cfg.classes):
        raise RuntimeError(f"{cfg.dataset} class mismatch:\n" f"CineMA: {classes}\n" f"SCAG:   {cfg.classes}")
    return config


def _load_mnms2_splits(cfg):
    splits = {split: pd.read_csv(cfg.data_root / f"{split}_metadata.csv", dtype={"pid": str}) for split in ("train", "val", "test")}
    for split, frame in splits.items():
        frame = frame[frame["pathology"].isin(cfg.classes)].reset_index(drop=True)
        splits[split] = frame
    return splits


def _load_acdc_splits(cfg):
    development = pd.read_csv(cfg.data_root / "train_metadata.csv", dtype={"pid": str})
    test = pd.read_csv(cfg.data_root / "test_metadata.csv", dtype={"pid": str})
    val_pids = development.groupby("pathology", group_keys=False).sample(n=2, random_state=0)["pid"].astype(str).tolist()
    train = development[~development["pid"].astype(str).isin(val_pids)].reset_index(drop=True)
    val = development[development["pid"].astype(str).isin(val_pids)].reset_index(drop=True)
    return {"train": train, "val": val, "test": test.reset_index(drop=True)}


def _validate_splits(splits, cfg):
    expected = cfg.expected_sizes
    for split in ("train", "val", "test"):
        frame = splits[split]
        if len(frame) != expected[split]:
            raise RuntimeError(f"{cfg.dataset} {split}: " f"{len(frame)} patients, " f"expected {expected[split]}")
        observed = set(frame["pathology"].astype(str).unique())
        if observed != set(cfg.classes):
            raise RuntimeError(f"{cfg.dataset} {split} classes " f"{sorted(observed)} != " f"{sorted(cfg.classes)}")
    train_ids = set(splits["train"]["pid"].astype(str))
    val_ids = set(splits["val"]["pid"].astype(str))
    test_ids = set(splits["test"]["pid"].astype(str))
    if train_ids & val_ids:
        raise RuntimeError("Patient overlap between train and val.")
    if train_ids & test_ids:
        raise RuntimeError("Patient overlap between train and test.")
    if val_ids & test_ids:
        raise RuntimeError("Patient overlap between val and test.")


# ============================================================
# SHARED MASK HELPER
# ============================================================

def _pad_masks_to_image(segments, image, pid):
    if segments.shape[-2:] == image.shape[-2:]:
        return segments
    dh = image.shape[-2] - segments.shape[-2]
    dw = image.shape[-1] - segments.shape[-1]
    if dh < 0 or dw < 0:
        raise RuntimeError(f"Mask larger than image for {pid}: " f"{segments.shape} vs {image.shape}")
    return F.pad(segments, (0, dw, 0, dh))


# ============================================================
# M&Ms-2
# ============================================================

class MNMS2SemanticDataset(torch.utils.data.Dataset):
    labels = ["background", "RV", "MYO", "LV"]
    label_names = dict(enumerate(labels))
    num_labels = len(labels)

    def __init__(self, split, metadata, transform, cfg, cinema_config):
        self.split = split
        self.data_dir = cfg.data_root / split
        self.base = EndDiastoleEndSystoleDataset(data_dir=self.data_dir, meta_df=metadata.reset_index(drop=True), class_col=cinema_config.data.class_column, classes=list(cfg.classes), views=cfg.view, transform=transform)
        self.metadata = self.base.meta_df
        self.images = [f"{split}/{pid}" for pid in self.metadata["pid"].astype(str)]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        pid = str(sample["pid"])
        segments = []
        for phase in ("ed", "es"):
            path = self.data_dir / pid / f"{pid}_lax_4c_{phase}_gt.nii.gz"
            volume = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))
            segments.append(torch.from_numpy(volume[..., 0].astype(np.int64)))
        segments = torch.stack(segments)
        image = sample["lax_4c_image"]
        segments = _pad_masks_to_image(segments, image, pid)
        return image, segments, {"pid": pid, "pathology": sample["class"]}


def _load_mnms2_datasets(cfg):
    config = _load_cinema_config(cfg)
    splits = _load_mnms2_splits(cfg)
    _, eval_transform = get_image_transforms(config)
    train = MNMS2SemanticDataset("train", splits["train"], eval_transform, cfg, config)
    val = MNMS2SemanticDataset("val", splits["val"], eval_transform, cfg, config)
    test = MNMS2SemanticDataset("test", splits["test"], eval_transform, cfg, config)
    return train, val, test, splits


# ============================================================
# ACDC
# ============================================================

def _acdc_split_directory(data_root, split):
    # ACDC validation patients come from
    # the official training directory.
    directory = "train" if split == "val" else split
    return data_root / directory


class ACDCMidSAX2DDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, metadata, transform):
        self.data_dir = data_dir
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[int(index)]
        pid = str(row["pid"])
        volumes = []
        for phase in ("ed", "es"):
            path = self.data_dir / pid / f"{pid}_sax_{phase}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            volume = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(str(path)))).astype(np.float32, copy=False)
            volumes.append(volume)
        available = min(int(row["n_slices"]), volumes[0].shape[-1], volumes[1].shape[-1])
        if available < 1:
            raise RuntimeError(f"No SAX slices for {pid}")
        slice_index = (available - 1) // 2
        image = np.stack([volume[..., slice_index] for volume in volumes], axis=0)
        pathology = str(row["pathology"])
        sample = {"pid": pid, "class": pathology, "sax_image": torch.from_numpy(image), "slice_index": torch.tensor(slice_index, dtype=torch.long)}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def _get_acdc_eval_transform(config):
    patch_size = tuple(int(value) for value in config.data.sax.patch_size[:2])
    return Compose([ScaleIntensityd(keys="sax_image"), SpatialPadd(keys="sax_image", spatial_size=patch_size, method="end", lazy=True)])


class ACDCSemanticDataset(torch.utils.data.Dataset):
    labels = ["background", "RV", "MYO", "LV"]
    label_names = dict(enumerate(labels))
    num_labels = len(labels)

    def __init__(self, split, metadata, transform, cfg):
        self.split = split
        self.data_dir = _acdc_split_directory(cfg.data_root, split)
        self.base = ACDCMidSAX2DDataset(self.data_dir, metadata, transform)
        self.metadata = self.base.metadata
        self.images = [f"{split}/{pid}" for pid in self.metadata["pid"].astype(str)]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        pid = str(sample["pid"])
        slice_index = int(sample["slice_index"])
        segments = []
        for phase in ("ed", "es"):
            path = self.data_dir / pid / f"{pid}_sax_{phase}_gt.nii.gz"
            volume = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))
            segments.append(torch.from_numpy(volume[..., slice_index].astype(np.int64)))
        segments = torch.stack(segments)
        image = sample["sax_image"]
        segments = _pad_masks_to_image(segments, image, pid)
        return image, segments, {"pid": pid, "pathology": sample["class"], "slice_index": slice_index}


def _load_acdc_datasets(cfg):
    config = _load_cinema_config(cfg)
    splits = _load_acdc_splits(cfg)
    eval_transform = _get_acdc_eval_transform(config)
    train = ACDCSemanticDataset("train", splits["train"], eval_transform, cfg)
    val = ACDCSemanticDataset("val", splits["val"], eval_transform, cfg)
    test = ACDCSemanticDataset("test", splits["test"], eval_transform, cfg)
    return train, val, test, splits


# ============================================================
# PUBLIC DATA INTERFACE
# ============================================================

def load_datasets(cfg):
    if not cfg.data_root.is_dir():
        raise FileNotFoundError(f"{cfg.dataset} dataset not found: " f"{cfg.data_root}")
    if cfg.dataset == "mnms2":
        datasets = _load_mnms2_datasets(cfg)
    elif cfg.dataset == "acdc":
        datasets = _load_acdc_datasets(cfg)
    else:
        raise ValueError(f"Unsupported dataset: " f"{cfg.dataset}")
    train, val, test, splits = datasets
    _validate_splits(splits, cfg)
    print(f"Loaded {cfg.dataset.upper()}: " f"train={len(train)}, " f"val={len(val)}, " f"test={len(test)}")
    return train, val, test, splits


# ============================================================
# GRAPH ANALYSIS COHORT
# ============================================================

def build_graph_analysis_cohort(val_dataset, test_dataset, cfg):
    dataset = torch.utils.data.ConcatDataset([val_dataset, test_dataset])
    dataset.images = val_dataset.images + test_dataset.images
    expected = cfg.expected_sizes["eval"]
    if len(dataset) != expected:
        raise RuntimeError(f"Expected {expected} " f"evaluation patients, " f"found {len(dataset)}")
    print(f"[{cfg.dataset}] " f"graph cohort: " f"val ({len(val_dataset)}) + " f"test ({len(test_dataset)}) " f"= {len(dataset)}")
    print(f"[{cfg.dataset}] " "official test is not held out " "from graph analysis.")
    return dataset


# ============================================================
# SPLIT SAFETY
# ============================================================

def _dataset_paths(dataset):
    for attr in ("images", "image_files", "filenames", "files", "paths"):
        values = getattr(dataset, attr, None)
        try:
            if values is not None and len(values) == len(dataset):
                return [str(v) for v in values], attr
        except Exception:
            pass
    return None, None


def make_split_safe(train_dataset, eval_dataset):
    train_paths, train_source = _dataset_paths(train_dataset)
    eval_paths, eval_source = _dataset_paths(eval_dataset)
    if train_paths is None or eval_paths is None:
        print("[split-safe] " "path metadata unavailable; " "using full evaluation cohort.")
        return eval_dataset
    train_paths = set(train_paths)
    keep = [i for i, path in enumerate(eval_paths) if path not in train_paths]
    print(f"[split-safe] " f"train={train_source}, " f"eval={eval_source}")
    print(f"[split-safe] " f"{len(eval_dataset)} " f"-> {len(keep)} " f"(dropped " f"{len(eval_dataset)-len(keep)})")
    return torch.utils.data.Subset(eval_dataset, keep)


# ============================================================
# LOADERS
# ============================================================

def build_loaders(train_dataset, eval_dataset, batch_size=16, num_workers=0):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, eval_loader


# ============================================================
# SHARED PHASE ENCODING
# ============================================================

def phase_encode_segments(segments):
    encoded = segments.clone().long()
    encoded[:, 1] = torch.where(encoded[:, 1] > 0, encoded[:, 1] + 3, encoded[:, 1])
    return encoded
