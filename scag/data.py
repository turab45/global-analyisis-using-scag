import importlib
import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from omegaconf import OmegaConf

os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")


class EndDiastoleEndSystoleDataset(torch.utils.data.Dataset):
    """ED/ES image loader extracted from the CineMA dataset implementation."""

    def __init__(self, data_dir, meta_df, class_col, classes, views, transform=None, dtype=np.float32):
        for col in ["pid", "n_slices"]:
            if col not in meta_df.columns:
                raise ValueError(f"Column {col} is required in meta_df.")
        self.data_dir = Path(data_dir)
        self.meta_df = meta_df
        self.views = [views] if isinstance(views, str) else views
        self.transform = transform
        self.dtype = dtype
        self.classes = classes
        self.class_col = class_col

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.meta_df.iloc[idx]
        pid = row["pid"]
        pid_dir = self.data_dir / pid
        cls = row[self.class_col]
        data = {"pid": pid, "class": cls, "label": torch.tensor(self.classes.index(cls))}

        for view in self.views:
            ed = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(pid_dir / f"{pid}_{view}_ed.nii.gz")))
            es = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(pid_dir / f"{pid}_{view}_es.nii.gz")))
            image = np.stack([ed, es], axis=0)
            if view != "sax":
                image = image[..., 0]
            data[f"{view}_image"] = torch.from_numpy(image)

        return self.transform(data) if self.transform else data


def load_cinema_config(cfg):
    """Load the exact CineMA M&Ms2 config used by the training code."""
    package = importlib.import_module("cinema.classification.mnms2")
    path = Path(inspect.getfile(package)).resolve().parent / "config.yaml"
    config = OmegaConf.load(path)
    config.data.dir = str(cfg.data_root)
    config.model.views = cfg.view
    config.seed = cfg.model_seed

    classes = tuple(config.data[config.data.class_column])
    if classes != cfg.classes:
        raise RuntimeError(f"Installed CineMA class list changed: {list(classes)} != {list(cfg.classes)}")
    return config


def get_eval_transform(config):
    """Evaluation transform used by the supplied CineMA dataset code."""
    views = config.model.views
    views = [views] if isinstance(views, str) else views
    transforms = []
    for view in views:
        patch_size = config.data.sax.patch_size if view == "sax" else config.data.lax.patch_size
        transforms += [
            ScaleIntensityd(keys=f"{view}_image"),
            SpatialPadd(keys=f"{view}_image", spatial_size=patch_size, method="end", lazy=True),
        ]
    return Compose(transforms)


def split_metadata(cfg):
    """Load and validate the official M&Ms2 train/val/test metadata."""
    splits = {
        name: pd.read_csv(cfg.data_root / f"{name}_metadata.csv", dtype={"pid": str})
        for name in ["train", "val", "test"]
    }
    for name in splits:
        splits[name] = splits[name][splits[name]["pathology"].isin(cfg.classes)].reset_index(drop=True)

    expected = {"train": cfg.expected_train, "val": cfg.expected_val, "test": cfg.expected_test}
    for name, frame in splits.items():
        if len(frame) != expected[name]:
            raise RuntimeError(f"{cfg.task} {name} has {len(frame)} patients; expected {expected[name]}.")
        observed = set(frame["pathology"].unique())
        if observed != set(cfg.classes):
            raise RuntimeError(f"{cfg.task} {name} classes {sorted(observed)} != {sorted(cfg.classes)}")

    pid_sets = {k: set(v["pid"].astype(str)) for k, v in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = pid_sets[a] & pid_sets[b]
        if overlap:
            raise RuntimeError(f"Patient leakage in {cfg.task}: {a}/{b}: {sorted(overlap)}")
    return splits


class MNMS2SemanticDataset(torch.utils.data.Dataset):
    """Training-compatible LAX-4C ED/ES inputs plus cardiac anatomy masks."""

    labels = ["background", "RV", "MYO", "LV"]
    label_names = dict(enumerate(labels))
    num_labels = len(labels)
    categories = ["anatomy"]
    label_category = np.zeros(num_labels, dtype=np.int64)

    def __init__(self, split, metadata, transform, cfg, cinema_config):
        self.split = split
        self.data_dir = cfg.data_root / split
        self.base = EndDiastoleEndSystoleDataset(
            data_dir=self.data_dir,
            meta_df=metadata.reset_index(drop=True),
            class_col=cinema_config.data.class_column,
            classes=list(cinema_config.data[cinema_config.data.class_column]),
            views=cfg.view,
            transform=transform,
        )
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

        if segments.shape[-2:] != image.shape[-2:]:
            dh = image.shape[-2] - segments.shape[-2]
            dw = image.shape[-1] - segments.shape[-1]
            if dh < 0 or dw < 0:
                raise RuntimeError(f"Mask larger than image for {pid}: {segments.shape} vs {image.shape}")
            segments = torch.nn.functional.pad(segments, (0, dw, 0, dh))

        return image, segments, {"pid": pid, "pathology": sample["class"]}


def load_mnms2_datasets(cfg):
    if not cfg.data_root.is_dir():
        raise FileNotFoundError(f"Processed MnMs2 dataset not found: {cfg.data_root}")
    if not cfg.model_root.is_dir():
        raise FileNotFoundError(f"MnMs2 LAX-4C checkpoint bank not found: {cfg.model_root}")

    cinema_config = load_cinema_config(cfg)
    splits = split_metadata(cfg)
    eval_transform = get_eval_transform(cinema_config)

    train = MNMS2SemanticDataset("train", splits["train"], eval_transform, cfg, cinema_config)
    val = MNMS2SemanticDataset("val", splits["val"], eval_transform, cfg, cinema_config)
    test = MNMS2SemanticDataset("test", splits["test"], eval_transform, cfg, cinema_config)

    sizes = (len(train), len(val), len(test))
    expected = (cfg.expected_train, cfg.expected_val, cfg.expected_test)
    if sizes != expected:
        raise RuntimeError(f"Unexpected dataset sizes {sizes}; expected {expected}")

    print(f"Loaded MnMs2 LAX-4C train patients: {len(train)}")
    print(f"Loaded MnMs2 LAX-4C val patients: {len(val)}")
    print(f"Loaded MnMs2 LAX-4C test patients: {len(test)}")
    print(f"Anatomy concepts: {train.label_names}")
    return train, val, test, splits


def build_graph_analysis_cohort(val_dataset, test_dataset):
    dataset = torch.utils.data.ConcatDataset([val_dataset, test_dataset])
    dataset.images = val_dataset.images + test_dataset.images
    print(f"[SCAG] Graph cohort: val ({len(val_dataset)}) + test ({len(test_dataset)}) = {len(dataset)} patients.")
    print("[SCAG] Official test set is NOT held out in this graph analysis.")
    return dataset


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


def make_split_safe(train_dataset, eval_dataset):
    train_paths, train_source = _dataset_paths(train_dataset)
    eval_paths, eval_source = _dataset_paths(eval_dataset)

    if train_paths is None or eval_paths is None:
        print("[split-safe] path metadata not found; using full graph cohort.")
        return eval_dataset

    train_paths = set(train_paths)
    keep = [i for i, path in enumerate(eval_paths) if path not in train_paths]
    print(f"[split-safe] train source={train_source}, eval source={eval_source}")
    print(f"[split-safe] {len(eval_dataset)} -> {len(keep)} (dropped {len(eval_dataset) - len(keep)} train-overlapping)")
    return torch.utils.data.Subset(eval_dataset, keep)


def build_loaders(train_dataset, eval_dataset, batch_size=16, num_workers=0):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, eval_loader


def phase_encode_segments(segments):
    encoded = segments.clone().long()
    encoded[:, 1] = torch.where(encoded[:, 1] > 0, encoded[:, 1] + 3, encoded[:, 1])
    return encoded
