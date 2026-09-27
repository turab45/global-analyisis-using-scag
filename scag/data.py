"""Patient cohorts and paired images/masks for global analysis."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .preprocessing import load_volume, preprocess_pair


def load_splits(root, *, dataset, classes, expected_sizes=None, validation_per_class=2, split_seed=0):
    """Load the existing CineMA splits, checking patient-level separation."""
    root = Path(root).expanduser()
    names = ("train", "test") if dataset == "acdc" else ("train", "val", "test")
    if dataset not in ("acdc", "mnms2"):
        raise ValueError("dataset must be acdc or mnms2")
    splits = {}
    for name in names:
        frame = pd.read_csv(root / f"{name}_metadata.csv", dtype={"pid": str})
        missing = {"pid", "pathology", "n_slices"} - set(frame.columns)
        if missing or frame[["pid", "pathology", "n_slices"]].isna().any().any():
            raise ValueError(f"Invalid {name} metadata; required columns: pid, pathology, n_slices")
        if dataset == "mnms2":
            frame = frame[frame.pathology.isin(classes)].copy()
        if frame.pid.duplicated().any() or not set(frame.pathology).issubset(classes):
            raise ValueError(f"Duplicate patient IDs or unknown classes in {name}")
        if any(Path(pid).name != pid or pid in (".", "..") for pid in frame.pid):
            raise ValueError("Patient IDs must be single path components")
        splits[name] = frame.reset_index(drop=True)
    if dataset == "acdc":
        development = splits["train"]
        val = development.groupby("pathology", group_keys=False).sample(n=validation_per_class, random_state=split_seed)
        splits["val"] = development[development.pid.isin(val.pid)].reset_index(drop=True)
        splits["train"] = development[~development.pid.isin(val.pid)].reset_index(drop=True)
    for name, frame in splits.items():
        if expected_sizes is not None and len(frame) != expected_sizes[name]:
            raise ValueError(f"{name}: found {len(frame)} patients, expected {expected_sizes[name]}")
        if set(frame.pathology) != set(classes):
            raise ValueError(f"{name} does not contain the configured class set")
    for first, second in combinations(splits, 2):
        overlap = set(splits[first].pid) & set(splits[second].pid)
        if overlap:
            raise ValueError(f"Patient leakage between {first} and {second}: {sorted(overlap)}")
    return splits


class SemanticDataset(Dataset):
    """One ED/ES image pair and matching RV/MYO/LV masks per patient."""

    def __init__(self, root, splits, selected_splits, *, dataset, view="sax", minimum_size=(192, 192), masks_root=None):
        self.root = Path(root).expanduser()
        self.masks_root = Path(masks_root).expanduser() if masks_root else self.root
        self.view = view
        self.minimum_size = tuple(minimum_size)
        if view not in ("sax", "lax_4c"):
            raise ValueError("Only central SAX and LAX-4C analysis are supported")
        if not selected_splits or len(set(selected_splits)) != len(selected_splits):
            raise ValueError("Select distinct, non-empty analysis splits")
        rows = []
        for split in selected_splits:
            frame = splits[split].copy()
            frame["split"] = split
            frame["directory"] = "train" if dataset == "acdc" and split == "val" else split
            rows.append(frame)
        self.metadata = pd.concat(rows, ignore_index=True)
        self.patient_ids = self.metadata.pid.tolist()
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("Patients appear more than once in this cohort")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        pid, directory = row.pid, row.directory
        images, masks = [], []
        for phase in ("ed", "es"):
            filename = f"{pid}_{self.view}_{phase}"
            images.append(load_volume(self.root / directory / pid / f"{filename}.nii.gz"))
            masks.append(load_volume(self.masks_root / directory / pid / f"{filename}_gt.nii.gz"))
        if any(image.shape != mask.shape for image, mask in zip(images, masks)):
            raise ValueError(f"Image/mask geometry differs for {pid}")
        if self.view == "sax":
            count = min(int(row.n_slices), images[0].shape[-1], images[1].shape[-1])
            if count < 1:
                raise ValueError(f"No acquired slices for {pid}")
            slice_index = (count - 1) // 2
        else:
            slice_index = 0
        image, mask = preprocess_pair(
            np.stack([value[..., slice_index] for value in images]),
            np.stack([value[..., slice_index] for value in masks]),
            self.minimum_size,
        )
        return {
            "image": image,
            "masks": mask,
            "pid": pid,
            "pathology": row.pathology,
            "split": row.split,
            "slice_index": slice_index,
        }


def collate_samples(samples):
    """Keep native geometry; activation extraction groups equal-sized inputs."""
    return samples
