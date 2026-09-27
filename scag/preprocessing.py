"""Deterministic analysis preprocessing; no augmentation or fitting."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def load_volume(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    volume = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError(f"Expected a finite 3-D NIfTI volume: {path}")
    return volume


def preprocess_pair(image, masks, minimum_size=(192, 192)):
    """Match CineMA evaluation: pairwise min/max scaling, then end padding.

    Masks follow the same geometry and retain their integer anatomy IDs. Images
    larger than minimum_size are not resized or cropped.
    """
    image = torch.as_tensor(image, dtype=torch.float32)
    masks = torch.as_tensor(masks)
    if image.ndim != 3 or image.shape[0] != 2 or masks.shape != image.shape:
        raise ValueError("Images and masks must both have shape (2, H, W)")
    if not torch.isfinite(image).all() or not torch.isfinite(masks).all():
        raise ValueError("Images and masks must be finite")
    if not torch.all((masks >= 0) & (masks <= 3) & (masks == masks.round())):
        raise ValueError("Anatomy masks must use integer IDs: 0 background, 1 RV, 2 MYO, 3 LV")
    low, high = image.min(), image.max()
    image = (image - low) / (high - low) if high > low else torch.zeros_like(image)
    height, width = (int(value) for value in minimum_size)
    if height <= 0 or width <= 0:
        raise ValueError("minimum_size must be positive")
    padding = (0, max(0, width - image.shape[-1]), 0, max(0, height - image.shape[-2]))
    return F.pad(image, padding), F.pad(masks.long(), padding)
