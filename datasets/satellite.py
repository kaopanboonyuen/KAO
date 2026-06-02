# =============================================================================
# KAO: Kernel-Adaptive Optimization in Diffusion for Satellite Image
# =============================================================================
# Author  : Teerapong Panboonyuen (aka Kao Panboonyuen)
# Paper   : "KAO: Kernel-Adaptive Optimization in Diffusion for Satellite Image"
# Journal : IEEE Transactions on Geoscience and Remote Sensing (IF: 8.9)
# DOI     : https://doi.org/10.1109/TGRS.2025.3621738
# Project : https://kaopanboonyuen.github.io/KAO/
# ORCID   : https://orcid.org/0000-0001-8464-4476
# =============================================================================
# This file implements the dataset loaders described in Section IV-B:
#
#   Massachusetts Roads Dataset
#     • 1171 aerial RGB images, 1500×1500 px
#     • Road structure recovery (thin, sharp features)
#     • Reference: Mnih, V., "Machine Learning for Aerial Image Labeling",
#                  PhD Thesis, University of Toronto, 2013.
#
#   DeepGlobe 2018 Dataset
#     • 803 VHR satellite images, 50 cm/pixel
#     • Diverse land cover: urban, agricultural, forested areas
#     • Reference: Demir et al., "DeepGlobe 2018", CVPR Workshops 2018.
#
#   Masking strategy (Appendix — Training Setup):
#     • Random masks covering 30–50% of image area
#     • Augmentation: random flips, rotations, scaling
# =============================================================================
# MIT License — Copyright (c) 2025 Teerapong Panboonyuen
# =============================================================================

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Mask generation utilities  (Appendix — Data Augmentation and Masking Strategy)
# ---------------------------------------------------------------------------

def generate_random_mask(
    height: int,
    width:  int,
    min_ratio: float = 0.30,
    max_ratio: float = 0.50,
    num_rectangles: int = 5,
) -> torch.Tensor:
    """
    Generate a random binary mask (1 = known, 0 = missing).

    Follows the masking strategy described in the Appendix:
        "We randomly mask 30-50% of the image using binary masks,
         following the strategy from [RePaint]."

    Returns:
        mask : (1, H, W) float tensor where 1 = known region
    """
    mask   = torch.ones(1, height, width)
    target = random.uniform(min_ratio, max_ratio)
    total  = height * width
    masked = 0

    for _ in range(num_rectangles):
        if masked / total >= target:
            break
        # Sample random rectangular block
        h   = random.randint(height // 10, height // 3)
        w   = random.randint(width  // 10, width  // 3)
        top = random.randint(0, height - h)
        left= random.randint(0, width  - w)
        mask[:, top:top + h, left:left + w] = 0
        masked += h * w

    return mask


def generate_irregular_mask(
    height: int,
    width:  int,
    max_ratio: float = 0.50,
) -> torch.Tensor:
    """
    Generate an irregular free-form mask (better simulates cloud occlusions).

    Appendix — Limitations and Discussion:
        "Cloud occlusions act as large-area masks with low-frequency textures,
         effectively managed by our adaptive blending strategy."

    Returns:
        mask : (1, H, W) float tensor where 1 = known region
    """
    import cv2 as _cv2  # optional heavy import

    mask = np.ones((height, width), dtype=np.float32)
    num_strokes = random.randint(3, 8)

    for _ in range(num_strokes):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        for _ in range(random.randint(5, 15)):
            angle    = random.uniform(0, 2 * np.pi)
            length   = random.randint(height // 10, height // 4)
            x2 = int(np.clip(x1 + length * np.cos(angle), 0, width  - 1))
            y2 = int(np.clip(y1 + length * np.sin(angle), 0, height - 1))
            thickness = random.randint(10, 40)
            _cv2.line(mask, (x1, y1), (x2, y2), 0, thickness)
            x1, y1 = x2, y2

    return torch.from_numpy(mask).unsqueeze(0)


# ---------------------------------------------------------------------------
# Base satellite dataset
# ---------------------------------------------------------------------------

class SatelliteInpaintingDataset(Dataset):
    """
    Generic satellite image inpainting dataset.

    Returns dict with:
        'image'     : (C, H, W) float in [-1, 1]
        'mask'      : (1, H, W) float in {0, 1}   (1 = known)
        'masked'    : (C, H, W) — image × mask (conditioned region)
        'filename'  : str
    """

    def __init__(
        self,
        image_dir:    str,
        image_size:   int  = 256,
        mask_ratio:   Tuple[float, float] = (0.30, 0.50),
        mask_type:    str  = "random",   # 'random' | 'irregular'
        augment:      bool = True,
        split:        str  = "train",    # 'train' | 'val' | 'test'
        val_ratio:    float = 0.10,
        seed:         int  = 42,
    ):
        super().__init__()
        self.image_size  = image_size
        self.mask_ratio  = mask_ratio
        self.mask_type   = mask_type

        # Collect image paths
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        all_paths = sorted([
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in exts
        ])
        if not all_paths:
            raise FileNotFoundError(f"No images found in: {image_dir}")

        # Train / val / test split (reproducible)
        rng     = random.Random(seed)
        indices = list(range(len(all_paths)))
        rng.shuffle(indices)
        n_val   = max(1, int(len(indices) * val_ratio))

        if split == "train":
            self.paths = [all_paths[i] for i in indices[n_val:]]
        elif split == "val":
            self.paths = [all_paths[i] for i in indices[:n_val]]
        else:  # test — use all
            self.paths = all_paths

        # Image transforms (Appendix — Data Augmentation and Masking Strategy)
        aug_list = [transforms.Resize((image_size, image_size))]
        if augment and split == "train":
            aug_list += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
            ]
        aug_list += [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # → [-1, 1]
        ]
        self.transform = transforms.Compose(aug_list)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path  = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)        # (3, H, W) in [-1, 1]

        H, W = self.image_size, self.image_size
        if self.mask_type == "irregular":
            try:
                mask = generate_irregular_mask(H, W)
            except ImportError:
                mask = generate_random_mask(H, W, *self.mask_ratio)
        else:
            mask = generate_random_mask(H, W, *self.mask_ratio)

        return {
            "image":    image,
            "mask":     mask,
            "masked":   image * mask,    # conditioned region x_cond
            "filename": path.name,
        }


# ---------------------------------------------------------------------------
# Massachusetts Roads Dataset  (Section IV-B)
# ---------------------------------------------------------------------------

class MassachusettsRoadsDataset(SatelliteInpaintingDataset):
    """
    Massachusetts Roads Dataset loader.

    From Section IV-B:
        "Contains 1171 aerial RGB images of 1500×1500 pixels each.
         Focuses on road structure recovery with sharp, thin features —
         ideal for testing spatial accuracy in inpainting."

    Reference:
        Mnih, V. "Machine Learning for Aerial Image Labeling."
        PhD Thesis, University of Toronto, 2013.

    Expected directory layout:
        massachusetts_roads/
            train/
                *.tiff
            val/
                *.tiff
            test/
                *.tiff
    """

    NAME = "Massachusetts Roads Dataset"

    def __init__(self, root: str, split: str = "train",
                 image_size: int = 256, **kwargs):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            split_dir = root   # fallback: flat directory
        super().__init__(
            image_dir  = split_dir,
            image_size = image_size,
            split      = split,
            **kwargs,
        )
        print(f"[{self.NAME}] split={split!r}  images={len(self)}")


# ---------------------------------------------------------------------------
# DeepGlobe 2018 Dataset  (Section IV-B)
# ---------------------------------------------------------------------------

class DeepGlobeDataset(SatelliteInpaintingDataset):
    """
    DeepGlobe 2018 Dataset loader.

    From Section IV-B:
        "Includes 803 VHR satellite images at 50 cm/pixel resolution.
         Diverse land cover types — ideal for testing generalizability
         and robustness of inpainting models."

    Reference:
        Demir et al., "DeepGlobe 2018: A Challenge to Parse the Earth
        Through Satellite Images," CVPR Workshops, 2018.

    Expected directory layout:
        deepglobe/
            images/
                *.jpg  (or *.png / *.tif)
    """

    NAME = "DeepGlobe 2018 Dataset"

    def __init__(self, root: str, split: str = "train",
                 image_size: int = 256, **kwargs):
        img_dir = os.path.join(root, "images")
        if not os.path.isdir(img_dir):
            img_dir = root
        super().__init__(
            image_dir  = img_dir,
            image_size = image_size,
            split      = split,
            **kwargs,
        )
        print(f"[{self.NAME}] split={split!r}  images={len(self)}")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset_name: str,
    root:         str,
    split:        str  = "train",
    image_size:   int  = 256,
    batch_size:   int  = 16,          # Appendix training setup: batch=16
    num_workers:  int  = 4,
    mask_type:    str  = "random",
    **kwargs,
) -> DataLoader:
    """
    Factory for KAO dataloaders.

    Batch size of 16 follows the training setup described in the Appendix:
        "Training is conducted with a batch size of 16 using a
         single NVIDIA A40 GPU."
    """
    dataset_cls = {
        "massachusetts": MassachusettsRoadsDataset,
        "deepglobe":     DeepGlobeDataset,
        "generic":       SatelliteInpaintingDataset,
    }.get(dataset_name.lower())

    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from: massachusetts, deepglobe, generic")

    dataset = dataset_cls(
        root       = root,
        split      = split,
        image_size = image_size,
        mask_type  = mask_type,
        augment    = (split == "train"),
        **kwargs,
    )

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (split == "train"),
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = (split == "train"),
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    from PIL import Image as PILImage

    # Create a tiny synthetic dataset for testing
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(8):
            img = PILImage.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(os.path.join(tmp, f"sat_{i:04d}.png"))

        ds = SatelliteInpaintingDataset(
            tmp, image_size=64, split="train", augment=True
        )
        batch = ds[0]
        print(f"[Datasets] image:  {tuple(batch['image'].shape)}")
        print(f"[Datasets] mask :  {tuple(batch['mask'].shape)}")
        print(f"[Datasets] masked: {tuple(batch['masked'].shape)}")
        print("[Datasets] Sanity check passed ✓")
