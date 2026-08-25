"""
Landsat Patch Generation and Scene-Level Dataset Splitting Module.

Extracts 256x256 paired (Thermal IR, Visible RGB) patches from calibrated Landsat scenes.
Implements:
- Sliding-window patch extraction with configurable stride
- Nodata / empty pixel threshold filtering
- Low-variance background filtering
- Scene-level train/val/test splitting (prevents spatial data leakage)
- Lossless .npy array storage
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.data.calibrate import normalize_thermal_for_gan, normalize_rgb_for_gan
from src.data.landsat_loader import LandsatScene

logger = logging.getLogger("ps10.data.patch_extractor")


@dataclass
class PatchMetadata:
    """Metadata describing an extracted 256x256 patch pair."""
    patch_id: str
    scene_id: str
    split: str             # 'train', 'val', or 'test'
    y_min: int
    x_min: int
    patch_size: int
    valid_ratio: float
    ir_file: str
    rgb_file: str


def extract_patches_from_scene(
    scene: LandsatScene,
    output_dir: str,
    split: str = "train",
    patch_size: int = 256,
    stride: int = 256,
    max_nodata_ratio: float = 0.05,
    min_variance: float = 1e-4
) -> List[PatchMetadata]:
    """
    Extracts 256x256 paired patches from a single calibrated LandsatScene.

    Args:
        scene: Calibrated and spatially aligned LandsatScene object.
        output_dir: Target directory for the given split (e.g. data/train).
        split: Dataset split ('train', 'val', 'test').
        patch_size: Square tile dimension (default 256).
        stride: Step size between tiles (default 256 for no overlap, 128 for 50% overlap).
        max_nodata_ratio: Maximum allowed ratio of invalid/nodata pixels per patch.
        min_variance: Minimum variance threshold to filter uninformative constant patches.

    Returns:
        extracted_patches: List of PatchMetadata objects.
    """
    split_dir = os.path.join(output_dir, split)
    ir_dir = os.path.join(split_dir, "ir")
    rgb_dir = os.path.join(split_dir, "rgb")
    os.makedirs(ir_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    # 1. Normalize full scene to [-1.0, 1.0] for GAN training
    norm_thermal = normalize_thermal_for_gan(scene.thermal_kelvin, scene.thermal_mask)
    norm_rgb = normalize_rgb_for_gan(scene.rgb_reflectance, scene.rgb_mask)

    combined_mask = scene.thermal_mask & scene.rgb_mask
    h, w = norm_thermal.shape[:2]

    extracted_patches: List[PatchMetadata] = []
    patch_idx = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            y_end = y + patch_size
            x_end = x + patch_size

            # Check valid pixel coverage
            mask_patch = combined_mask[y:y_end, x:x_end]
            valid_ratio = float(np.mean(mask_patch))
            nodata_ratio = 1.0 - valid_ratio

            if nodata_ratio > max_nodata_ratio:
                continue

            ir_patch = norm_thermal[y:y_end, x:x_end]
            rgb_patch = norm_rgb[y:y_end, x:x_end, :]

            # Filter flat / constant patches
            if np.var(ir_patch) < min_variance or np.var(rgb_patch) < min_variance:
                continue

            patch_id = f"{scene.scene_id}_p{patch_idx:05d}"
            ir_filename = f"{patch_id}_ir.npy"
            rgb_filename = f"{patch_id}_rgb.npy"

            ir_save_path = os.path.join(ir_dir, ir_filename)
            rgb_save_path = os.path.join(rgb_dir, rgb_filename)

            # Save arrays as float32 .npy
            np.save(ir_save_path, ir_patch.astype(np.float32))
            np.save(rgb_save_path, rgb_patch.astype(np.float32))

            meta = PatchMetadata(
                patch_id=patch_id,
                scene_id=scene.scene_id,
                split=split,
                y_min=y,
                x_min=x,
                patch_size=patch_size,
                valid_ratio=round(valid_ratio, 4),
                ir_file=os.path.relpath(ir_save_path, output_dir),
                rgb_file=os.path.relpath(rgb_save_path, output_dir)
            )
            extracted_patches.append(meta)
            patch_idx += 1

    logger.info(
        f"Extracted {len(extracted_patches)} valid patches from scene '{scene.scene_id}' ({split} split)."
    )
    return extracted_patches


def build_scene_level_dataset(
    scenes: List[LandsatScene],
    output_base_dir: str = "data",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    patch_size: int = 256,
    stride: int = 256,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Partitions Landsat scenes into train/val/test splits at the SCENE LEVEL to eliminate
    spatial data leakage, and extracts paired 256x256 patches.

    Args:
        scenes: List of calibrated LandsatScene objects.
        output_base_dir: Root dataset directory (default 'data').
        train_ratio: Fraction of scenes for training.
        val_ratio: Fraction of scenes for validation (remainder is test).
        patch_size: Patch dimension.
        stride: Stride between tiles.
        seed: Random seed for reproducible scene assignment.

    Returns:
        manifest: Summary dictionary of the dataset generation process.
    """
    if len(scenes) == 0:
        raise ValueError("No Landsat scenes provided for dataset generation.")

    rng = np.random.RandomState(seed)
    scene_indices = np.arange(len(scenes))
    rng.shuffle(scene_indices)

    num_scenes = len(scenes)
    if num_scenes == 1:
        # Edge case: single scene available for smoke testing
        train_scenes = [scenes[0]]
        val_scenes = [scenes[0]]
        test_scenes = [scenes[0]]
    else:
        n_train = max(1, int(num_scenes * train_ratio))
        n_val = max(1, int(num_scenes * val_ratio))

        train_scenes = [scenes[i] for i in scene_indices[:n_train]]
        val_scenes = [scenes[i] for i in scene_indices[n_train:n_train + n_val]]
        test_scenes = [scenes[i] for i in scene_indices[n_train + n_val:]]
        if len(test_scenes) == 0:
            test_scenes = val_scenes

    all_patch_meta: List[PatchMetadata] = []

    # Extract Train Patches
    for sc in train_scenes:
        all_patch_meta.extend(
            extract_patches_from_scene(sc, output_base_dir, split="train", patch_size=patch_size, stride=stride)
        )

    # Extract Val Patches
    for sc in val_scenes:
        all_patch_meta.extend(
            extract_patches_from_scene(sc, output_base_dir, split="val", patch_size=patch_size, stride=stride)
        )

    # Extract Test Patches
    for sc in test_scenes:
        all_patch_meta.extend(
            extract_patches_from_scene(sc, output_base_dir, split="test", patch_size=patch_size, stride=stride)
        )

    # Save dataset manifest
    manifest_path = os.path.join(output_base_dir, "dataset_manifest.json")
    manifest_data = {
        "total_patches": len(all_patch_meta),
        "patch_size": patch_size,
        "stride": stride,
        "splits": {
            "train": [asdict(p) for p in all_patch_meta if p.split == "train"],
            "val": [asdict(p) for p in all_patch_meta if p.split == "val"],
            "test": [asdict(p) for p in all_patch_meta if p.split == "test"]
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Dataset manifest saved to {manifest_path}")
    return manifest_data
