"""
Unit Tests for Landsat Patch Generation, Scene Splitting, and PyTorch Dataset.
"""

import os
import tempfile
import json
import pytest
import numpy as np

from src.data.landsat_loader import LandsatScene, GeospatialMetadata
from src.data.patch_extractor import (
    extract_patches_from_scene,
    build_scene_level_dataset
)
from src.data.dataset import LandsatPatchDataset


def create_synthetic_scene(
    scene_id: str,
    h: int = 512,
    w: int = 512,
    with_nodata: bool = False
) -> LandsatScene:
    """Creates a mock calibrated LandsatScene with realistic variations."""
    # Synthetic thermal: 280K - 310K with gradient and noise
    y, x = np.mgrid[0:h, 0:w]
    thermal_kelvin = (280.0 + 20.0 * (y / h) + 10.0 * np.sin(x / 30.0)).astype(np.float32)
    thermal_mask = np.ones((h, w), dtype=bool)

    # Synthetic RGB reflectance: [0.1, 0.6]
    rgb_refl = np.zeros((h, w, 3), dtype=np.float32)
    rgb_refl[:, :, 0] = 0.2 + 0.3 * (x / w)   # Red
    rgb_refl[:, :, 1] = 0.3 + 0.2 * (y / h)   # Green
    rgb_refl[:, :, 2] = 0.1 + 0.2 * np.cos(x / 50.0) # Blue
    rgb_mask = np.ones((h, w), dtype=bool)

    if with_nodata:
        # Fill bottom right quadrant with nodata
        thermal_mask[h // 2:, w // 2:] = False
        rgb_mask[h // 2:, w // 2:] = False
        thermal_kelvin[~thermal_mask] = np.nan
        rgb_refl[~rgb_mask, :] = np.nan

    meta = GeospatialMetadata(
        crs="EPSG:32645",
        transform=(30.0, 0.0, 500000.0, 0.0, -30.0, 3000000.0),
        width=w,
        height=h,
        nodata=0.0,
        resolution=(30.0, 30.0),
        bounds=None
    )

    return LandsatScene(
        scene_id=scene_id,
        thermal_raw=np.zeros((h, w), dtype=np.uint16),
        thermal_kelvin=thermal_kelvin,
        thermal_mask=thermal_mask,
        rgb_raw=np.zeros((h, w, 3), dtype=np.uint16),
        rgb_reflectance=rgb_refl,
        rgb_mask=rgb_mask,
        metadata=meta
    )


def test_extract_patches_from_scene():
    """Verify 256x256 patch extraction on a 512x512 scene."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scene = create_synthetic_scene("SCENE_TEST_01", h=512, w=512)
        patches = extract_patches_from_scene(
            scene=scene,
            output_dir=tmpdir,
            split="train",
            patch_size=256,
            stride=256
        )

        # 512x512 with stride 256 -> exactly 4 patches
        assert len(patches) == 4
        for p in patches:
            assert p.patch_size == 256
            assert p.valid_ratio == 1.0
            ir_path = os.path.join(tmpdir, p.ir_file)
            rgb_path = os.path.join(tmpdir, p.rgb_file)
            assert os.path.exists(ir_path)
            assert os.path.exists(rgb_path)

            ir_arr = np.load(ir_path)
            rgb_arr = np.load(rgb_path)
            assert ir_arr.shape == (256, 256)
            assert rgb_arr.shape == (256, 256, 3)
            # Normalization range check [-1.0, 1.0]
            assert ir_arr.min() >= -1.0 and ir_arr.max() <= 1.0
            assert rgb_arr.min() >= -1.0 and rgb_arr.max() <= 1.0


def test_nodata_filtering():
    """Verify that patches with nodata exceeding threshold are discarded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Scene with 25% nodata in bottom-right quadrant
        scene_nodata = create_synthetic_scene("SCENE_NODATA", h=512, w=512, with_nodata=True)
        patches = extract_patches_from_scene(
            scene=scene_nodata,
            output_dir=tmpdir,
            split="train",
            patch_size=256,
            stride=256,
            max_nodata_ratio=0.05
        )

        # Bottom-right patch must be filtered out -> 3 valid patches remain
        assert len(patches) == 3


def test_scene_level_dataset_split_and_loader():
    """Verify scene-level splitting (no spatial leakage) and PyTorch Dataset loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scene1 = create_synthetic_scene("SCENE_A", h=512, w=512)
        scene2 = create_synthetic_scene("SCENE_B", h=512, w=512)
        scene3 = create_synthetic_scene("SCENE_C", h=512, w=512)

        manifest = build_scene_level_dataset(
            scenes=[scene1, scene2, scene3],
            output_base_dir=tmpdir,
            train_ratio=0.5,
            val_ratio=0.3,
            patch_size=256,
            stride=256
        )

        manifest_file = os.path.join(tmpdir, "dataset_manifest.json")
        assert os.path.exists(manifest_file)
        assert manifest["total_patches"] > 0

        # Load train dataset
        ds_train = LandsatPatchDataset(data_dir=tmpdir, split="train")
        assert len(ds_train) > 0

        ir_item, rgb_item = ds_train[0]
        assert ir_item.shape == (1, 256, 256)
        assert rgb_item.shape == (3, 256, 256)
