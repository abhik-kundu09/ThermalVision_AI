"""
Unit Tests for Landsat Geospatial Raster Alignment and Reprojection Module.
"""

import pytest
import numpy as np

from src.data.landsat_loader import GeospatialMetadata, LandsatScene
from src.data.align import (
    check_spatial_alignment,
    reproject_raster_to_reference,
    align_landsat_scene
)


def create_mock_metadata(
    crs: str = "EPSG:32645",
    width: int = 100,
    height: int = 100,
    res: tuple = (30.0, 30.0),
    transform: tuple = (30.0, 0.0, 500000.0, 0.0, -30.0, 3000000.0)
) -> GeospatialMetadata:
    return GeospatialMetadata(
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        nodata=0.0,
        resolution=res,
        bounds=(500000.0, 3000000.0 - height * 30.0, 500000.0 + width * 30.0, 3000000.0)
    )


def test_check_spatial_alignment_identical():
    """Identical metadata should report aligned."""
    meta1 = create_mock_metadata()
    meta2 = create_mock_metadata()

    is_aligned, diffs = check_spatial_alignment(meta1, meta2)
    assert is_aligned is True
    assert len(diffs) == 0


def test_check_spatial_alignment_mismatch():
    """Mismatched CRS or dimensions should report misalignment."""
    meta_base = create_mock_metadata(crs="EPSG:32645", width=100, height=100)
    meta_diff_crs = create_mock_metadata(crs="EPSG:4326", width=100, height=100)
    meta_diff_dim = create_mock_metadata(crs="EPSG:32645", width=120, height=100)

    aligned_crs, diff_crs = check_spatial_alignment(meta_base, meta_diff_crs)
    assert aligned_crs is False
    assert "crs_mismatch" in diff_crs

    aligned_dim, diff_dim = check_spatial_alignment(meta_base, meta_diff_dim)
    assert aligned_dim is False
    assert "dimension_mismatch" in diff_dim


def test_reproject_raster_to_reference():
    """Verify resampling from one grid to another."""
    src_meta = create_mock_metadata(width=50, height=50, res=(60.0, 60.0))
    dst_meta = create_mock_metadata(width=100, height=100, res=(30.0, 30.0))

    src_arr = np.full((50, 50), 290.0, dtype=np.float32)
    dst_arr = reproject_raster_to_reference(src_arr, src_meta, dst_meta, resampling_method="bilinear")

    assert dst_arr.shape == (100, 100)
    assert pytest.approx(dst_arr[50, 50], abs=1e-2) == 290.0


def test_align_landsat_scene_end_to_end():
    """Verify alignment of a scene with different thermal vs RGB dimensions."""
    # Thermal at (40x40), RGB at (80x80)
    thermal_raw = np.full((40, 40), 40000, dtype=np.uint16)
    thermal_kelvin = np.full((40, 40), 285.7, dtype=np.float32)
    thermal_mask = np.ones((40, 40), dtype=bool)

    rgb_raw = np.full((80, 80, 3), 15000, dtype=np.uint16)
    rgb_refl = np.full((80, 80, 3), 0.21, dtype=np.float32)
    rgb_mask = np.ones((80, 80), dtype=bool)

    meta = create_mock_metadata(width=80, height=80)

    scene = LandsatScene(
        scene_id="ALIGN_TEST_01",
        thermal_raw=thermal_raw,
        thermal_kelvin=thermal_kelvin,
        thermal_mask=thermal_mask,
        rgb_raw=rgb_raw,
        rgb_reflectance=rgb_refl,
        rgb_mask=rgb_mask,
        metadata=meta
    )

    aligned_scene = align_landsat_scene(scene)

    assert aligned_scene.thermal_kelvin.shape == (80, 80)
    assert aligned_scene.rgb_reflectance.shape == (80, 80, 3)
    assert aligned_scene.thermal_mask.shape == (80, 80)
    assert pytest.approx(aligned_scene.thermal_kelvin[40, 40], abs=0.5) == 285.7
