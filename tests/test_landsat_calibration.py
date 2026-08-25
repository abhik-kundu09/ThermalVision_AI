"""
Unit Tests for Landsat Collection 2 Radiometric Calibration and Scene Loading.
"""

import os
import tempfile
import pytest
import numpy as np
import cv2

from src.data.calibrate import (
    calibrate_surface_temperature,
    calibrate_surface_reflectance,
    kelvin_to_celsius,
    normalize_thermal_for_gan,
    normalize_rgb_for_gan,
    LANDSAT_ST_SCALE,
    LANDSAT_ST_OFFSET,
    LANDSAT_SR_SCALE,
    LANDSAT_SR_OFFSET
)
from src.data.landsat_loader import (
    load_landsat_scene,
    read_band_geotiff,
    GeospatialMetadata
)


def test_surface_temperature_calibration_formula():
    """Verify USGS formula: Kelvin = DN * 0.00341802 + 149.0."""
    # Test typical land temperature DN (~40,000)
    raw_dn = np.array([[40000, 45000], [35000, 0]], dtype=np.uint16)
    kelvin, mask = calibrate_surface_temperature(raw_dn, nodata_val=0)

    # Expected for 40000: 40000 * 0.00341802 + 149.0 = 285.7208 K
    expected_40k = 40000 * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET
    assert pytest.approx(kelvin[0, 0], abs=1e-3) == expected_40k
    assert mask[0, 0] is np.True_ or mask[0, 0] == True

    # Nodata check (DN = 0)
    assert np.isnan(kelvin[1, 1])
    assert mask[1, 1] is np.False_ or mask[1, 1] == False


def test_kelvin_to_celsius_conversion():
    """Verify Kelvin to Celsius temperature conversion."""
    kelvin = np.array([273.15, 300.0, 310.15], dtype=np.float32)
    celsius = kelvin_to_celsius(kelvin)

    assert pytest.approx(celsius[0], abs=1e-3) == 0.0
    assert pytest.approx(celsius[1], abs=1e-3) == 26.85
    assert pytest.approx(celsius[2], abs=1e-3) == 37.0


def test_surface_reflectance_calibration_formula():
    """Verify USGS formula: Reflectance = DN * 0.0000275 - 0.2."""
    raw_dn = np.array([[15000, 20000], [8000, 0]], dtype=np.uint16)
    refl, mask = calibrate_surface_reflectance(raw_dn, nodata_val=0)

    # Expected for 15000: 15000 * 0.0000275 - 0.2 = 0.2125
    expected_15k = 15000 * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
    assert pytest.approx(refl[0, 0], abs=1e-4) == expected_15k
    assert mask[0, 0] is np.True_ or mask[0, 0] == True

    # Atmospheric overcorrection / negative clipping
    # For DN = 7000: 7000 * 0.0000275 - 0.2 = -0.0075 -> clipped to 0.0
    raw_low = np.array([[7000]], dtype=np.uint16)
    refl_low, mask_low = calibrate_surface_reflectance(raw_low, nodata_val=0)
    assert refl_low[0, 0] == 0.0

    # Nodata check
    assert np.isnan(refl[1, 1])
    assert mask[1, 1] is np.False_ or mask[1, 1] == False


def test_normalize_thermal_for_gan():
    """Verify thermal normalization maps to [-1.0, 1.0] for Pix2Pix."""
    # Min temp 275K, max temp 325K
    kelvin = np.array([[275.0, 300.0, 325.0], [np.nan, 200.0, 400.0]], dtype=np.float32)
    mask = np.array([[True, True, True], [False, True, True]], dtype=bool)

    norm = normalize_thermal_for_gan(kelvin, mask, min_temp=275.0, max_temp=325.0)

    # 275K -> -1.0
    assert pytest.approx(norm[0, 0], abs=1e-3) == -1.0
    # 300K -> 0.0 (midpoint)
    assert pytest.approx(norm[0, 1], abs=1e-3) == 0.0
    # 325K -> +1.0
    assert pytest.approx(norm[0, 2], abs=1e-3) == 1.0
    # Invalid / nodata -> -1.0
    assert norm[1, 0] == -1.0
    # Out of bounds clipped
    assert norm[1, 1] == -1.0
    assert norm[1, 2] == 1.0


def test_normalize_rgb_for_gan():
    """Verify RGB reflectance normalization maps [0, max_reflectance] -> [-1, 1]."""
    refl = np.zeros((10, 10, 3), dtype=np.float32)
    refl[:, :, :] = 0.15  # Midpoint of [0.0, 0.30]
    mask = np.ones((10, 10), dtype=bool)

    norm_rgb = normalize_rgb_for_gan(refl, mask, max_reflectance=0.30)

    assert norm_rgb.shape == (10, 10, 3)
    # 0.15 / 0.30 * 2 - 1 = 0.0
    assert pytest.approx(norm_rgb[0, 0, 0], abs=1e-3) == 0.0
    assert pytest.approx(norm_rgb.min(), abs=1e-3) == 0.0
    assert pytest.approx(norm_rgb.max(), abs=1e-3) == 0.0



def test_load_landsat_scene_mock():
    """Verify end-to-end Landsat scene loading on synthetic multi-band TIFFs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock 16-bit TIFF files for ST_B10, SR_B4, SR_B3, SR_B2
        h, w = 64, 64
        st_data = np.random.randint(35000, 45000, (h, w), dtype=np.uint16)
        b4_data = np.random.randint(10000, 25000, (h, w), dtype=np.uint16)
        b3_data = np.random.randint(10000, 25000, (h, w), dtype=np.uint16)
        b2_data = np.random.randint(10000, 25000, (h, w), dtype=np.uint16)

        st_path = os.path.join(tmpdir, "LC08_ST_B10.TIF")
        b4_path = os.path.join(tmpdir, "LC08_SR_B4.TIF")
        b3_path = os.path.join(tmpdir, "LC08_SR_B3.TIF")
        b2_path = os.path.join(tmpdir, "LC08_SR_B2.TIF")

        cv2.imwrite(st_path, st_data)
        cv2.imwrite(b4_path, b4_data)
        cv2.imwrite(b3_path, b3_data)
        cv2.imwrite(b2_path, b2_data)

        scene = load_landsat_scene(st_path, b4_path, b3_path, b2_path, scene_id="TEST_LC08")

        assert scene.scene_id == "TEST_LC08"
        assert scene.thermal_kelvin.shape == (h, w)
        assert scene.rgb_reflectance.shape == (h, w, 3)
        assert np.all(scene.thermal_kelvin >= 200.0)
        assert np.all(scene.rgb_reflectance >= 0.0)
        assert np.all(scene.rgb_reflectance <= 1.0)
