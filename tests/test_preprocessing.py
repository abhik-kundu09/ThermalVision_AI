"""
Unit Tests for Local Thermal IR Image Preprocessing Module.
"""

import pytest
import numpy as np
import cv2
from backend.preprocessing import (
    decode_image_bytes,
    normalize_thermal_to_grayscale_8bit,
    apply_clahe,
    apply_bilateral_filter,
    preprocess_thermal_image,
    PreprocessingError
)


def create_dummy_png_bytes(shape=(120, 160), channels=1, dtype=np.uint8):
    """Helper to generate encoded PNG bytes for testing."""
    if channels == 1:
        img = np.random.randint(0, 255, shape, dtype=dtype)
    elif channels == 3:
        img = np.random.randint(0, 255, (*shape, 3), dtype=dtype)
    elif channels == 4:
        img = np.random.randint(0, 255, (*shape, 4), dtype=dtype)
    else:
        raise ValueError("Unsupported channels")
    
    success, buffer = cv2.imencode(".png", img)
    assert success
    return buffer.tobytes()


def test_decode_image_bytes_valid():
    """Test decoding valid image bytes."""
    valid_bytes = create_dummy_png_bytes((100, 100))
    decoded = decode_image_bytes(valid_bytes)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (100, 100)


def test_decode_image_bytes_empty_and_corrupt():
    """Test decoding empty and corrupted byte arrays."""
    with pytest.raises(PreprocessingError, match="Empty image byte buffer"):
        decode_image_bytes(b"")

    with pytest.raises(PreprocessingError, match="Failed to decode image data"):
        decode_image_bytes(b"corrupted_non_image_bytes_12345")


def test_normalize_thermal_16bit_to_8bit():
    """Test min-max normalization on 16-bit FLIR sensor data."""
    # 16-bit raw array ranging from 1000 to 50000 counts
    raw_16bit = np.linspace(1000, 50000, 10000, dtype=np.uint16).reshape((100, 100))
    normalized, depth_info = normalize_thermal_to_grayscale_8bit(raw_16bit)

    assert normalized.dtype == np.uint8
    assert normalized.shape == (100, 100)
    assert normalized.min() == 0
    assert normalized.max() == 255
    assert "uint16" in depth_info


def test_normalize_thermal_rgb_and_rgba():
    """Test multi-channel conversion to grayscale."""
    # 3-channel RGB
    rgb = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    gray_from_rgb, info_rgb = normalize_thermal_to_grayscale_8bit(rgb)
    assert gray_from_rgb.shape == (80, 80)
    assert gray_from_rgb.dtype == np.uint8

    # 4-channel RGBA
    rgba = np.random.randint(0, 255, (80, 80, 4), dtype=np.uint8)
    gray_from_rgba, info_rgba = normalize_thermal_to_grayscale_8bit(rgba)
    assert gray_from_rgba.shape == (80, 80)
    assert gray_from_rgba.dtype == np.uint8


def test_apply_clahe_contrast_adaptation():
    """Test that CLAHE increases localized contrast without exploding dynamic range."""
    # Create low-contrast image (narrow dynamic range 100-140)
    low_contrast = np.random.randint(100, 140, (100, 100), dtype=np.uint8)
    enhanced = apply_clahe(low_contrast, clip_limit=3.0, grid_size=8)

    assert enhanced.shape == (100, 100)
    assert enhanced.dtype == np.uint8
    # Standard deviation of enhanced should be greater than low contrast input
    assert np.std(enhanced) > np.std(low_contrast)


def test_apply_bilateral_filter_denoising():
    """Test bilateral filtering on noisy step edge."""
    # Step edge image with high-frequency noise
    step = np.zeros((100, 100), dtype=np.uint8)
    step[:, 50:] = 200
    noise = np.random.normal(0, 15, (100, 100)).astype(np.int16)
    noisy_step = np.clip(step.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    denoised = apply_bilateral_filter(noisy_step, d=9, sigma_color=75, sigma_space=75)
    assert denoised.shape == (100, 100)
    assert denoised.dtype == np.uint8
    # Noise variance should decrease
    assert np.var(denoised[:40, :40]) < np.var(noisy_step[:40, :40])


def test_preprocess_thermal_image_full_pipeline():
    """Test end-to-end preprocessing pipeline execution."""
    raw_bytes = create_dummy_png_bytes((120, 160), channels=1)
    raw_gray, prep_gray, prep_bgr, depth_info, dims = preprocess_thermal_image(
        raw_bytes,
        clahe_clip_limit=2.0,
        clahe_grid_size=8,
        bilateral_d=9,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0
    )

    assert raw_gray.shape == (120, 160)
    assert prep_gray.shape == (120, 160)
    assert prep_bgr.shape == (120, 160, 3)
    assert dims == (120, 160)
