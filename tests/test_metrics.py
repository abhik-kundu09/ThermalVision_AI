"""
Unit Tests for Thermal IR Image Quality and Performance Metrics.
Validates scientific fidelity of reference-free metrics and ground-truth enforcement.
"""

import pytest
import numpy as np
import cv2
from backend.metrics import (
    calculate_shannon_entropy,
    calculate_tenengrad_sharpness,
    calculate_rms_contrast,
    calculate_edge_preservation_index,
    calculate_full_reference_metrics,
    compute_all_metrics
)
from backend.schemas import LatencyBreakdown


def test_shannon_entropy_calculation():
    """Test entropy calculation on uniform vs random distributions."""
    # Uniform image (constant color) -> Entropy should be 0.0
    flat_img = np.full((100, 100), 128, dtype=np.uint8)
    entropy_flat = calculate_shannon_entropy(flat_img)
    assert entropy_flat == 0.0

    # Random noise image -> High entropy (~8.0 bits for 256 uniform levels)
    noisy_img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    entropy_noisy = calculate_shannon_entropy(noisy_img)
    assert 7.5 < entropy_noisy <= 8.0


def test_tenengrad_sharpness():
    """Test Tenengrad gradient index responds proportionally to edge sharpness."""
    # Blurred smooth circle
    blurred = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(blurred, (50, 50), 30, 255, -1)
    blurred = cv2.GaussianBlur(blurred, (21, 21), 10)

    # Sharp circle
    sharp = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(sharp, (50, 50), 30, 255, -1)

    t_blurred = calculate_tenengrad_sharpness(blurred)
    t_sharp = calculate_tenengrad_sharpness(sharp)

    assert t_sharp > t_blurred


def test_edge_preservation_index():
    """Test Edge Preservation Index (EPI) between identical and corrupted images."""
    base = np.random.randint(50, 200, (100, 100), dtype=np.uint8)

    # Identical image -> EPI should be 1.0
    epi_identical = calculate_edge_preservation_index(base, base)
    assert pytest.approx(epi_identical, abs=0.01) == 1.0

    # Heavily blurred image -> EPI should be significantly lower
    heavily_blurred = cv2.GaussianBlur(base, (25, 25), 10)
    epi_blurred = calculate_edge_preservation_index(base, heavily_blurred)
    assert epi_blurred < 0.6


def test_full_reference_metrics_without_ground_truth():
    """Test that missing ground truth returns None with explicit scientific disclaimer."""
    pred_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    psnr, ssim, has_gt, disclaimer = calculate_full_reference_metrics(pred_bgr, None)

    assert psnr is None
    assert ssim is None
    assert has_gt is False
    assert disclaimer is not None
    assert "cross-spectral" in disclaimer.lower() or "mathematically invalid" in disclaimer.lower()


def test_full_reference_metrics_with_ground_truth():
    """Test PSNR/SSIM calculation when ground truth is provided."""
    gt_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Identical ground truth: PSNR should be infinite (or capped high), SSIM should be 1.0
    psnr_id, ssim_id, has_gt, _ = calculate_full_reference_metrics(gt_bgr, gt_bgr.copy())
    assert has_gt is True
    assert ssim_id == 1.0
    assert psnr_id > 80.0 or psnr_id == float('inf') or psnr_id >= 50.0

    # Slightly perturbed ground truth
    noise = np.random.normal(0, 5, (100, 100, 3)).astype(np.int16)
    noisy_pred = np.clip(gt_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    psnr_noisy, ssim_noisy, _, _ = calculate_full_reference_metrics(noisy_pred, gt_bgr)
    assert 20.0 < psnr_noisy < 60.0
    assert 0.7 < ssim_noisy < 1.0
