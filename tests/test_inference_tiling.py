"""
Unit Tests for Full-Scene Tiled Sliding-Window Inference and Raster Stitching Engine.
"""

import os
import tempfile
import pytest
import numpy as np
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.generator import UNetGenerator
from src.inference.predict import (
    create_2d_blend_window,
    predict_thermal_array,
    predict_landsat_geotiff
)


def test_create_2d_blend_window():
    """Verify 2D Hann blending window properties."""
    win = create_2d_blend_window(patch_size=256)
    assert win.shape == (256, 256)
    assert win.dtype == np.float32
    assert win.min() > 0.0
    assert win.max() <= 1.0
    # Peak should be at the center
    center = win[128, 128]
    corner = win[0, 0]
    assert center > corner


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for inference tests")
def test_predict_thermal_array_arbitrary_dimensions():
    """Verify tiled sliding-window inference on non-square, arbitrary-sized images."""
    net_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=8)

    # Non-standard dimensions: 600 height x 450 width
    h, w = 600, 450
    thermal_in = np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32)

    pred_rgb = predict_thermal_array(
        thermal_normalized=thermal_in,
        net_g=net_g,
        patch_size=256,
        overlap=64,
        device="cpu"
    )

    assert pred_rgb.shape == (h, w, 3)
    assert pred_rgb.dtype == np.uint8
    assert pred_rgb.min() >= 0
    assert pred_rgb.max() <= 255


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for inference tests")
def test_predict_landsat_geotiff_end_to_end():
    """Verify end-to-end GeoTIFF prediction pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock input 16-bit thermal TIFF
        h, w = 300, 300
        mock_raw = np.random.randint(35000, 45000, (h, w), dtype=np.uint16)
        in_path = os.path.join(tmpdir, "test_ST_B10.tif")
        out_path = os.path.join(tmpdir, "predicted_output.png")
        ckpt_path = os.path.join(tmpdir, "generator_best.pth")

        cv2.imwrite(in_path, mock_raw)

        # Save mock generator checkpoint
        net_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=8)
        torch.save(net_g.state_dict(), ckpt_path)

        res_path = predict_landsat_geotiff(
            thermal_tiff_path=in_path,
            output_path=out_path,
            checkpoint_path=ckpt_path,
            device="cpu"
        )

        assert os.path.exists(res_path)
        out_img = cv2.imread(res_path)
        assert out_img.shape == (h, w, 3)
