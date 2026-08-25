"""
Unit Tests for Full-Reference Scientific Evaluation Suite.
"""

import os
import tempfile
import json
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.evaluation.evaluate import (
    tensor_to_uint8_image,
    evaluate_model_on_test_set
)
from src.models.generator import UNetGenerator


def test_tensor_to_uint8_image_conversion():
    """Verify tensor conversion from [-1.0, 1.0] float to [0, 255] uint8."""
    # 3-channel RGB tensor in [-1.0, 1.0]
    tensor = np.array([
        [[-1.0, 0.0], [1.0, -1.0]],
        [[0.0, 1.0], [-1.0, 0.0]],
        [[1.0, -1.0], [0.0, 1.0]]
    ], dtype=np.float32)  # Shape: (3, 2, 2)

    uint8_img = tensor_to_uint8_image(tensor)

    assert uint8_img.shape == (2, 2, 3)
    assert uint8_img.dtype == np.uint8
    assert uint8_img[0, 0, 0] == 0       # -1.0 -> 0
    assert uint8_img[0, 1, 0] == 127     # 0.0 -> 127
    assert uint8_img[1, 0, 0] == 255     # 1.0 -> 255


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for evaluation tests")
def test_evaluate_model_on_test_set_mock():
    """Verify evaluation loop produces valid metrics and summary JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup mock test dataset
        test_dir = os.path.join(tmpdir, "data", "test")
        ir_dir = os.path.join(test_dir, "ir")
        rgb_dir = os.path.join(test_dir, "rgb")
        os.makedirs(ir_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)

        for i in range(3):
            ir_arr = np.random.uniform(-1.0, 1.0, (256, 256)).astype(np.float32)
            rgb_arr = np.random.uniform(-1.0, 1.0, (256, 256, 3)).astype(np.float32)
            np.save(os.path.join(ir_dir, f"patch_{i:04d}_ir.npy"), ir_arr)
            np.save(os.path.join(rgb_dir, f"patch_{i:04d}_rgb.npy"), rgb_arr)

        # 2. Save mock generator checkpoint
        ckpt_path = os.path.join(tmpdir, "generator_test.pth")
        net_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=8)
        torch.save(net_g.state_dict(), ckpt_path)

        # 3. Run evaluation
        out_eval_dir = os.path.join(tmpdir, "outputs", "evaluation")
        results = evaluate_model_on_test_set(
            checkpoint_path=ckpt_path,
            data_dir=os.path.join(tmpdir, "data"),
            output_dir=out_eval_dir,
            num_samples_to_save=2
        )

        assert results["num_test_samples"] == 3
        metrics = results["metrics"]
        assert "psnr_mean_db" in metrics
        assert "ssim_mean" in metrics
        assert "mae_mean_pixels" in metrics
        assert "tenengrad_mean" in metrics
        assert "entropy_mean_bits" in metrics
        assert "epi_mean" in metrics

        summary_file = os.path.join(out_eval_dir, "evaluation_summary.json")
        assert os.path.exists(summary_file)
