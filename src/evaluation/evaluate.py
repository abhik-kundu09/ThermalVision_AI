"""
Full-Reference Scientific Evaluation Suite for Landsat IR -> RGB Translation.

Evaluates trained Pix2Pix generator on held-out test scenes:
- PSNR (Peak Signal-to-Noise Ratio) against authentic ground-truth RGB
- SSIM (Structural Similarity Index) across visible color channels
- MAE (Mean Absolute Error)
- Reference-free metrics: Tenengrad sharpness, Shannon entropy, Edge Preservation Index (EPI)
- Exports visual triplet comparison panels [Input IR | Predicted RGB | Ground Truth RGB]
- Produces statistical evaluation_summary.json
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.generator import UNetGenerator
from src.data.dataset import LandsatPatchDataset
from backend.metrics import (
    calculate_tenengrad_sharpness,
    calculate_shannon_entropy,
    calculate_edge_preservation_index
)

logger = logging.getLogger("ps10.evaluation")


def tensor_to_uint8_image(tensor_neg1_1: np.ndarray) -> np.ndarray:
    """
    Converts a normalized float array in [-1.0, 1.0] (C, H, W) to (H, W, C) uint8 in [0, 255].
    """
    if tensor_neg1_1.ndim == 3:
        if tensor_neg1_1.shape[0] in (1, 3):
            arr = np.transpose(tensor_neg1_1, (1, 2, 0))
        else:
            arr = tensor_neg1_1
    elif tensor_neg1_1.ndim == 2:
        arr = np.expand_dims(tensor_neg1_1, axis=-1)
    else:
        arr = tensor_neg1_1

    # Map [-1.0, 1.0] -> [0.0, 1.0] -> [0, 255]
    norm_0_1 = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    uint8_img = (norm_0_1 * 255.0).astype(np.uint8)

    if uint8_img.shape[-1] == 1:
        uint8_img = np.squeeze(uint8_img, axis=-1)

    return uint8_img


def evaluate_model_on_test_set(
    checkpoint_path: str,
    data_dir: str = "data",
    output_dir: str = "outputs/evaluation",
    num_samples_to_save: int = 10,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs full evaluation over the test dataset split.

    Args:
        checkpoint_path: Path to generator_best.pth.
        data_dir: Base directory containing dataset.
        output_dir: Directory to save visual comparison images and metrics JSON.
        num_samples_to_save: Number of visual triplet comparison panels to export.
        device: 'cuda' or 'cpu'.

    Returns:
        summary_results: Dictionary containing aggregate benchmark metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1. Load Generator — infer num_filters from checkpoint to avoid shape mismatch
    num_filters = 64  # production default
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=dev, weights_only=True)
        # down1.block.0.weight shape: [num_filters, in_channels, kH, kW]
        if "down1.block.0.weight" in state_dict:
            num_filters = state_dict["down1.block.0.weight"].shape[0]
        net_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=num_filters)
        net_g.load_state_dict(state_dict)
        logger.info(f"Loaded generator weights from {checkpoint_path} (num_filters={num_filters})")
    else:
        net_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=num_filters)
        logger.warning(f"Checkpoint not found at {checkpoint_path}; evaluating initialized weights.")

    net_g.to(dev)
    net_g.eval()

    # 2. Load Test Dataset
    test_ds = LandsatPatchDataset(data_dir=data_dir, split="test")
    if len(test_ds) == 0:
        logger.warning("Test dataset is empty. Check data/test directory.")
        return {"error": "Test dataset is empty"}

    psnr_scores: List[float] = []
    ssim_scores: List[float] = []
    mae_scores: List[float] = []
    tenengrad_scores: List[float] = []
    entropy_scores: List[float] = []
    epi_scores: List[float] = []

    logger.info(f"Evaluating {len(test_ds)} test patches on {dev}...")

    with torch.no_grad():
        for idx in range(len(test_ds)):
            ir_tensor, rgb_gt_tensor = test_ds[idx]

            # Shape: [1, 1, 256, 256]
            if HAS_TORCH and isinstance(ir_tensor, torch.Tensor):
                ir_input = ir_tensor.unsqueeze(0).to(dev)
                pred_tensor = net_g(ir_input).squeeze(0).cpu().numpy()
                ir_np = ir_tensor.numpy()
                gt_np = rgb_gt_tensor.numpy()
            else:
                ir_np = ir_tensor
                gt_np = rgb_gt_tensor
                pred_tensor = np.zeros_like(gt_np)

            # Convert to uint8 images [0, 255]
            ir_img = tensor_to_uint8_image(ir_np)       # (256, 256) grayscale
            pred_rgb = tensor_to_uint8_image(pred_tensor) # (256, 256, 3) RGB
            gt_rgb = tensor_to_uint8_image(gt_np)        # (256, 256, 3) RGB

            # Compute Full-Reference Metrics (Predicted RGB vs Ground Truth RGB)
            psnr_val = float(compute_psnr(gt_rgb, pred_rgb, data_range=255))
            ssim_val = float(compute_ssim(gt_rgb, pred_rgb, channel_axis=2, data_range=255))
            mae_val = float(np.mean(np.abs(gt_rgb.astype(float) - pred_rgb.astype(float))))

            # Compute Reference-Free Metrics
            pred_gray = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2GRAY)
            tenengrad_val = calculate_tenengrad_sharpness(pred_gray)
            entropy_val = calculate_shannon_entropy(pred_gray)
            epi_val = calculate_edge_preservation_index(ir_img, pred_gray)

            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)
            mae_scores.append(mae_val)
            tenengrad_scores.append(tenengrad_val)
            entropy_scores.append(entropy_val)
            epi_scores.append(epi_val)

            # Export comparison visual panel
            if idx < num_samples_to_save:
                # Convert IR to 3-channel for side-by-side concatenation
                ir_bgr = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
                pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
                gt_bgr = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)

                # Error heat map
                diff_bgr = cv2.applyColorMap(
                    np.clip(np.abs(gt_rgb.astype(float) - pred_rgb.astype(float)).mean(axis=2) * 3, 0, 255).astype(np.uint8),
                    cv2.COLORMAP_JET
                )

                # Assemble 4-panel comparison: [Input IR | Predicted RGB | Ground Truth RGB | Error Map]
                panel = np.hstack([ir_bgr, pred_bgr, gt_bgr, diff_bgr])
                panel_path = os.path.join(output_dir, f"test_sample_{idx:04d}_psnr_{psnr_val:.1f}dB.png")
                cv2.imwrite(panel_path, panel)

    summary_results = {
        "num_test_samples": len(test_ds),
        "metrics": {
            "psnr_mean_db": round(float(np.mean(psnr_scores)), 2),
            "psnr_std_db": round(float(np.std(psnr_scores)), 2),
            "psnr_max_db": round(float(np.max(psnr_scores)), 2),
            "ssim_mean": round(float(np.mean(ssim_scores)), 4),
            "ssim_std": round(float(np.std(ssim_scores)), 4),
            "mae_mean_pixels": round(float(np.mean(mae_scores)), 2),
            "tenengrad_mean": round(float(np.mean(tenengrad_scores)), 2),
            "entropy_mean_bits": round(float(np.mean(entropy_scores)), 4),
            "epi_mean": round(float(np.mean(epi_scores)), 4)
        }
    }

    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {summary_file}")
    logger.info(
        f"Mean PSNR: {summary_results['metrics']['psnr_mean_db']} dB | "
        f"Mean SSIM: {summary_results['metrics']['ssim_mean']} | "
        f"Mean EPI: {summary_results['metrics']['epi_mean']}"
    )

    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pix2Pix Landsat IR -> RGB Translation Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/generator_best.pth", help="Generator checkpoint path")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Evaluation outputs directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of comparison panels to export")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    evaluate_model_on_test_set(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples_to_save=args.samples
    )
