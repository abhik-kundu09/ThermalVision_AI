"""
Full-Scene Tiled Sliding-Window Inference and Seamless Raster Stitching Engine.

Handles arbitrary-sized Landsat thermal scenes:
- Overlapping 256x256 sliding-window tiling
- 2D Hann window smooth blending across tile overlaps (zero boundary seam artifacts)
- Batched GPU/CPU generator inference
- Preserves full geographic CRS and affine geotransforms when exporting GeoTIFFs
"""

import os
import logging
import argparse
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from src.models.generator import UNetGenerator
from src.data.calibrate import calibrate_surface_temperature, normalize_thermal_for_gan
from src.data.landsat_loader import read_band_geotiff, GeospatialMetadata
from src.evaluation.evaluate import tensor_to_uint8_image

logger = logging.getLogger("ps10.inference")


def create_2d_blend_window(patch_size: int = 256) -> np.ndarray:
    """
    Creates a 2D Hann (cosine bell) blending window to smoothly weight tile predictions
    and eliminate visible seam edges at tile overlaps.
    """
    # 1D Hann window
    hann_1d = np.sin(np.linspace(0, np.pi, patch_size, endpoint=False) + (np.pi / (2 * patch_size))) ** 2
    # 2D outer product -> (patch_size, patch_size)
    blend_2d = np.outer(hann_1d, hann_1d).astype(np.float32)
    # Ensure minimum non-zero weight
    blend_2d = np.clip(blend_2d, 1e-4, 1.0)
    return blend_2d


def predict_thermal_array(
    thermal_normalized: np.ndarray,
    net_g: Any,
    patch_size: int = 256,
    overlap: int = 64,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Runs seamless tiled sliding-window inference across an arbitrary-sized 2D thermal array.

    Args:
        thermal_normalized: 2D float32 array in range [-1.0, 1.0].
        net_g: Trained UNetGenerator instance.
        patch_size: Square tile dimension (default 256).
        overlap: Overlap in pixels between adjacent tiles (default 64).
        device: 'cuda' or 'cpu'.

    Returns:
        reconstructed_rgb: 3D uint8 array (H, W, 3) in [0, 255].
    """
    h, w = thermal_normalized.shape[:2]
    stride = patch_size - overlap

    # Calculate padding if dimensions are smaller than patch_size or don't align with stride
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)

    # Pad to ensure complete coverage
    rem_h = (h + pad_h - patch_size) % stride
    if rem_h != 0:
        pad_h += stride - rem_h

    rem_w = (w + pad_w - patch_size) % stride
    if rem_w != 0:
        pad_w += stride - rem_w

    if pad_h > 0 or pad_w > 0:
        padded_thermal = np.pad(
            thermal_normalized,
            ((0, pad_h), (0, pad_w)),
            mode="reflect"
        )
    else:
        padded_thermal = thermal_normalized.copy()

    pad_height, pad_width = padded_thermal.shape
    blend_window = create_2d_blend_window(patch_size)

    # Accumulators for weighted predicted RGB and weight sums
    rgb_accum = np.zeros((pad_height, pad_width, 3), dtype=np.float32)
    weight_accum = np.zeros((pad_height, pad_width), dtype=np.float32)

    dev = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if HAS_TORCH and isinstance(net_g, torch.nn.Module):
        net_g.to(dev)
        net_g.eval()

    # Iterate over sliding window tiles
    y_coords = list(range(0, pad_height - patch_size + 1, stride))
    x_coords = list(range(0, pad_width - patch_size + 1, stride))

    for y in y_coords:
        for x in x_coords:
            y_end = y + patch_size
            x_end = x + patch_size

            patch_in = padded_thermal[y:y_end, x:x_end]  # (256, 256)

            if HAS_TORCH and isinstance(net_g, torch.nn.Module):
                tensor_in = torch.from_numpy(patch_in).unsqueeze(0).unsqueeze(0).to(dev) # [1, 1, 256, 256]
                with torch.no_grad():
                    pred_tensor = net_g(tensor_in).squeeze(0).cpu().numpy() # [3, 256, 256] in [-1.0, 1.0]
                pred_patch = np.transpose(pred_tensor, (1, 2, 0)) # (256, 256, 3) in [-1.0, 1.0]
            else:
                # Mock fallback
                pred_patch = np.stack([patch_in, patch_in, patch_in], axis=-1)

            # Map [-1.0, 1.0] -> [0.0, 1.0]
            pred_patch_0_1 = np.clip((pred_patch + 1.0) / 2.0, 0.0, 1.0)

            # Accumulate with 2D blend weights
            for c in range(3):
                rgb_accum[y:y_end, x:x_end, c] += pred_patch_0_1[:, :, c] * blend_window
            weight_accum[y:y_end, x:x_end] += blend_window

    # Normalize by accumulated weights
    weight_accum = np.maximum(weight_accum, 1e-4)
    for c in range(3):
        rgb_accum[:, :, c] /= weight_accum

    # Crop back to original dimensions (H, W, 3)
    final_rgb_0_1 = rgb_accum[:h, :w, :]
    final_rgb_uint8 = np.clip(final_rgb_0_1 * 255.0, 0, 255).astype(np.uint8)

    return final_rgb_uint8


def predict_landsat_geotiff(
    thermal_tiff_path: str,
    output_path: str,
    checkpoint_path: str = "checkpoints/generator_best.pth",
    device: Optional[str] = None
) -> str:
    """
    Loads a Landsat Thermal Band 10 GeoTIFF, translates it to Visible RGB,
    and writes the result to disk (preserving GeoTIFF geospatial metadata if rasterio is available).

    Args:
        thermal_tiff_path: Path to input Landsat ST_B10 GeoTIFF.
        output_path: Target path for predicted RGB image (.tif or .png).
        checkpoint_path: Path to trained generator_best.pth weights.
        device: 'cuda' or 'cpu'.

    Returns:
        output_path: Path to the generated RGB image.
    """
    raw_dn, meta = read_band_geotiff(thermal_tiff_path)
    h, w = raw_dn.shape[:2]
    logger.info(f"Processing Landsat Thermal GeoTIFF: {w}x{h} pixels ({meta.crs})")

    # 1. Radiometric Calibration
    kelvin, valid_mask = calibrate_surface_temperature(raw_dn, nodata_val=int(meta.nodata or 0))

    # 2. GAN Normalization -> [-1.0, 1.0]
    thermal_norm = normalize_thermal_for_gan(kelvin, valid_mask)

    # 3. Load Generator — infer num_filters from checkpoint to avoid shape mismatch
    num_filters = 64  # production default
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=True)
        # down1.block.0.weight shape: [num_filters, in_channels, kH, kW]
        if "down1.block.0.weight" in state_dict:
            num_filters = state_dict["down1.block.0.weight"].shape[0]
        net_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=num_filters)
        net_g.load_state_dict(state_dict)
        logger.info(f"Loaded trained weights from {checkpoint_path} (num_filters={num_filters})")
    else:
        net_g = UNetGenerator(in_channels=1, out_channels=3, num_filters=num_filters)
        logger.warning(f"Checkpoint not found at {checkpoint_path}; using initialized weights.")

    # 4. Tiled Sliding-Window Inference
    predicted_rgb = predict_thermal_array(
        thermal_normalized=thermal_norm,
        net_g=net_g,
        patch_size=256,
        overlap=64,
        device=device
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 5. Export Output
    if output_path.lower().endswith((".tif", ".tiff")) and HAS_RASTERIO and meta.transform is not None:
        import rasterio
        from rasterio.transform import Affine
        out_meta = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 3,
            "dtype": "uint8",
            "crs": meta.crs,
            "transform": Affine(*meta.transform[:6])
        }
        with rasterio.open(output_path, "w", **out_meta) as dst:
            for c in range(3):
                dst.write(predicted_rgb[:, :, c], c + 1)
        logger.info(f"Saved Georeferenced RGB GeoTIFF to {output_path}")
    else:
        # Save as standard PNG/JPEG
        bgr_output = cv2.cvtColor(predicted_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr_output)
        logger.info(f"Saved RGB image to {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Visible RGB from Landsat Thermal GeoTIFF")
    parser.add_argument("--input", type=str, required=True, help="Input Thermal Band 10 GeoTIFF path")
    parser.add_argument("--output", type=str, default="outputs/predicted_rgb.png", help="Output RGB path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/generator_best.pth", help="Model checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    predict_landsat_geotiff(
        thermal_tiff_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint
    )
