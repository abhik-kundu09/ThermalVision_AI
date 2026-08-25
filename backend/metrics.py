"""
Scientifically Validated Image Quality and Performance Metrics for Thermal Computer Vision.
Implements reference-free metrics (Tenengrad, Entropy, RMS Contrast, EPI)
and strictly validates full-reference metrics (PSNR, SSIM) when paired ground-truth is available.
"""

import math
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from backend.schemas import ImageMetrics, LatencyBreakdown


def calculate_shannon_entropy(image_gray: np.ndarray) -> float:
    """
    Computes Shannon Information Entropy (in bits) of an 8-bit image:
    H = - sum( p(i) * log2(p(i)) ) for all gray levels i.
    Higher entropy indicates richer radiometric detail distribution.
    """
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    total_pixels = image_gray.shape[0] * image_gray.shape[1]
    if total_pixels == 0:
        return 0.0

    prob = hist.flatten() / total_pixels
    # Filter out zero probabilities to avoid log2(0)
    prob_non_zero = prob[prob > 0]
    entropy = -float(np.sum(prob_non_zero * np.log2(prob_non_zero)))
    return round(entropy, 4)


def calculate_tenengrad_sharpness(image_gray: np.ndarray) -> float:
    """
    Calculates Tenengrad Gradient Sharpness metric:
    T = sum( (Sobel_x(I))^2 + (Sobel_y(I))^2 ) / (N * M)
    Quantifies edge strength and structural sharpness across thermal contours.
    """
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag_sq = sobel_x**2 + sobel_y**2
    tenengrad = float(np.mean(gradient_mag_sq))
    return round(tenengrad, 2)


def calculate_rms_contrast(image_gray: np.ndarray) -> float:
    """
    Calculates Root Mean Square (RMS) Contrast:
    Standard deviation of pixel intensities normalized to [0, 1].
    C_rms = sqrt( 1/(N*M) * sum( (I(x,y)/255 - I_mean/255)^2 ) )
    """
    norm = image_gray.astype(np.float64) / 255.0
    rms = float(np.std(norm))
    return round(rms, 4)


def calculate_edge_preservation_index(input_gray: np.ndarray, output_gray: np.ndarray) -> float:
    """
    Calculates Edge Preservation Index (EPI):
    Measures the correlation between high-pass Laplacian filters of the preprocessed 
    input and the colorized output luminance to verify that thermal edges were preserved 
    and not blurred or hallucinatively shifted.
    """
    # Compute Laplacian high-pass
    lap_in = cv2.Laplacian(input_gray, cv2.CV_64F)
    lap_out = cv2.Laplacian(output_gray, cv2.CV_64F)

    delta_in = lap_in - np.mean(lap_in)
    delta_out = lap_out - np.mean(lap_out)

    numerator = np.sum(delta_in * delta_out)
    denominator = math.sqrt(np.sum(delta_in**2) * np.sum(delta_out**2))

    if denominator == 0:
        return 1.0

    epi = float(numerator / denominator)
    return round(max(0.0, min(1.0, epi)), 4)


def calculate_full_reference_metrics(
    pred_bgr: np.ndarray,
    ground_truth_bgr: Optional[np.ndarray]
) -> Tuple[Optional[float], Optional[float], bool, Optional[str]]:
    """
    Calculates PSNR and SSIM ONLY when a valid paired ground-truth RGB image is provided.
    
    Scientific Note:
    Comparing cross-spectral data (single-channel IR vs 3-channel predicted RGB) without 
    a ground-truth reference is mathematically meaningless because the pixel domains are 
    fundamentally non-commensurate (long-wave infrared thermal radiance vs visible RGB reflectance).
    """
    if ground_truth_bgr is None:
        disclaimer = (
            "PSNR and SSIM are reference-based full-fidelity metrics requiring a paired ground-truth "
            "RGB image captured at identical optical perspective. Direct mathematical comparison "
            "between cross-spectral single-channel thermal radiance and synthesized 3-channel RGB "
            "is mathematically invalid without a ground-truth reference."
        )
        return None, None, False, disclaimer

    try:
        # Resize ground truth if slightly mismatched
        if ground_truth_bgr.shape != pred_bgr.shape:
            ground_truth_bgr = cv2.resize(
                ground_truth_bgr,
                (pred_bgr.shape[1], pred_bgr.shape[0]),
                interpolation=cv2.INTER_AREA
            )

        # Convert BGR to RGB for standard metric alignment
        pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(ground_truth_bgr, cv2.COLOR_BGR2RGB)

        psnr_val = float(compute_psnr(gt_rgb, pred_rgb, data_range=255))
        ssim_val = float(compute_ssim(gt_rgb, pred_rgb, channel_axis=2, data_range=255))

        return round(psnr_val, 2), round(ssim_val, 4), True, None

    except Exception as exc:
        return None, None, False, f"Failed calculating reference metrics: {exc}"


def compute_all_metrics(
    raw_gray: np.ndarray,
    final_bgr: np.ndarray,
    latency_breakdown: LatencyBreakdown,
    ground_truth_bgr: Optional[np.ndarray] = None
) -> ImageMetrics:
    """
    Calculates the complete suite of performance and image quality metrics.
    """
    final_gray = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2GRAY)

    # 1. No-Reference Quality Metrics
    tenengrad_in = calculate_tenengrad_sharpness(raw_gray)
    tenengrad_out = calculate_tenengrad_sharpness(final_gray)

    entropy_in = calculate_shannon_entropy(raw_gray)
    entropy_out = calculate_shannon_entropy(final_gray)

    rms_in = calculate_rms_contrast(raw_gray)
    rms_out = calculate_rms_contrast(final_gray)

    epi = calculate_edge_preservation_index(raw_gray, final_gray)

    # 2. Reference Metrics (PSNR / SSIM)
    psnr, ssim, has_gt, disclaimer = calculate_full_reference_metrics(final_bgr, ground_truth_bgr)

    return ImageMetrics(
        latency=latency_breakdown,
        tenengrad_sharpness_input=tenengrad_in,
        tenengrad_sharpness_output=tenengrad_out,
        shannon_entropy_input=entropy_in,
        shannon_entropy_output=entropy_out,
        rms_contrast_input=rms_in,
        rms_contrast_output=rms_out,
        edge_preservation_index=epi,
        has_ground_truth=has_gt,
        psnr=psnr,
        ssim=ssim,
        reference_disclaimer=disclaimer
    )
