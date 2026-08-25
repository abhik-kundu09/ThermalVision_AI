"""
Post-Processing Module for Thermal Image Colorization.
Implements luminance-isolated unsharp masking to enhance structural contours 
and thermal boundaries without amplifying chrominance artifacts.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger("thermal_vision.postprocessing")


class PostprocessingError(Exception):
    """Raised when post-processing pipeline fails."""
    pass


def apply_unsharp_mask(
    image_bgr: np.ndarray,
    radius: float = 1.5,
    amount: float = 1.2,
    threshold: int = 3
) -> np.ndarray:
    """
    Applies an edge-preserving unsharp mask on the luminance (L) channel of the colorized image.
    
    Scientific Rationale:
    Direct RGB unsharp masking often produces chromatic fringing and oversaturated color halos around 
    thermal boundary transitions. By transforming to CIE-Lab color space and applying the unsharp mask 
    strictly on the L (Luminance) channel with a contrast threshold, we restore sharp thermal edges 
    and texture while preserving natural, smooth color transitions in the A/B chrominance planes.
    
    Args:
        image_bgr: Input colorized image (H, W, 3) in uint8 BGR.
        radius: Standard deviation for Gaussian blur kernel (sigma).
        amount: Sharpening multiplier (1.0 = normal, >1.0 = enhanced edge contrast).
        threshold: Minimum difference required between original and blurred pixel before sharpening 
                   is applied (avoids boosting smooth background noise).
                   
    Returns:
        Sharpened image (H, W, 3) in uint8 BGR.
    """
    if amount <= 0:
        return image_bgr.copy()

    try:
        # 1. Convert to CIE-Lab space
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 2. Compute Gaussian blurred baseline of luminance
        # Kernel size derived from sigma: (6 * sigma + 1) rounded to odd integer
        ksize = int(2 * round(3 * radius) + 1)
        ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
        
        blurred_l = cv2.GaussianBlur(l_channel, (ksize, ksize), sigmaX=radius, sigmaY=radius)

        # 3. Compute high-pass detail mask: (Original - Blurred)
        l_float = l_channel.astype(np.float32)
        blurred_float = blurred_l.astype(np.float32)
        diff = l_float - blurred_float

        # 4. Apply thresholding to prevent sharpening sensor noise in flat areas
        if threshold > 0:
            mask = np.abs(diff) >= threshold
            sharpened_l = np.where(mask, l_float + amount * diff, l_float)
        else:
            sharpened_l = l_float + amount * diff

        # 5. Clip to valid 8-bit dynamic range [0, 255]
        sharpened_l = np.clip(sharpened_l, 0, 255).astype(np.uint8)

        # 6. Merge back with pristine chromatic channels
        sharpened_lab = cv2.merge([sharpened_l, a_channel, b_channel])
        result_bgr = cv2.cvtColor(sharpened_lab, cv2.COLOR_LAB2BGR)
        return result_bgr

    except Exception as exc:
        logger.error(f"Unsharp masking failed: {exc}")
        raise PostprocessingError(f"Postprocessing unsharp mask failed: {exc}")


def postprocess_colorized_image(
    colorized_bgr: np.ndarray,
    unsharp_radius: float = 1.5,
    unsharp_amount: float = 1.2,
    unsharp_threshold: int = 3
) -> np.ndarray:
    """
    Executes the post-processing pipeline on the raw AI colorization output.
    """
    return apply_unsharp_mask(
        colorized_bgr,
        radius=unsharp_radius,
        amount=unsharp_amount,
        threshold=unsharp_threshold
    )
