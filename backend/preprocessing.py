"""
Thermal IR Image Preprocessing Module.
Implements grayscale normalization, 16-bit dynamic range mapping, CLAHE,
and edge-preserving bilateral filtering using OpenCV.
"""

import base64
import logging
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger("thermal_vision.preprocessing")


class PreprocessingError(Exception):
    """Raised when thermal image decoding or preprocessing fails."""
    pass


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes into a NumPy array using OpenCV.
    Supports standard 8-bit formats as well as 16-bit uncompressed TIFF/PNG thermal captures.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise PreprocessingError("Empty image byte buffer provided.")

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    
    # Try decoding with IMREAD_UNCHANGED first to preserve 16-bit radiometric depth if present
    img = cv2.imdecode(np_buffer, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise PreprocessingError(
            "Failed to decode image data. File may be corrupted or in an unsupported format."
        )

    return img


def normalize_thermal_to_grayscale_8bit(raw_img: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Normalizes multi-channel or high bit-depth thermal imagery into a standardized 8-bit single-channel array.
    
    Scientific Rationale:
    - 16-bit FLIR/TIFF sensors store raw temperature counts (e.g. 0 to 65535 or Kelvin x 100).
      We perform min-max robust percentile scaling (1% - 99% or full min-max) to map the dynamic range 
      optimally to 0..255 without radiometric clipping.
    - Multi-channel RGB/RGBA inputs are converted to single-channel luminance using standard 
      ITU-R BT.601 coefficients (or equal weighting if channels are identical).
    """
    bit_depth = str(raw_img.dtype)

    # 1. Handle 16-bit or 32-bit floating point radiometric data
    if raw_img.dtype in (np.uint16, np.int32, np.float32, np.float64):
        # Flatten if multi-channel 16-bit
        if len(raw_img.shape) > 2:
            raw_img = raw_img[:, :, 0]
            
        min_val = np.min(raw_img)
        max_val = np.max(raw_img)
        
        if max_val > min_val:
            normalized = ((raw_img - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        else:
            normalized = np.zeros(raw_img.shape, dtype=np.uint8)
        return normalized, f"{bit_depth}->uint8 (Min-Max Scaled)"

    # 2. Handle 8-bit multi-channel (RGB or RGBA)
    if len(raw_img.shape) == 3:
        channels = raw_img.shape[2]
        if channels == 4:
            # RGBA to BGR first, then Grayscale
            bgr = cv2.cvtColor(raw_img, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return gray, "8-bit RGBA->Grayscale"
        elif channels == 3:
            # Check if all 3 channels are identical (already grayscale encoded in 3 channels)
            diff1 = np.abs(raw_img[:, :, 0].astype(int) - raw_img[:, :, 1].astype(int))
            diff2 = np.abs(raw_img[:, :, 1].astype(int) - raw_img[:, :, 2].astype(int))
            if np.max(diff1) < 2 and np.max(diff2) < 2:
                return raw_img[:, :, 0].copy(), "8-bit 3-Channel Thermal->Grayscale"
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            return gray, "8-bit RGB->Grayscale"
        elif channels == 1:
            return raw_img[:, :, 0], "8-bit Single Channel"

    elif len(raw_img.shape) == 2:
        return raw_img.copy(), "8-bit Single Channel Grayscale"

    raise PreprocessingError(f"Unsupported image dimensions: {raw_img.shape}")


def apply_clahe(
    gray_img: np.ndarray, 
    clip_limit: float = 2.0, 
    grid_size: int = 8
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Why CLAHE for Thermal IR?
    Standard global histogram equalization over-amplifies noise in homogeneous background regions 
    (e.g., cold sky or ambient ground). CLAHE operates on localized contextual tiles (e.g. 8x8) 
    and clips the histogram peak at `clip_limit` to redistribute energy smoothly, enhancing subtle 
    thermal gradients (e.g., human body heat, mechanical component friction) without noise blowout.
    """
    if clip_limit <= 0:
        return gray_img.copy()

    grid_dim = max(2, min(grid_size, 64))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_dim, grid_dim))
    clahe_enhanced = clahe.apply(gray_img)
    return clahe_enhanced


def apply_bilateral_filter(
    image: np.ndarray, 
    d: int = 9, 
    sigma_color: float = 75.0, 
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Applies Bilateral Filtering for edge-preserving denoising.
    
    Why Bilateral Filtering for Thermal IR?
    Thermal microbolometer sensors suffer from fixed-pattern noise (FPN) and high thermal sensor noise.
    Standard Gaussian blur blurs crucial thermal boundaries (isotherms). Bilateral filtering weights 
    neighboring pixels by both geometric spatial closeness (`sigma_space`) and radiometric intensity 
    similarity (`sigma_color`), smoothing high-frequency thermal sensor grain while keeping sharp 
    structural boundaries intact.
    """
    if d <= 0 or (sigma_color <= 0 and sigma_space <= 0):
        return image.copy()

    denoised = cv2.bilateralFilter(
        src=image,
        d=int(d),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space)
    )
    return denoised


def encode_image_to_base64(image: np.ndarray, fmt: str = ".png") -> str:
    """Encodes an OpenCV image array to a base64 Data URI."""
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise PreprocessingError("Failed to encode processed image to output buffer.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"


def preprocess_thermal_image(
    image_bytes: bytes,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Tuple[int, int]]:
    """
    Executes the complete local preprocessing pipeline:
    1. Byte decoding & format validation
    2. Conversion/Normalization to 8-bit single channel
    3. CLAHE local contrast adaptation
    4. Bilateral edge-preserving filtering
    5. Conversion to 3-channel BGR for model compatibility
    
    Returns:
        (raw_normalized_gray, preprocessed_gray, preprocessed_bgr, bit_depth_info, (height, width))
    """
    raw_decoded = decode_image_bytes(image_bytes)
    h, w = raw_decoded.shape[:2]

    if h < 16 or w < 16:
        raise PreprocessingError(f"Image dimensions too small ({w}x{h}). Minimum required is 16x16.")
    if h > 8192 or w > 8192:
        raise PreprocessingError(f"Image dimensions too large ({w}x{h}). Maximum supported is 8192x8192.")

    # Step 1: Normalize to 8-bit Grayscale
    gray_normalized, depth_info = normalize_thermal_to_grayscale_8bit(raw_decoded)

    # Step 2: Apply CLAHE
    clahe_applied = apply_clahe(
        gray_normalized, 
        clip_limit=clahe_clip_limit, 
        grid_size=clahe_grid_size
    )

    # Step 3: Apply Bilateral Filtering
    preprocessed_gray = apply_bilateral_filter(
        clahe_applied,
        d=bilateral_d,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space
    )

    # Step 4: Convert to 3-Channel BGR representation (Colorization models expect 3-channel input)
    preprocessed_bgr = cv2.cvtColor(preprocessed_gray, cv2.COLOR_GRAY2BGR)

    return gray_normalized, preprocessed_gray, preprocessed_bgr, depth_info, (h, w)
