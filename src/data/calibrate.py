"""
Landsat Collection 2 Level-2 Radiometric Calibration Module.

Implements USGS official calibration formulas for:
1. Surface Temperature (ST - Band 10):
   Kelvin = DN * 0.00341802 + 149.0
2. Surface Reflectance (SR - Bands 2, 3, 4):
   Reflectance = DN * 0.0000275 - 0.2
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

# Landsat 8/9 Collection 2 Level-2 Calibration Constants (USGS Official)
LANDSAT_ST_SCALE = 0.00341802
LANDSAT_ST_OFFSET = 149.0

LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2

# Default Nodata / Fill Value in Landsat Level-2 integer rasters
LANDSAT_NODATA_INT = 0


def calibrate_surface_temperature(
    raw_dn: np.ndarray,
    nodata_val: int = LANDSAT_NODATA_INT,
    clip_valid_range: bool = True,
    min_kelvin: float = 200.0,
    max_kelvin: float = 350.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts raw Landsat Collection 2 Band 10 Digital Numbers (DN) to physical temperature in Kelvin.

    Formula:
        Temperature (K) = raw_dn * 0.00341802 + 149.0

    Args:
        raw_dn: 2D array of unsigned 16-bit integers (uint16) or float.
        nodata_val: Pixel value indicating nodata/fill.
        clip_valid_range: Whether to clip extreme unphysical temperatures.
        min_kelvin: Lower physical temperature bound (default 200 K / -73.15 C).
        max_kelvin: Upper physical temperature bound (default 350 K / +76.85 C).

    Returns:
        temp_kelvin: 2D float32 array in Kelvin (nodata pixels are set to np.nan).
        valid_mask: 2D boolean array (True where pixel is valid, False where nodata).
    """
    dn_float = raw_dn.astype(np.float32)
    valid_mask = (raw_dn != nodata_val) & (raw_dn > 0)

    # Initialize with NaN
    temp_kelvin = np.full(raw_dn.shape, np.nan, dtype=np.float32)

    # Apply calibration to valid pixels
    temp_kelvin[valid_mask] = dn_float[valid_mask] * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET

    if clip_valid_range:
        temp_kelvin[valid_mask] = np.clip(temp_kelvin[valid_mask], min_kelvin, max_kelvin)

    return temp_kelvin, valid_mask


def kelvin_to_celsius(temp_kelvin: np.ndarray) -> np.ndarray:
    """Converts temperature array from Kelvin to Celsius."""
    return temp_kelvin - 273.15


def calibrate_surface_reflectance(
    raw_dn: np.ndarray,
    nodata_val: int = LANDSAT_NODATA_INT,
    clip_valid_range: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts raw Landsat Collection 2 Surface Reflectance (SR) Digital Numbers to unitless reflectance.

    Formula:
        Reflectance = raw_dn * 0.0000275 - 0.2

    Args:
        raw_dn: Array of unsigned 16-bit integers (uint16) or float.
        nodata_val: Pixel value indicating nodata/fill.
        clip_valid_range: Whether to clip reflectance to physical [0.0, 1.0].

    Returns:
        reflectance: float32 array with reflectance in [0.0, 1.0] (nodata pixels are np.nan).
        valid_mask: boolean array (True where pixel is valid).
    """
    dn_float = raw_dn.astype(np.float32)
    valid_mask = (raw_dn != nodata_val) & (raw_dn > 0)

    reflectance = np.full(raw_dn.shape, np.nan, dtype=np.float32)
    reflectance[valid_mask] = dn_float[valid_mask] * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET

    if clip_valid_range:
        # Negative values due to atmospheric overcorrection are clipped to 0.0
        reflectance[valid_mask] = np.clip(reflectance[valid_mask], 0.0, 1.0)

    return reflectance, valid_mask


def normalize_thermal_for_gan(
    temp_kelvin: np.ndarray,
    valid_mask: np.ndarray,
    min_temp: float = 260.0,
    max_temp: float = 330.0
) -> np.ndarray:
    """
    Normalizes thermal Kelvin array to [-1.0, 1.0] dynamic range for Pix2Pix Generator input.

    Args:
        temp_kelvin: 2D float32 array in Kelvin.
        valid_mask: 2D boolean array of valid pixels.
        min_temp: Minimum temperature in Kelvin (mapped to -1.0).
        max_temp: Maximum temperature in Kelvin (mapped to +1.0).

    Returns:
        normalized_thermal: 2D float32 array in [-1.0, 1.0] with nodata pixels filled with -1.0.
    """
    clipped = np.clip(temp_kelvin, min_temp, max_temp)
    # Map [min_temp, max_temp] to [0.0, 1.0]
    norm_0_1 = (clipped - min_temp) / (max_temp - min_temp)
    # Map [0.0, 1.0] to [-1.0, 1.0]
    norm_neg1_1 = norm_0_1 * 2.0 - 1.0

    # Fill invalid/nodata with -1.0 (cold background representation)
    norm_neg1_1[~valid_mask] = -1.0
    return norm_neg1_1.astype(np.float32)


def normalize_rgb_for_gan(
    rgb_reflectance: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    Normalizes surface reflectance RGB array from [0.0, 1.0] to [-1.0, 1.0] for Pix2Pix Target.

    Args:
        rgb_reflectance: 3D float32 array (H, W, 3) in [0.0, 1.0].
        valid_mask: 2D or 3D boolean array of valid pixels.

    Returns:
        normalized_rgb: 3D float32 array (H, W, 3) in [-1.0, 1.0].
    """
    clipped = np.clip(rgb_reflectance, 0.0, 1.0)
    norm_neg1_1 = clipped * 2.0 - 1.0

    if valid_mask.ndim == 2:
        norm_neg1_1[~valid_mask] = -1.0
    else:
        norm_neg1_1[~valid_mask] = -1.0

    return norm_neg1_1.astype(np.float32)
