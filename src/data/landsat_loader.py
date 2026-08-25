"""
Landsat Collection 2 GeoTIFF Ingestion and Scene Loader.

Uses rasterio (with fallback support) to load:
- Band 10: Surface Temperature (ST)
- Band 4: Red (SR_B4)
- Band 3: Green (SR_B3)
- Band 2: Blue (SR_B2)
and extracts full geospatial metadata (CRS, affine transform, bounds, resolution).
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import cv2

from src.data.calibrate import (
    calibrate_surface_temperature,
    calibrate_surface_reflectance,
    normalize_thermal_for_gan,
    normalize_rgb_for_gan
)

logger = logging.getLogger("ps10.data.loader")


@dataclass
class GeospatialMetadata:
    """Geospatial metadata container for a Landsat raster band."""
    crs: str
    transform: Optional[Tuple[float, ...]]  # Affine transform tuple
    width: int
    height: int
    nodata: Optional[float]
    resolution: Tuple[float, float]
    bounds: Optional[Tuple[float, float, float, float]]  # (left, bottom, right, top)


@dataclass
class LandsatScene:
    """Container holding calibrated Landsat arrays and associated spatial metadata."""
    scene_id: str
    thermal_raw: np.ndarray          # Raw DN uint16
    thermal_kelvin: np.ndarray       # Calibrated Kelvin float32
    thermal_mask: np.ndarray         # Boolean valid mask
    rgb_raw: np.ndarray              # Raw DN uint16 (H, W, 3) [B4, B3, B2]
    rgb_reflectance: np.ndarray      # Calibrated [0.0, 1.0] float32 (H, W, 3)
    rgb_mask: np.ndarray             # Boolean valid mask (H, W)
    metadata: GeospatialMetadata


def read_band_geotiff(file_path: str) -> Tuple[np.ndarray, GeospatialMetadata]:
    """
    Reads a single-band GeoTIFF file and extracts its pixel array and spatial metadata.

    Args:
        file_path: Absolute or relative path to the GeoTIFF file.

    Returns:
        band_data: 2D numpy array containing raw digital numbers.
        metadata: GeospatialMetadata dataclass instance.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Landsat band file not found: {file_path}")

    if HAS_RASTERIO:
        with rasterio.open(file_path) as src:
            band_data = src.read(1)
            crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
            transform_tuple = tuple(src.transform) if src.transform else None
            res = src.res if hasattr(src, "res") else (30.0, 30.0)
            bounds_tuple = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top) if src.bounds else None
            meta = GeospatialMetadata(
                crs=crs_str,
                transform=transform_tuple,
                width=src.width,
                height=src.height,
                nodata=src.nodata,
                resolution=res,
                bounds=bounds_tuple
            )
            return band_data, meta
    else:
        # Fallback for systems without rasterio/GDAL compiled
        logger.warning(f"rasterio not installed; using OpenCV fallback for {file_path}.")
        raw_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            raise ValueError(f"Failed to read image at {file_path} via OpenCV fallback.")
        
        if len(raw_img.shape) > 2:
            raw_img = raw_img[:, :, 0]

        h, w = raw_img.shape[:2]
        meta = GeospatialMetadata(
            crs="UNKNOWN (rasterio not installed)",
            transform=None,
            width=w,
            height=h,
            nodata=0.0,
            resolution=(30.0, 30.0),
            bounds=None
        )
        return raw_img, meta


def load_landsat_scene(
    st_b10_path: str,
    sr_b4_path: str,
    sr_b3_path: str,
    sr_b2_path: str,
    scene_id: Optional[str] = None
) -> LandsatScene:
    """
    Loads and radiometrically calibrates a complete Landsat 8/9 scene (ST + RGB SR bands).

    Args:
        st_b10_path: Path to Surface Temperature Band 10 GeoTIFF.
        sr_b4_path: Path to Red Band 4 GeoTIFF.
        sr_b3_path: Path to Green Band 3 GeoTIFF.
        sr_b2_path: Path to Blue Band 2 GeoTIFF.
        scene_id: Optional identifier for the scene.

    Returns:
        LandsatScene dataclass containing calibrated thermal and RGB arrays with metadata.
    """
    sid = scene_id or os.path.basename(st_b10_path).split("_")[0]

    # 1. Read individual band GeoTIFFs
    thermal_raw, st_meta = read_band_geotiff(st_b10_path)
    b4_raw, b4_meta = read_band_geotiff(sr_b4_path)
    b3_raw, _ = read_band_geotiff(sr_b3_path)
    b2_raw, _ = read_band_geotiff(sr_b2_path)

    # 2. Stack RGB bands (B4=Red, B3=Green, B2=Blue) -> shape (H, W, 3)
    rgb_raw = np.stack([b4_raw, b3_raw, b2_raw], axis=-1)

    # 3. Radiometric Calibration
    thermal_kelvin, thermal_mask = calibrate_surface_temperature(
        thermal_raw,
        nodata_val=int(st_meta.nodata) if st_meta.nodata is not None else 0
    )

    r_refl, r_mask = calibrate_surface_reflectance(b4_raw)
    g_refl, g_mask = calibrate_surface_reflectance(b3_raw)
    b_refl, b_mask = calibrate_surface_reflectance(b2_raw)

    rgb_reflectance = np.stack([r_refl, g_refl, b_refl], axis=-1)
    rgb_mask = r_mask & g_mask & b_mask

    return LandsatScene(
        scene_id=sid,
        thermal_raw=thermal_raw,
        thermal_kelvin=thermal_kelvin,
        thermal_mask=thermal_mask,
        rgb_raw=rgb_raw,
        rgb_reflectance=rgb_reflectance,
        rgb_mask=rgb_mask,
        metadata=st_meta
    )
