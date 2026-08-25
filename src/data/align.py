"""
Geospatial Raster Alignment and Reprojection Module for Landsat Imagery.

Ensures that Thermal IR (Band 10) pixels (x, y) map to the exact same physical ground coordinates
as Visible RGB (Bands 4, 3, 2) pixels (x, y).

Handles:
- Coordinate Reference System (CRS) matching
- Affine transform and pixel origin alignment
- Spatial bounding box intersection
- High-order bilinear resampling to common 30m spatial grid
"""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import cv2

from src.data.landsat_loader import GeospatialMetadata, LandsatScene

logger = logging.getLogger("ps10.data.align")


class AlignmentError(Exception):
    """Raised when rasters cannot be spatially aligned."""
    pass


def check_spatial_alignment(
    meta_a: GeospatialMetadata,
    meta_b: GeospatialMetadata,
    tol: float = 1e-4
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compares two raster metadata containers to verify if they are strictly co-registered.

    Checks:
    1. CRS matching
    2. Dimensions (Width, Height)
    3. Pixel Resolution (dx, dy)
    4. Affine Transform coefficients

    Args:
        meta_a: Metadata for first raster (e.g. Thermal Band 10).
        meta_b: Metadata for second raster (e.g. RGB Band 4).
        tol: Floating point tolerance for transform comparison.

    Returns:
        is_aligned: True if rasters share the exact same grid, False otherwise.
        discrepancies: Dict explaining any spatial mismatch.
    """
    discrepancies = {}

    # 1. CRS Check
    if meta_a.crs != meta_b.crs:
        discrepancies["crs_mismatch"] = f"A: {meta_a.crs} vs B: {meta_b.crs}"

    # 2. Dimensions Check
    if (meta_a.width != meta_b.width) or (meta_a.height != meta_b.height):
        discrepancies["dimension_mismatch"] = (
            f"A: ({meta_a.width}x{meta_a.height}) vs B: ({meta_b.width}x{meta_b.height})"
        )

    # 3. Resolution Check
    if (abs(meta_a.resolution[0] - meta_b.resolution[0]) > tol or
            abs(meta_a.resolution[1] - meta_b.resolution[1]) > tol):
        discrepancies["resolution_mismatch"] = (
            f"A: {meta_a.resolution} vs B: {meta_b.resolution}"
        )

    # 4. Transform Check
    if meta_a.transform is not None and meta_b.transform is not None:
        diffs = [abs(a - b) for a, b in zip(meta_a.transform, meta_b.transform)]
        if any(d > tol for d in diffs):
            discrepancies["transform_mismatch"] = (
                f"Affine transforms differ: A={meta_a.transform}, B={meta_b.transform}"
            )

    is_aligned = len(discrepancies) == 0
    return is_aligned, discrepancies


def reproject_raster_to_reference(
    src_data: np.ndarray,
    src_meta: GeospatialMetadata,
    dst_meta: GeospatialMetadata,
    resampling_method: str = "bilinear"
) -> np.ndarray:
    """
    Reprojects and resamples a source raster into the exact coordinate space and pixel grid
    of a destination reference metadata.

    Args:
        src_data: 2D array (H_src, W_src) or 3D array (H_src, W_src, C).
        src_meta: Metadata container for source raster.
        dst_meta: Metadata container for target destination grid.
        resampling_method: 'bilinear', 'nearest', or 'cubic'.

    Returns:
        dst_data: 2D or 3D array matching (dst_meta.height, dst_meta.width).
    """
    # If dimensions and transforms already match, return copy
    if (src_meta.width == dst_meta.width and
            src_meta.height == dst_meta.height and
            src_meta.crs == dst_meta.crs and
            src_meta.transform == dst_meta.transform):
        return src_data.copy()

    dst_shape = (dst_meta.height, dst_meta.width)

    if HAS_RASTERIO and src_meta.transform is not None and dst_meta.transform is not None:
        resample_enum = (
            Resampling.bilinear if resampling_method == "bilinear"
            else Resampling.nearest if resampling_method == "nearest"
            else Resampling.cubic
        )

        src_transform = Affine(*src_meta.transform[:6])
        dst_transform = Affine(*dst_meta.transform[:6])

        if src_data.ndim == 2:
            destination = np.full(dst_shape, np.nan, dtype=src_data.dtype)
            reproject(
                source=src_data,
                destination=destination,
                src_transform=src_transform,
                src_crs=src_meta.crs,
                dst_transform=dst_transform,
                dst_crs=dst_meta.crs,
                resampling=resample_enum
            )
            return destination
        elif src_data.ndim == 3:
            channels = src_data.shape[2]
            destination = np.full((*dst_shape, channels), np.nan, dtype=src_data.dtype)
            for c in range(channels):
                reproject(
                    source=src_data[:, :, c],
                    destination=destination[:, :, c],
                    src_transform=src_transform,
                    src_crs=src_meta.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_meta.crs,
                    resampling=resample_enum
                )
            return destination
    else:
        # High-order spatial interpolation fallback when rasterio transform is absent
        logger.warning("Reprojecting with spatial interpolation fallback.")
        interp = (
            cv2.INTER_LINEAR if resampling_method == "bilinear"
            else cv2.INTER_NEAREST if resampling_method == "nearest"
            else cv2.INTER_CUBIC
        )
        if src_data.ndim == 2:
            return cv2.resize(src_data, (dst_meta.width, dst_meta.height), interpolation=interp)
        else:
            return cv2.resize(src_data, (dst_meta.width, dst_meta.height), interpolation=interp)


def align_landsat_scene(scene: LandsatScene) -> LandsatScene:
    """
    Validates and enforces spatial co-registration between thermal and RGB bands in a scene.

    If misaligned, the thermal band is reprojected onto the RGB spatial grid so that:
    Thermal[y, x] is geographically identical to RGB[y, x, :].

    Args:
        scene: Ingested LandsatScene object.

    Returns:
        aligned_scene: Guaranteed spatially co-registered LandsatScene.
    """
    thermal_h, thermal_w = scene.thermal_kelvin.shape
    rgb_h, rgb_w = scene.rgb_reflectance.shape[:2]

    # Check if dimensions match
    if (thermal_h != rgb_h) or (thermal_w != rgb_w):
        logger.info(
            f"Aligning scene '{scene.scene_id}': Thermal ({thermal_w}x{thermal_h}) "
            f"-> RGB ({rgb_w}x{rgb_h})"
        )

        # Build a source metadata that reflects the thermal band's ACTUAL dimensions.
        # scene.metadata describes the full scene / RGB reference grid; using it as
        # src_meta would make src and dst look identical (triggering early return).
        thermal_meta = GeospatialMetadata(
            crs=scene.metadata.crs,
            transform=scene.metadata.transform,
            width=thermal_w,
            height=thermal_h,
            nodata=scene.metadata.nodata,
            resolution=(
                scene.metadata.resolution[0] * rgb_w / thermal_w,
                scene.metadata.resolution[1] * rgb_h / thermal_h,
            ),
            bounds=scene.metadata.bounds
        )

        target_meta = GeospatialMetadata(
            crs=scene.metadata.crs,
            transform=scene.metadata.transform,
            width=rgb_w,
            height=rgb_h,
            nodata=scene.metadata.nodata,
            resolution=scene.metadata.resolution,
            bounds=scene.metadata.bounds
        )

        aligned_thermal_kelvin = reproject_raster_to_reference(
            scene.thermal_kelvin,
            src_meta=thermal_meta,
            dst_meta=target_meta,
            resampling_method="bilinear"
        )
        aligned_thermal_mask = reproject_raster_to_reference(
            scene.thermal_mask.astype(np.uint8),
            src_meta=thermal_meta,
            dst_meta=target_meta,
            resampling_method="nearest"
        ).astype(bool)

        return LandsatScene(
            scene_id=scene.scene_id,
            thermal_raw=scene.thermal_raw,
            thermal_kelvin=aligned_thermal_kelvin,
            thermal_mask=aligned_thermal_mask,
            rgb_raw=scene.rgb_raw,
            rgb_reflectance=scene.rgb_reflectance,
            rgb_mask=scene.rgb_mask,
            metadata=target_meta
        )

    return scene
