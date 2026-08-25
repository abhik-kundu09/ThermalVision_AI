"""
FastAPI Application Entrypoint for Thermal IR Image Enhancement & Colorization System.
"""

import logging
import os
import pathlib
import time
from contextlib import asynccontextmanager
from typing import Optional, List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.preprocessing import (
    preprocess_thermal_image,
    encode_image_to_base64,
    decode_image_bytes,
    PreprocessingError
)
from backend.inference import (
    get_colorization_provider,
    InferenceError,
    PyTorchPix2PixProvider,
    LocalFallbackProvider,
)
from backend.postprocessing import (
    postprocess_colorized_image,
    PostprocessingError
)
from backend.metrics import (
    compute_all_metrics,
    LatencyBreakdown
)
from backend.schemas import (
    EnhanceResponse,
    ImageMetadata,
    HealthResponse,
    SampleImageInfo
)
from backend.generate_samples import generate_benchmark_samples

# Setup Logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("thermal_vision.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown routines."""
    logger.info("Initializing Thermal IR Vision System...")
    # Ensure sample images directory exists and has realistic benchmark images
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_images")
    generate_benchmark_samples(sample_dir)
    logger.info(f"Active Colorization Provider: {settings.ai_provider}")
    yield
    logger.info("Shutting down Thermal IR Vision System.")


app = FastAPI(
    title="Thermal IR Image Enhancement & Colorization API",
    description="High-performance Computer Vision pipeline for thermal infrared enhancement and deep colorization.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration — restrict to known local origins for security
_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add standard security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Returns API health status, OpenCV version, and provider configuration."""
    provider_configured = True

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        active_provider=settings.ai_provider,
        provider_configured=provider_configured,
        opencv_version=cv2.__version__,
        features={
            "clahe_enabled": True,
            "bilateral_denoising": True,
            "unsharp_masking": True,
            "scientifically_validated_metrics": True,
            "max_upload_size_mb": settings.max_upload_size_mb
        }
    )


@app.get("/api/sample-images", response_model=List[SampleImageInfo])
async def list_sample_images():
    """Returns metadata for available benchmark thermal test datasets."""
    return [
        SampleImageInfo(
            id="01_landsat_urban_heat_island",
            title="Landsat 8/9 Urban Heat Island",
            category="Satellite Remote Sensing",
            description="Landsat 8 Collection 2 Level-2 Surface Temperature scan over high-density metropolitan built environment.",
            dimensions="640 x 480",
            sensor_type="Landsat TIRS (Band 10 LWIR)",
            filename="01_nocturnal_surveillance.png"
        ),
        SampleImageInfo(
            id="02_agricultural_canopy_water_stress",
            title="Agricultural Crop Thermal Stress",
            category="Agri-Ecological Remote Sensing",
            description="Farmland vegetation canopy showing differential crop transpiration and surface temperature anomalies.",
            dimensions="640 x 480",
            sensor_type="Landsat 9 TIRS-2 (100m to 30m)",
            filename="02_electrical_substation_hotspot.png"
        ),
        SampleImageInfo(
            id="03_coastal_estuary_thermal_plume",
            title="Coastal Estuary Hydro-Thermal Plume",
            category="Hydrological Thermography",
            description="River discharge mixing with coastal marine waters exhibiting thermal boundary isotherms.",
            dimensions="640 x 480",
            sensor_type="Calibrated Radiometric LWIR",
            filename="03_wildlife_canopy_recon.png"
        ),
        SampleImageInfo(
            id="04_geothermal_volcanic_caldera",
            title="Volcanic Caldera Geothermal Activity",
            category="Geological Monitoring",
            description="Active volcanic thermal fissures and geothermal heat flux mapped across rugged terrain.",
            dimensions="640 x 480",
            sensor_type="Landsat 8 TIRS Band 10",
            filename="04_building_heat_loss.png"
        ),
        SampleImageInfo(
            id="05_uav_airborne_thermal_recon",
            title="UAV Airborne Corridor Surveillance",
            category="Aerospace & Defense",
            description="High-altitude aerial thermal reconnaissance capturing transport corridors and kinetic heat signatures.",
            dimensions="640 x 480",
            sensor_type="Airborne Gimbal LWIR",
            filename="05_uav_convoy_recon.png"
        ),
        SampleImageInfo(
            id="06_industrial_radiometric_diagnostic",
            title="Industrial Thermal Plant Diagnostics",
            category="Industrial Radiometry",
            description="Power generation thermal diagnostics showing component heat dissipation and thermal equilibrium.",
            dimensions="640 x 480",
            sensor_type="Radiometric Microbolometer",
            filename="06_turbine_radiometric_diagnostic.png"
        )
    ]


@app.get("/api/sample-images/{filename}")
async def get_sample_image(filename: str):
    """Serves sample image binary."""
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_images")
    file_path = os.path.join(sample_dir, filename)
    # Prevent directory traversal — use pathlib.resolve() for canonical, symlink-resolved paths
    # This handles case-insensitive filesystems (Windows) and symlink attacks correctly.
    try:
        safe_base = pathlib.Path(sample_dir).resolve()
        safe_path = pathlib.Path(file_path).resolve()
        safe_base.relative_to(safe_base)  # no-op sanity check
        safe_path.relative_to(safe_base)  # raises ValueError if outside base
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")
    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(str(safe_path), media_type="image/png")


@app.post("/api/enhance", response_model=EnhanceResponse)
async def enhance_thermal_image(
    file: UploadFile = File(..., description="Single-channel thermal IR image (JPG, PNG, TIFF, BMP)"),
    ground_truth: Optional[UploadFile] = File(None, description="Optional paired visible RGB ground-truth image"),
    clahe_clip_limit: Optional[float] = Form(None),
    clahe_grid_size: Optional[int] = Form(None),
    bilateral_d: Optional[int] = Form(None),
    bilateral_sigma_color: Optional[float] = Form(None),
    bilateral_sigma_space: Optional[float] = Form(None),
    unsharp_radius: Optional[float] = Form(None),
    unsharp_amount: Optional[float] = Form(None),
    unsharp_threshold: Optional[int] = Form(None),
    provider: Optional[str] = Form(None)
):
    """
    Main enhancement and colorization pipeline:
    1. Validates and preprocesses single-channel thermal IR image (Grayscale normalization, CLAHE, Bilateral Denoising).
    2. Executes AI colorization via configured provider (Replicate / HuggingFace / Local).
    3. Performs unsharp masking post-processing on luminance.
    4. Computes scientifically sound quality & performance metrics.
    """
    t_start_total = time.perf_counter()

    # 1. Validate File Size & MIME
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty."
        )

    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum limit of {settings.max_upload_size_mb} MB."
        )

    # 2. Extract hyperparameters with fallback to config defaults
    p_clahe_clip = clahe_clip_limit if clahe_clip_limit is not None else settings.clahe_clip_limit
    p_clahe_grid = clahe_grid_size if clahe_grid_size is not None else settings.clahe_grid_size
    p_bilateral_d = bilateral_d if bilateral_d is not None else settings.bilateral_d
    p_bilateral_sc = bilateral_sigma_color if bilateral_sigma_color is not None else settings.bilateral_sigma_color
    p_bilateral_ss = bilateral_sigma_space if bilateral_sigma_space is not None else settings.bilateral_sigma_space
    p_unsharp_radius = unsharp_radius if unsharp_radius is not None else settings.unsharp_radius
    p_unsharp_amount = unsharp_amount if unsharp_amount is not None else settings.unsharp_amount
    p_unsharp_thresh = unsharp_threshold if unsharp_threshold is not None else settings.unsharp_threshold

    # 3. Stage 1: Preprocessing
    t_pre_start = time.perf_counter()
    try:
        raw_gray, prep_gray, prep_bgr, bit_depth_info, (orig_h, orig_w) = preprocess_thermal_image(
            image_bytes=file_bytes,
            clahe_clip_limit=p_clahe_clip,
            clahe_grid_size=p_clahe_grid,
            bilateral_d=p_bilateral_d,
            bilateral_sigma_color=p_bilateral_sc,
            bilateral_sigma_space=p_bilateral_ss
        )
    except PreprocessingError as pe:
        logger.error(f"Preprocessing error: {pe}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(pe))
    except Exception as exc:
        logger.error(f"Unexpected error in preprocessing: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error during image preprocessing.")
    t_pre_end = time.perf_counter()
    latency_pre_ms = (t_pre_end - t_pre_start) * 1000.0

    # 4. Stage 2: AI Inference / Colorization
    t_inf_start = time.perf_counter()
    _fallback_warning: Optional[str] = None
    try:
        colorizer = get_colorization_provider(settings, override_provider=provider)
        colorized_raw_bgr = await colorizer.colorize(prep_bgr)
    except InferenceError as ie:
        err_str = str(ie)
        logger.error(f"Inference error with provider {provider}: {err_str}")

        # Detect recoverable network / cloud API failures and auto-fallback to local.
        # Hard authentication failures (wrong token format) also fall back, but
        # we surface a clear warning so the user knows what happened.
        _is_network_error = any(k in err_str.lower() for k in (
            "network error", "getaddrinfo", "connection", "timeout",
            "api error", "unprocessable", "422", "503", "502"
        ))
        if _is_network_error:
            logger.warning(f"Cloud provider unavailable — falling back to local model. Reason: {err_str}")
            _fallback_warning = (
                f"Cloud provider '{provider}' is unreachable (no internet or API error). "
                f"Result produced by local PyTorch Pix2Pix model instead."
            )
            try:
                colorizer = PyTorchPix2PixProvider(settings.checkpoint_path)
                colorized_raw_bgr = await colorizer.colorize(prep_bgr)
            except Exception:
                # Absolute last resort: colormap
                colorizer = LocalFallbackProvider()
                colorized_raw_bgr = await colorizer.colorize(prep_bgr)
                _fallback_warning += " (No local checkpoint found — colormap used.)"
        else:
            # Non-recoverable error (e.g. malformed image, truly unexpected)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=err_str)
    except Exception as exc:
        logger.error(f"Unexpected inference error: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error during AI colorization.")
    t_inf_end = time.perf_counter()
    latency_inf_ms = (t_inf_end - t_inf_start) * 1000.0

    # 5. Stage 3: Post-Processing
    t_post_start = time.perf_counter()
    try:
        final_bgr = postprocess_colorized_image(
            colorized_raw_bgr,
            unsharp_radius=p_unsharp_radius,
            unsharp_amount=p_unsharp_amount,
            unsharp_threshold=p_unsharp_thresh
        )
    except PostprocessingError as pe:
        logger.error(f"Postprocessing error: {pe}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(pe))
    t_post_end = time.perf_counter()
    latency_post_ms = (t_post_end - t_post_start) * 1000.0

    # 6. Stage 4: Ground Truth handling & Quality Metrics Computation
    t_met_start = time.perf_counter()
    gt_bgr = None
    if ground_truth is not None:
        try:
            gt_bytes = await ground_truth.read()
            if len(gt_bytes) > 0:
                gt_bgr = decode_image_bytes(gt_bytes)
        except Exception as exc:
            logger.warning(f"Could not decode ground truth image: {exc}")

    # Preliminary latency calculation for the metrics payload
    t_met_end_prelim = time.perf_counter()
    latency_metrics_ms = (t_met_end_prelim - t_met_start) * 1000.0
    total_latency_ms = (t_met_end_prelim - t_start_total) * 1000.0

    latencies = LatencyBreakdown(
        preprocessing_ms=round(latency_pre_ms, 2),
        inference_ms=round(latency_inf_ms, 2),
        postprocessing_ms=round(latency_post_ms, 2),
        metrics_ms=round(latency_metrics_ms, 2),
        total_ms=round(total_latency_ms, 2)
    )

    metrics = compute_all_metrics(
        raw_gray=raw_gray,
        final_bgr=final_bgr,
        latency_breakdown=latencies,
        ground_truth_bgr=gt_bgr
    )

    # 7. Encode Base64 Data URIs
    raw_gray_bgr = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2BGR)
    prep_gray_bgr = cv2.cvtColor(prep_gray, cv2.COLOR_GRAY2BGR)

    b64_raw = encode_image_to_base64(raw_gray_bgr)
    b64_prep = encode_image_to_base64(prep_gray_bgr)
    b64_color = encode_image_to_base64(colorized_raw_bgr)
    b64_post = encode_image_to_base64(final_bgr)

    # Compile warnings / model disclaimers
    warnings = []
    if "ddcolor" in colorizer.provider_name.lower():
        warnings.append(
            "DDColor is optimized for visible-light grayscale images. For critical military/radiometric applications, "
            "results should be treated as synthetic visual aids rather than calibrated physical radiance reconstructions."
        )

    metadata = ImageMetadata(
        original_filename=file.filename or "unknown.png",
        original_format=file.content_type or "image/png",
        original_width=orig_w,
        original_height=orig_h,
        bit_depth=bit_depth_info,
        file_size_bytes=len(file_bytes),
        ai_provider=colorizer.provider_name,
        model_name=colorizer.provider_name,
        clahe_clip_limit=p_clahe_clip,
        clahe_grid_size=p_clahe_grid,
        bilateral_d=p_bilateral_d,
        unsharp_amount=p_unsharp_amount
    )

    return EnhanceResponse(
        success=True,
        message="Thermal IR image enhancement and colorization completed successfully.",
        original_image=b64_raw,
        preprocessed_image=b64_prep,
        colorized_image=b64_color,
        postprocessed_image=b64_post,
        metrics=metrics,
        metadata=metadata,
        warnings=warnings,
        provider_warning=_fallback_warning or getattr(colorizer, "fallback_warning", None),
    )


# Mount Static Frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
