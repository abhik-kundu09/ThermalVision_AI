"""
Data schemas and serialization models for the Thermal IR Enhancement API.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class LatencyBreakdown(BaseModel):
    """Execution timing profile across individual pipeline stages in milliseconds."""
    preprocessing_ms: float = Field(..., description="Local decoding, grayscale & filtering latency in ms")
    inference_ms: float = Field(..., description="Cloud or local AI colorization model latency in ms")
    postprocessing_ms: float = Field(..., description="Unsharp masking and refinement latency in ms")
    metrics_ms: float = Field(..., description="Image quality metrics computation latency in ms")
    total_ms: float = Field(..., description="Total end-to-end request latency in ms")


class ImageMetrics(BaseModel):
    """Scientifically validated image quality & performance metrics."""
    latency: LatencyBreakdown
    # No-Reference Metrics
    tenengrad_sharpness_input: float = Field(..., description="Tenengrad gradient sharpness of raw input")
    tenengrad_sharpness_output: float = Field(..., description="Tenengrad gradient sharpness of enhanced output")
    shannon_entropy_input: float = Field(..., description="Information entropy of input thermal image in bits")
    shannon_entropy_output: float = Field(..., description="Information entropy of colorized output in bits")
    rms_contrast_input: float = Field(..., description="Root Mean Square contrast index of input")
    rms_contrast_output: float = Field(..., description="Root Mean Square contrast index of output")
    edge_preservation_index: float = Field(..., description="Edge Preservation Index (EPI) between input and output")

    # Full-Reference Metrics (Only populated if paired ground-truth RGB is provided)
    has_ground_truth: bool = Field(default=False, description="Whether paired ground truth RGB was provided")
    psnr: Optional[float] = Field(default=None, description="Peak Signal-to-Noise Ratio (dB) against ground-truth")
    ssim: Optional[float] = Field(default=None, description="Structural Similarity Index against ground-truth")
    reference_disclaimer: Optional[str] = Field(
        default=None,
        description="Scientific explanation for reference-dependent metric calculation status"
    )


class ImageMetadata(BaseModel):
    """Metadata describing the input thermal image and processing session."""
    original_filename: str
    original_format: str
    original_width: int
    original_height: int
    bit_depth: str
    file_size_bytes: int
    ai_provider: str
    model_name: str
    clahe_clip_limit: float
    clahe_grid_size: int
    bilateral_d: int
    unsharp_amount: float


class EnhanceResponse(BaseModel):
    """Primary response schema returned by the /api/enhance endpoint."""
    success: bool
    message: str
    original_image: str = Field(..., description="Base64 data URI of normalized thermal input")
    preprocessed_image: str = Field(..., description="Base64 data URI of CLAHE + Bilateral filtered image")
    colorized_image: str = Field(..., description="Base64 data URI of raw AI colorization output")
    postprocessed_image: str = Field(..., description="Base64 data URI of final sharpened & colorized image")
    metrics: ImageMetrics
    metadata: ImageMetadata
    warnings: List[str] = Field(default_factory=list)
    provider_warning: Optional[str] = Field(
        default=None,
        description=(
            "Set when the system fell back to the local colormap provider instead of an AI model. "
            "Results may not reflect genuine AI colorization."
        )
    )


class HealthResponse(BaseModel):
    """API health and service telemetry schema."""
    status: str
    version: str
    active_provider: str
    provider_configured: bool
    opencv_version: str
    features: Dict[str, Any]


class SampleImageInfo(BaseModel):
    """Information for preloaded thermal sample datasets."""
    id: str
    title: str
    category: str
    description: str
    dimensions: str
    sensor_type: str
    filename: str
