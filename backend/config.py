"""
Application Configuration and Settings Module.
Loads environment variables safely using Pydantic Settings and python-dotenv.
"""

from typing import Literal, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System settings for Thermal IR -> RGB Pix2Pix Translation & Dashboard."""

    # AI Provider Selection
    ai_provider: Literal["pytorch_pix2pix", "local"] = Field(
        default="pytorch_pix2pix",
        description="Active AI translation backend: pytorch_pix2pix (local trained model) or local (colormap)"
    )

    # PyTorch Model Checkpoint
    generator_checkpoint_path: str = Field(
        default="checkpoints/generator_best.pth",
        description="Path to trained PyTorch Pix2Pix UNetGenerator weights (.pth)"
    )
    inference_device: str = Field(
        default="cpu",
        description="Execution device for PyTorch model ('cuda' or 'cpu')"
    )


    # Server & Operational Settings
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")
    debug: bool = Field(default=False, description="Debug mode")
    request_timeout_seconds: float = Field(default=60.0, description="External inference HTTP timeout")

    # Security & Upload Validation
    max_upload_size_mb: int = Field(default=50, description="Maximum image upload size in Megabytes")
    allowed_mime_types: list[str] = Field(
        default_factory=lambda: [
            "image/jpeg",
            "image/png",
            "image/tiff",
            "image/webp",
            "image/bmp"
        ],
        description="Permitted input image MIME types"
    )

    # Preprocessing Hyperparameters
    clahe_clip_limit: float = Field(default=2.0, description="CLAHE contrast clip limit")
    clahe_grid_size: int = Field(default=8, description="CLAHE tile grid dimension (N x N)")
    bilateral_d: int = Field(default=9, description="Bilateral filter pixel neighborhood diameter")
    bilateral_sigma_color: float = Field(default=75.0, description="Bilateral filter filter sigma in color space")
    bilateral_sigma_space: float = Field(default=75.0, description="Bilateral filter filter sigma in coordinate space")

    # Postprocessing Hyperparameters
    unsharp_radius: float = Field(default=1.5, description="Gaussian blur radius for unsharp masking")
    unsharp_amount: float = Field(default=1.2, description="Sharpening magnification coefficient")
    unsharp_threshold: int = Field(default=3, description="Minimum contrast delta threshold for sharpening")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
