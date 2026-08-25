"""
Thermal IR Image Enhancement & Colorization System - Application Runner.
Executes the FastAPI server with Uvicorn.
"""

import os
import sys
import uvicorn
from backend.config import settings
from backend.generate_samples import generate_benchmark_samples


def main():
    print("=" * 70)
    print("  THERMAL VISION AI - ENHANCEMENT & COLORIZATION SYSTEM")
    print("=" * 70)
    print(f"  Active AI Provider   : {settings.ai_provider}")
    print(f"  Host / Port          : http://{settings.host}:{settings.port}")
    print(f"  Interactive Dashboard: http://localhost:{settings.port}")
    print(f"  API Documentation    : http://localhost:{settings.port}/docs")
    print("=" * 70)

    # Ensure benchmark sample images are generated
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    generate_benchmark_samples(sample_dir)

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
