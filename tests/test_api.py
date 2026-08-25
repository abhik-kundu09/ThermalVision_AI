"""
Integration Tests for FastAPI Endpoints.
Uses httpx / Starlette TestClient to validate route contracts, status codes, and payloads.
"""

import io
import os
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from backend.main import app

# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def create_test_image_bytes():
    """Generates valid PNG image bytes (120x160 grayscale)."""
    img = np.random.randint(40, 220, (120, 160), dtype=np.uint8)
    _, buffer = cv2.imencode(".png", img)
    return io.BytesIO(buffer.tobytes())


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    """Test GET /api/health."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "opencv_version" in data
    assert "active_provider" in data


def test_sample_images_endpoint(client):
    """Test GET /api/sample-images and retrieval of individual samples."""
    response = client.get("/api/sample-images")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 6
    assert samples[0]["filename"].endswith(".png")

    # Fetch first sample image binary
    first_sample = samples[0]["filename"]
    img_resp = client.get(f"/api/sample-images/{first_sample}")
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"


# Determine checkpoint path for conditional skip
_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "checkpoints", "generator_best.pth"
)


@pytest.mark.skipif(
    not os.path.exists(_CHECKPOINT_PATH),
    reason=(
        "Trained checkpoint 'checkpoints/generator_best.pth' not found. "
        "Run training or download weights before running this test."
    ),
)
def test_enhance_endpoint_pix2pix_success(client):
    """Test POST /api/enhance with local PyTorch Pix2Pix provider (requires trained checkpoint)."""
    img_io = create_test_image_bytes()
    files = {"file": ("test_thermal.png", img_io, "image/png")}
    data = {
        "provider": "pytorch_pix2pix",
        "clahe_clip_limit": "2.0",
        "clahe_grid_size": "8",
        "bilateral_d": "9",
        "unsharp_amount": "1.2"
    }

    response = client.post("/api/enhance", files=files, data=data)
    assert response.status_code == 200

    resp_data = response.json()
    assert resp_data["success"] is True
    assert "data:image/png;base64," in resp_data["original_image"]
    assert "data:image/png;base64," in resp_data["postprocessed_image"]
    assert resp_data["metrics"]["latency"]["total_ms"] > 0
    assert resp_data["metadata"]["original_width"] == 160
    assert resp_data["metadata"]["original_height"] == 120
    # provider_warning should be None when using the real Pix2Pix model
    assert resp_data["provider_warning"] is None


def test_enhance_endpoint_local_colormap_has_no_provider_warning(client):
    """Test POST /api/enhance with explicit local colormap — no warning expected."""
    img_io = create_test_image_bytes()
    files = {"file": ("test_thermal.png", img_io, "image/png")}
    data = {"provider": "local"}

    response = client.post("/api/enhance", files=files, data=data)
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["success"] is True
    # Explicitly choosing 'local' is intentional — no provider_warning
    assert resp_data["provider_warning"] is None


def test_enhance_endpoint_empty_file_rejection(client):
    """Test POST /api/enhance rejecting empty uploads with HTTP 400."""
    files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
    response = client.post("/api/enhance", files=files)
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "empty" in detail.lower()


def test_enhance_endpoint_corrupted_file_rejection(client):
    """Test POST /api/enhance rejecting corrupt non-image bytes with HTTP 400."""
    files = {"file": ("corrupt.png", io.BytesIO(b"this is not an image at all"), "image/png")}
    response = client.post("/api/enhance", files=files)
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()
