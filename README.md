# 🛰️ ThermalVision AI (IRIS)
### High-Performance Landsat Thermal IR $\to$ True-Color RGB Translation System

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenCV-4.9+-5C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Rasterio-1.3+-2E7D32.svg?style=for-the-badge&logo=geopandas&logoColor=white" alt="Rasterio">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Architecture-Conditional%20Pix2Pix%20GAN-blueviolet?style=for-the-badge" alt="Pix2Pix">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="MIT License">
</p>

---

## 📌 Overview

**ThermalVision AI (IRIS)** is an end-to-end, scientifically validated deep learning and computer vision system designed to translate single-band **Landsat 8/9 Collection 2 Level-2 Thermal Infrared imagery (Band 10 Surface Temperature, 10.6–11.19 µm)** into photorealistic **Visible True-Color RGB imagery (Bands 4, 3, 2 Surface Reflectance)**.

Built on a **Conditional Generative Adversarial Network (Pix2Pix U-Net Generator)** with seamless sliding-window 2D Hann window reconstruction, **ThermalVision AI** bridges the thermal-to-optical domain gap while preserving physical isotherms, structural boundaries, and radiometric fidelity.

### 🌟 Key Highlights
- **USGS Collection 2 Physical Radiometry:** Strict physical calibration ($DN \to \text{Kelvin} \to ^\circ\text{C}$ and $DN \to \text{Reflectance}$).
- **Zero Spatial Autocorrelation Leakage:** Enforces geographic **Scene-Level Partitioning** across train, validation, and test sets.
- **Seamless Tiled Inference:** 2D Hann window weighted blending eliminates seamline artifacts across large gigapixel satellite scenes.
- **Offline Local-First Architecture:** Self-contained inference engine requiring zero cloud subscriptions or API keys.
- **Reference & No-Reference Quality Metrics:** Quantitative evaluation suite including **PSNR**, **SSIM**, **Edge Preservation Index (EPI)**, **Tenengrad Focus Measure**, and **Shannon Entropy**.
- **Modern Mission-Control Dashboard:** Interactive browser dashboard featuring interactive split-comparison slider, 4-stage pipeline visualizer, real-time telemetry, and on-demand model switching.

---

## 🏗️ End-to-End System Pipeline

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                      Landsat 8/9 Collection 2 Level-2 GeoTIFF Products                 │
 │                      • Band 10: Thermal Surface Temperature (ST_B10.TIF)              │
 │                      • Bands 4, 3, 2: Surface Reflectance (SR_B4, SR_B3, SR_B2.TIF)    │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: Radiometric Physical Calibration & Quality Masking                            │
 │ • ST (Kelvin) = DN × 0.00341802 + 149.0                                                │
 │ • SR (Reflectance) = DN × 0.0000275 - 0.2                                              │
 │ • Invalid Fill & Saturated Pixel Rejection (DN = 0 → NaN)                             │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: Spatial Co-Registration & Geospatial Alignment                                │
 │ • Coordinate Reference System (CRS) & Projection Verification                          │
 │ • Affine Geotransform Matching                                                         │
 │ • Bilinear Reprojection to Shared 30m Grid Resolution                                  │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3: Preprocessing & Contrast Conditioning                                         │
 │ • CLAHE (Contrast Limited Adaptive Histogram Equalization)                             │
 │ • Bilateral Filtering (Edge-Preserving Thermal Noise Suppression)                      │
 │ • Normalization to Tensor Input Range [-1.0, 1.0]                                      │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 4: Pix2Pix Conditional GAN Neural Translation                                    │
 │ • Generator: 8-Layer U-Net with Skip Connections (in: 1 channel, out: 3 channels)      │
 │ • Discriminator: 70×70 PatchGAN Receptive Field                                        │
 │ • Loss Function: $\mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)$ ($\lambda=100$)             │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 5: Postprocessing & CIE-Lab Chroma Fusion                                        │
 │ • CIE-Lab Color Space Transformation                                                   │
 │ • High-Frequency Luminance (L*) Unsharp Mask Sharpening                                │
 │ • Chrominance (a*, b*) Edge-Bleed Prevention & Saturation Balance                      │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 6: Seamless Scene Stitching & Interactive Serving                                │
 │ • 2D Hann Window Overlap Blending across Tile Boundaries                               │
 │ • Full-Reference (PSNR/SSIM) & No-Reference (Tenengrad/Entropy/EPI) Telemetry          │
 │ • FastAPI REST Service & Interactive Dashboard UI                                      │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Scientific Foundations & Mathematics

### 1. USGS Radiometric Calibration Formulas
For Landsat 8/9 Collection 2 Level-2 science products:

$$\text{Surface Temperature } T\,(\text{Kelvin}) = \text{DN} \times 0.00341802 + 149.0$$

$$T\,(^\circ\text{C}) = T\,(\text{Kelvin}) - 273.15$$

$$\text{Surface Reflectance } \rho_{\lambda} = \text{DN} \times 0.0000275 - 0.2 \quad (\text{clipped to physical bounds } [0.0, 1.0])$$

### 2. Scene-Level Splitting (Preventing Spatial Leakage)
In satellite remote sensing, adjacent $256 \times 256$ spatial patches possess high spatial autocorrelation (Tobler's First Law of Geography). Splitting patches randomly results in identical geographic features appearing in both training and test sets, artificially inflating benchmark metrics. 

**ThermalVision AI** enforces strict **Scene-Level Geographic Partitioning**: entire independent Landsat scenes are segregated into `train/`, `val/`, and `test/` sets.

### 3. Seamless 2D Hann Window Overlap Blending
To eliminate tiling seam artifacts across large scenes ($> 4000 \times 4000$ pixels), the sliding-window engine uses a 2D cosine bell (Hann) weight matrix across overlapping tile windows:

$$W(x, y) = \sin^2\left(\frac{\pi x}{N}\right) \sin^2\left(\frac{\pi y}{N}\right), \quad \text{where } N = 256$$

The reconstructed pixel value is computed via normalized accumulated weighting:

$$I_{\text{stitched}}(x, y) = \frac{\sum_{i=1}^{K} I_i(x, y) \cdot W_i(x, y)}{\sum_{i=1}^{K} W_i(x, y)}$$

---

## 📁 Repository Structure

```text
IRIS/
├── backend/                       # FastAPI High-Performance Backend
│   ├── config.py                  # Pydantic system settings & hyperparameter configs
│   ├── schemas.py                 # Request / response serialization schemas
│   ├── preprocessing.py           # CLAHE, Bilateral filtering & base64 encoding
│   ├── postprocessing.py          # CIE-Lab unsharp masking & chrominance preservation
│   ├── metrics.py                 # PSNR, SSIM, Tenengrad, Shannon Entropy, EPI
│   ├── inference.py               # PyTorchPix2PixProvider & LocalFallbackProvider
│   └── main.py                    # REST endpoints, CORS & static frontend mount
│
├── frontend/                      # Interactive Mission Control Dashboard
│   ├── index.html                 # Semantic HTML5 layout with comparison tools
│   ├── styles.css                 # Dark glassmorphic styling & responsive grid
│   └── app.js                     # State management, split slider & telemetry graphs
│
├── src/                           # Core Machine Learning & Scientific Pipeline
│   ├── data/
│   │   ├── calibrate.py           # Radiometric calibration equations
│   │   ├── landsat_loader.py      # GeoTIFF loader & GeospatialMetadata container
│   │   ├── align.py               # Rasterio bilinear reprojection & CRS check
│   │   ├── patch_extractor.py     # 256x256 patch generation & scene-level split
│   │   ├── dataset.py             # PyTorch Dataset & DataLoader implementation
│   │   └── download_landsat.py    # USGS / Planetary Computer download harness
│   │
│   ├── models/
│   │   ├── generator.py           # 8-Layer U-Net Generator with skip connections
│   │   ├── discriminator.py       # 70x70 PatchGAN Discriminator
│   │   └── losses.py              # Composite Adversarial + L1 Loss ($L_{cGAN} + 100 \cdot L_1$)
│   │
│   ├── training/
│   │   └── train.py               # Mixed-precision (torch.amp) training loop
│   │
│   ├── evaluation/
│   │   └── evaluate.py            # Quantitative benchmark evaluation suite
│   │
│   └── inference/
│       └── predict.py             # Sliding-window GeoTIFF inference engine
│
├── checkpoints/                   # Trained model weights (generator_best.pth)
├── sample_images/                 # Real-world benchmark thermal imagery
├── tests/                         # Automated unit & integration tests
├── requirements.txt               # Pinned production dependencies
├── run.py                         # Application entrypoint & launcher
├── REPORT.md                      # Complete system audit & future roadmap
└── README.md                      # Project documentation
```

---

## ⚡ Quick Start Guide

### 1. Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/IRIS.git
cd IRIS

# 2. Create isolated virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation & Run Test Suite

```bash
python -m pytest tests/ -v
```

### 3. Launch Mission Control Dashboard

```bash
python run.py
```
Open your browser and navigate to:
- **Interactive Mission Control:** [http://localhost:8000](http://localhost:8000)
- **OpenAPI Interactive Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💻 CLI & Training Workflows

### Train the Pix2Pix Model
```bash
# Fast pipeline smoke test (2 epochs)
python -m src.training.train --smoke-test

# Full multi-epoch GPU training with mixed precision (AMP)
python -m src.training.train --epochs 50 --batch-size 16 --lr 0.0002 --device cuda
```

### Evaluate Model Against Benchmark Test Set
```bash
python -m src.evaluation.evaluate --checkpoint checkpoints/generator_best.pth --data-dir data
```

### Full-Scene GeoTIFF Inference
```bash
python -m src.inference.predict \
  --input data/raw/sample_ST_B10.TIF \
  --output outputs/translated_scene_rgb.png \
  --checkpoint checkpoints/generator_best.pth
```

---

## 📊 Scientific Quality & Benchmark Metrics

| Metric | Target Baseline | Description |
|---|:---:|---|
| **PSNR (dB)** | $> 26.0\text{ dB}$ | Peak Signal-to-Noise Ratio measuring pixel-level RGB fidelity against ground truth. |
| **SSIM** | $> 0.82$ | Structural Similarity Index capturing structural texture, luminance, and contrast. |
| **Edge Preservation Index (EPI)** | $> 0.88$ | High-pass Laplacian correlation measuring boundary retention across isotherms. |
| **Tenengrad Gradient Score** | Higher | Quantifies gradient focus and sharpness across thermal transitions. |
| **Shannon Entropy** | $> 7.2\text{ bits}$ | Measures information density and dynamic range preservation. |

---

## 📡 REST API Reference

### `POST /api/enhance`
Enhance and colorize a thermal infrared image through the 4-stage pipeline.

**Request:** `multipart/form-data`
- `file`: Raw image binary (JPEG, PNG, TIFF, WebP, BMP up to 15MB)
- `provider`: Translation engine (`pytorch_pix2pix` or `local`)
- `clahe_clip_limit`: Contrast clip limit (default: `2.0`)
- `bilateral_d`: Denoising diameter (default: `9`)
- `unsharp_amount`: High-frequency sharpening strength (default: `1.2`)

**Example `cURL` Request:**
```bash
curl -X POST "http://localhost:8000/api/enhance" \
  -F "file=@sample_images/01_nocturnal_surveillance.png" \
  -F "provider=pytorch_pix2pix" \
  -F "clahe_clip_limit=2.0" \
  -F "unsharp_amount=1.2"
```

**Response Format:**
```json
{
  "success": true,
  "message": "Thermal IR image enhancement and colorization completed successfully.",
  "original_image": "data:image/png;base64,...",
  "preprocessed_image": "data:image/png;base64,...",
  "colorized_image": "data:image/png;base64,...",
  "postprocessed_image": "data:image/png;base64,...",
  "metrics": {
    "tenengrad_raw": 34.2,
    "tenengrad_enhanced": 89.6,
    "tenengrad_gain_percent": 161.9,
    "entropy_raw": 6.84,
    "entropy_enhanced": 7.52,
    "edge_preservation_index": 0.91,
    "latency": {
      "preprocess_ms": 4.1,
      "inference_ms": 48.2,
      "postprocess_ms": 3.8,
      "total_ms": 56.1
    }
  },
  "metadata": {
    "original_width": 256,
    "original_height": 256,
    "ai_provider": "pytorch-pix2pix (cpu)"
  }
}
```

---

## 📜 License & Acknowledgments

This project is licensed under the **MIT License**.  
Developed for remote sensing, defense intelligence, environmental monitoring, and earth observation research.

- **USGS / NASA Landsat Program** for Collection 2 Level-2 Surface Temperature & Reflectance products.
- **Isola et al. (2017)** for the foundational Pix2Pix Image-to-Image Translation framework.
