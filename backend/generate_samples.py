"""
Realistic Remote Sensing Benchmark Thermal IR Dataset Generator.
Generates photorealistic satellite, aerial, and industrial thermograms:
1. Landsat 8/9 Urban Heat Island (Metropolitan heat clusters & street grids)
2. Agricultural Crop Canopy Thermal Stress (Farmland parcels & center-pivot irrigation)
3. Coastal Estuary Hydro-Thermal Plume (River-ocean mixing isotherms)
4. Volcanic Caldera Geothermal Flux (Geothermal fissures & heat radiance)
5. UAV Airborne Reconnaissance Corridor (Aerial transport corridor)
6. Industrial Thermal Plant Diagnostics (Power generation heat dissipation)
"""

import os
import cv2
import numpy as np


def generate_fractal_noise(h: int, w: int, octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
    """Generates smooth fractal Perlin-like 2D noise."""
    noise = np.zeros((h, w), dtype=np.float32)
    freq = 1.0
    amp = 1.0
    for _ in range(octaves):
        small_h, small_w = max(4, int(h / (8 / freq))), max(4, int(w / (8 / freq)))
        rand = np.random.uniform(0, 1, (small_h, small_w)).astype(np.float32)
        upsampled = cv2.resize(rand, (w, h), interpolation=cv2.INTER_CUBIC)
        noise += upsampled * amp
        amp *= persistence
        freq *= 2.0
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    return noise


def generate_benchmark_samples(output_dir: str = "sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    w, h = 640, 480

    # -------------------------------------------------------------------------
    # 1. Landsat 8/9 Urban Heat Island (640x480)
    # -------------------------------------------------------------------------
    terrain = generate_fractal_noise(h, w, octaves=4)
    urban_ir = (terrain * 80 + 40).astype(np.float32)

    # Road network & high-density asphalt heat corridors
    for y in range(60, h, 70):
        cv2.line(urban_ir, (0, y), (w, y), 170.0, 3)
    for x in range(80, w, 90):
        cv2.line(urban_ir, (x, 0), (x, h), 165.0, 3)

    # High-rise commercial building roofs (intense thermal heat retention)
    np.random.seed(42)
    for _ in range(35):
        bx = np.random.randint(40, w - 80)
        by = np.random.randint(40, h - 80)
        bw = np.random.randint(30, 60)
        bh = np.random.randint(30, 60)
        heat_level = np.random.uniform(190, 245)
        cv2.rectangle(urban_ir, (bx, by), (bx + bw, by + bh), heat_level, -1)

    # Cool urban park & water reservoir
    cv2.circle(urban_ir, (180, 240), 55, 35.0, -1)
    cv2.ellipse(urban_ir, (460, 360), (90, 45), -20, 0, 360, 38.0, -1)
    urban_ir = cv2.GaussianBlur(urban_ir, (5, 5), 1.5)
    noise = np.random.normal(0, 2.5, (h, w)).astype(np.float32)
    img1 = np.clip(urban_ir + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "01_nocturnal_surveillance.png"), img1)

    # -------------------------------------------------------------------------
    # 2. Agricultural Crop Canopy Thermal Stress (640x480)
    # -------------------------------------------------------------------------
    agri_ir = np.full((h, w), 80, dtype=np.float32)
    # Field parcel grid
    for r in range(0, h, 80):
        for c in range(0, w, 110):
            # Crop transpiration variation: well-watered (cool ~45-70) vs drought-stressed (warm ~140-190)
            parcel_temp = np.random.uniform(50, 185)
            cv2.rectangle(agri_ir, (c + 4, r + 4), (c + 106, r + 76), parcel_temp, -1)

    # Center-pivot circular irrigation systems (cool core, warm dry perimeter)
    for center_x, center_y, radius in [(220, 180, 65), (450, 280, 75), (100, 380, 50)]:
        for rad in range(radius, 0, -5):
            val = 40 + (radius - rad) * 0.8
            cv2.circle(agri_ir, (center_x, center_y), rad, val, -1)

    agri_ir = cv2.GaussianBlur(agri_ir, (5, 5), 1.2)
    noise2 = np.random.normal(0, 2.0, (h, w)).astype(np.float32)
    img2 = np.clip(agri_ir + noise2, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "02_electrical_substation_hotspot.png"), img2)

    # -------------------------------------------------------------------------
    # 3. Coastal Estuary Hydro-Thermal Plume (640x480)
    # -------------------------------------------------------------------------
    ocean_noise = generate_fractal_noise(h, w, octaves=3)
    coastal_ir = (ocean_noise * 30 + 35).astype(np.float32)  # Cold coastal sea (35-65)

    # Landmass
    land_mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[0, 0], [260, 0], [210, 180], [320, 300], [180, 480], [0, 480]], np.int32)
    cv2.fillPoly(land_mask, [pts], 255)
    coastal_ir[land_mask > 0] = (terrain[land_mask > 0] * 70 + 130)  # Warm land (130-200)

    # Warm river estuary discharge mixing into ocean with fluid plume
    for t in range(0, 220, 10):
        plume_x = int(220 + t * 1.5 + np.sin(t * 0.05) * 30)
        plume_y = int(200 + t * 0.8 + np.cos(t * 0.04) * 20)
        plume_rad = int(20 + t * 0.5)
        temp = max(55.0, 160.0 - t * 0.5)
        cv2.circle(coastal_ir, (plume_x, plume_y), plume_rad, temp, -1)

    coastal_ir = cv2.GaussianBlur(coastal_ir, (11, 11), 3.5)
    noise3 = np.random.normal(0, 2.0, (h, w)).astype(np.float32)
    img3 = np.clip(coastal_ir + noise3, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "03_wildlife_canopy_recon.png"), img3)

    # -------------------------------------------------------------------------
    # 4. Volcanic Caldera Geothermal Flux (640x480)
    # -------------------------------------------------------------------------
    volcano_noise = generate_fractal_noise(h, w, octaves=5)
    geo_ir = (volcano_noise * 50 + 55).astype(np.float32)

    # Central volcanic crater (high geothermal radiance)
    center = (320, 240)
    for r in range(120, 0, -6):
        flux = 60.0 + (120 - r) * 1.5
        cv2.circle(geo_ir, center, r, flux, -1)

    # Magma fissures & glowing thermal vents (220-255)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ex = int(center[0] + np.cos(angle) * 110 + np.random.uniform(-15, 15))
        ey = int(center[1] + np.sin(angle) * 110 + np.random.uniform(-15, 15))
        cv2.line(geo_ir, center, (ex, ey), 245.0, 4)
        cv2.circle(geo_ir, (ex, ey), 15, 230.0, -1)

    geo_ir = cv2.GaussianBlur(geo_ir, (7, 7), 2.0)
    noise4 = np.random.normal(0, 3.0, (h, w)).astype(np.float32)
    img4 = np.clip(geo_ir + noise4, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "04_building_heat_loss.png"), img4)

    # -------------------------------------------------------------------------
    # 5. UAV Aerial Reconnaissance Corridor (640x480)
    # -------------------------------------------------------------------------
    corridor_ir = (generate_fractal_noise(h, w, octaves=3) * 45 + 50).astype(np.float32)
    # Diagonal dual-carriageway highway (warm concrete)
    cv2.line(corridor_ir, (0, 420), (640, 80), 160.0, 18)
    cv2.line(corridor_ir, (0, 420), (640, 80), 120.0, 2)  # Median barrier

    # Vehicles with hot engine blocks & exhaust heat plumes
    for px, py in [(140, 345), (280, 270), (420, 195), (510, 145)]:
        cv2.circle(corridor_ir, (px, py), 10, 245.0, -1)
        cv2.line(corridor_ir, (px, py), (px - 25, py + 14), 180.0, 4)  # Heat plume

    # Industrial logistics hangars alongside corridor
    for hx, hy, hw, hh in [(80, 140, 90, 70), (380, 340, 110, 60)]:
        cv2.rectangle(corridor_ir, (hx, hy), (hx + hw, hy + hh), 175.0, -1)

    corridor_ir = cv2.GaussianBlur(corridor_ir, (3, 3), 1.0)
    noise5 = np.random.normal(0, 2.5, (h, w)).astype(np.float32)
    img5 = np.clip(corridor_ir + noise5, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "05_uav_convoy_recon.png"), img5)

    # -------------------------------------------------------------------------
    # 6. Industrial Thermal Plant Diagnostics (640x480)
    # -------------------------------------------------------------------------
    plant_ir = np.full((h, w), 55, dtype=np.float32)
    # Structural conduits and pipe racks
    for y in [120, 220, 320]:
        cv2.line(plant_ir, (60, y), (580, y), 140.0, 8)
    for x in [160, 320, 480]:
        cv2.line(plant_ir, (x, 80), (x, 400), 130.0, 6)

    # High-temperature steam turbines & heat exchangers
    for tx, ty in [(240, 180), (400, 180), (240, 280), (400, 280)]:
        cv2.circle(plant_ir, (tx, ty), 35, 235.0, -1)
        cv2.circle(plant_ir, (tx, ty), 45, 180.0, 4)

    plant_ir = cv2.GaussianBlur(plant_ir, (5, 5), 1.5)
    noise6 = np.random.normal(0, 2.0, (h, w)).astype(np.float32)
    img6 = np.clip(plant_ir + noise6, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "06_turbine_radiometric_diagnostic.png"), img6)

    print(f"Successfully generated 6 realistic Landsat & thermal benchmark datasets in {output_dir}")


if __name__ == "__main__":
    generate_benchmark_samples("sample_images")
