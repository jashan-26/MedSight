"""
Sample Chest X-Ray synthetic generator for offline demo and immediate testing.
Creates high-fidelity synthetic chest radiographs representing key pathology classes.
"""

import os
import cv2
import numpy as np
from PIL import Image

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))

def _create_base_chest_xray(width=512, height=512):
    """
    Renders realistic synthetic chest anatomical structures (ribs, lungs, heart shadow, clavicles).
    """
    img = np.zeros((height, width), dtype=np.float32)

    # 1. Soft background thoracic tissue padding
    cv2.ellipse(img, (width // 2, height // 2), (int(width * 0.45), int(height * 0.48)), 0, 0, 360, 40, -1)

    # 2. Lung fields (translucent dark cavities)
    # Left Lung (Right side on PA X-ray)
    cv2.ellipse(img, (int(width * 0.33), int(height * 0.48)), (int(width * 0.14), int(height * 0.32)), 0, 0, 360, 180, -1)
    # Right Lung
    cv2.ellipse(img, (int(width * 0.67), int(height * 0.48)), (int(width * 0.14), int(height * 0.32)), 0, 0, 360, 180, -1)

    # 3. Central Mediastinum & Cardiac Shadow
    cv2.ellipse(img, (int(width * 0.47), int(height * 0.52)), (int(width * 0.12), int(height * 0.20)), -15, 0, 360, 80, -1)

    # 4. Spine Column & Sternum
    cv2.rectangle(img, (int(width * 0.48), int(height * 0.1)), (int(width * 0.52), int(height * 0.9)), 70, -1)

    # 5. Rib cage arc structures
    for y_pos in range(int(height * 0.2), int(height * 0.8), 35):
        cv2.ellipse(img, (width // 2, y_pos), (int(width * 0.38), 25), 0, 180, 360, 110, 3)

    # 6. Clavicles (collarbones)
    cv2.line(img, (int(width * 0.15), int(height * 0.22)), (int(width * 0.45), int(height * 0.25)), 120, 6)
    cv2.line(img, (int(width * 0.85), int(height * 0.22)), (int(width * 0.55), int(height * 0.25)), 120, 6)

    # Gaussian blur & noise to simulate X-ray beam scatter & film grain
    img = cv2.GaussianBlur(img, (15, 15), 0)
    noise = np.random.normal(0, 8, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img


def generate_sample_images():
    """
    Generates X-ray sample files for each pathology condition.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    sample_specs = {
        "sample_normal.png": {"pathology": "Normal", "modifier": None},
        "sample_pneumonia.png": {"pathology": "Pneumonia", "modifier": "opacity"},
        "sample_atelectasis.png": {"pathology": "Atelectasis", "modifier": "linear_band"},
        "sample_cardiomegaly.png": {"pathology": "Cardiomegaly", "modifier": "enlarged_heart"},
        "sample_effusion.png": {"pathology": "Pleural Effusion", "modifier": "fluid_level"}
    }

    generated_paths = {}

    for filename, spec in sample_specs.items():
        filepath = os.path.join(SAMPLES_DIR, filename)
        base = _create_base_chest_xray()
        pathology = spec["pathology"]
        mod = spec["modifier"]

        if mod == "opacity":
            # Right lower lobe patchy infiltrate / consolidation (Pneumonia)
            overlay = np.zeros_like(base, dtype=np.uint8)
            cv2.circle(overlay, (340, 320), 45, 140, -1)
            cv2.circle(overlay, (370, 340), 30, 120, -1)
            overlay = cv2.GaussianBlur(overlay, (21, 21), 0)
            base = cv2.addWeighted(base, 1.0, overlay, 0.6, 0)

        elif mod == "linear_band":
            # Subsegmental linear opacity band in left lower zone (Atelectasis)
            overlay = np.zeros_like(base, dtype=np.uint8)
            cv2.ellipse(overlay, (180, 310), (50, 8), -25, 0, 360, 150, -1)
            overlay = cv2.GaussianBlur(overlay, (9, 9), 0)
            base = cv2.addWeighted(base, 1.0, overlay, 0.7, 0)

        elif mod == "enlarged_heart":
            # Significantly widened cardiac shadow (Cardiomegaly)
            overlay = np.zeros_like(base, dtype=np.uint8)
            cv2.ellipse(overlay, (240, 280), (105, 75), -15, 0, 360, 130, -1)
            overlay = cv2.GaussianBlur(overlay, (25, 25), 0)
            base = cv2.addWeighted(base, 1.0, overlay, 0.8, 0)

        elif mod == "fluid_level":
            # Right costophrenic angle blunting / fluid meniscus (Pleural Effusion)
            overlay = np.zeros_like(base, dtype=np.uint8)
            pts = np.array([[290, 380], [420, 360], [430, 440], [280, 440]], np.int32)
            cv2.fillPoly(overlay, [pts], 160)
            overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
            base = cv2.addWeighted(base, 1.0, overlay, 0.75, 0)

        # Convert to 3-channel RGB image
        rgb_img = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        Image.fromarray(rgb_img).save(filepath)
        generated_paths[pathology] = filepath

    return generated_paths


def get_sample_xrays():
    """
    Returns a dictionary of sample X-ray names and file paths.
    Generates them if missing.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    expected_files = [
        "sample_normal.png", "sample_pneumonia.png",
        "sample_atelectasis.png", "sample_cardiomegaly.png",
        "sample_effusion.png"
    ]
    
    missing = any(not os.path.exists(os.path.join(SAMPLES_DIR, f)) for f in expected_files)
    if missing:
        generate_sample_images()

    return {
        "Normal Chest X-Ray": os.path.join(SAMPLES_DIR, "sample_normal.png"),
        "Pneumonia (Right Lower Lobe Opacity)": os.path.join(SAMPLES_DIR, "sample_pneumonia.png"),
        "Atelectasis (Linear Basal Collapse)": os.path.join(SAMPLES_DIR, "sample_atelectasis.png"),
        "Cardiomegaly (Enlarged Cardiac Shadow)": os.path.join(SAMPLES_DIR, "sample_cardiomegaly.png"),
        "Pleural Effusion (Costophrenic Angle Blunting)": os.path.join(SAMPLES_DIR, "sample_effusion.png")
    }
