"""
Medical Image Preprocessing Pipeline for Chest X-Ray Screening.
Handles border removal, CLAHE contrast enhancement, resizing, and PyTorch normalization.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import IMAGE_SIZE, MEAN_IMAGENET, STD_IMAGENET


def crop_black_borders(img_np, threshold=15):
    """
    Detects and crops uninformative dark/black borders around medical images.
    Uses thresholding and contour bounding box detection.
    """
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # Threshold dark areas
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img_np

    # Find largest contour (the patient anatomical region)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Ensure crop area is reasonably large (> 30% of image dimensions)
    img_h, img_w = gray.shape
    if w > 0.3 * img_w and h > 0.3 * img_h:
        cropped = img_np[y:y+h, x:x+w]
        return cropped
    
    return img_np


def apply_clahe(img_np, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance lung parenchyma opacities and subtle detail contrast.
    """
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)

    # Convert back to 3-channel RGB for downstream network processing
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb


class ImagePreprocessor:
    def __init__(self, target_size=IMAGE_SIZE):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
        ])

    def preprocess(self, pil_image, auto_crop=True, enable_clahe=False, brightness=1.0, contrast=1.0):
        """
        Full medical image preprocessing pipeline.
        Returns:
            - processed_pil: PIL.Image ready for display
            - processed_np: numpy array (H, W, 3)
            - tensor: PyTorch batch tensor (1, 3, H, W)
        """
        # Convert PIL to RGB numpy
        img_np = np.array(pil_image.convert("RGB"))

        # 1. Border handling / Cropping
        if auto_crop:
            img_np = crop_black_borders(img_np)

        # 2. CLAHE Contrast Enhancement
        if enable_clahe:
            img_np = apply_clahe(img_np)

        # 3. Manual Brightness & Contrast adjustment if requested
        if brightness != 1.0 or contrast != 1.0:
            img_np = cv2.convertScaleAbs(img_np, alpha=contrast, beta=(brightness - 1.0) * 50)

        # Convert back to PIL
        processed_pil = Image.fromarray(img_np)

        # 4. PyTorch Transformation & Normalization
        tensor = self.transform(processed_pil).unsqueeze(0)  # Shape: (1, 3, 224, 224)

        return processed_pil, img_np, tensor
