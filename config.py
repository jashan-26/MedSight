"""
Configuration constants and settings for Medical Image Screening Assistant.
"""

# Pathologies and target classes
DISEASE_CLASSES = [
    "Normal",
    "Pneumonia",
    "Atelectasis",
    "Cardiomegaly",
    "Pleural Effusion"
]

# Clinical risk mapping based on primary disease finding and confidence
RISK_LEVELS = {
    "Normal": "LOW",
    "Atelectasis": "MODERATE",
    "Pneumonia": "HIGH",
    "Pleural Effusion": "HIGH",
    "Cardiomegaly": "HIGH"
}

# Anatomical attention regions for X-ray Findings (for explainable AI descriptions)
ANATOMICAL_MAP = {
    "Normal": "Clear lung fields, sharp costophrenic angles, normal cardiac silhouette.",
    "Pneumonia": "Consolidation and focal opacity localized in lower/mid lung zones.",
    "Atelectasis": "Linear opacity and partial lobar collapse in basal lung segments.",
    "Cardiomegaly": "Cardiothoracic ratio > 0.5 with widening of the cardiac silhouette.",
    "Pleural Effusion": "Blunting of costophrenic angle and dependent pleural cavity fluid accumulation."
}

# Decision & Confidence Thresholds
DEFAULT_THRESHOLD = 0.5
UNCERTAINTY_THRESHOLD_LOW = 0.60
UNCERTAINTY_THRESHOLD_HIGH = 0.85

# Model Options
AVAILABLE_MODELS = {
    "EfficientNet-B0": {
        "architecture": "efficientnet_b0",
        "params": "5.3M",
        "target_layer": "features.7",
        "description": "Lightweight, high-efficiency convolutional neural network optimized for clinical edge triage."
    },
    "ResNet-50": {
        "architecture": "resnet50",
        "params": "25.6M",
        "target_layer": "layer4",
        "description": "Deep residual architecture with rich spatial feature representations across complex lung parenchymal opacities."
    }
}

# Image Input Specifications
IMAGE_SIZE = (224, 224)
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

# Mandatory Medical Disclaimer
MEDICAL_DISCLAIMER = """
**LEGAL & CLINICAL DISCLAIMER:**
This software is an AI research, triage, and decision-support prototype.
It is **NOT** a certified medical diagnostic device and MUST NOT be used as a replacement for professional radiological evaluation, physician diagnosis, or direct clinical decision-making. All predictions, heatmaps, and confidence scores require independent verification by a licensed healthcare professional.
"""
