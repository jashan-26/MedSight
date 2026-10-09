"""
Models package for Chest X-Ray screening classifiers and explainability hooks.
"""
from .networks import MedicalXRayClassifier, get_model
from .gradcam import GradCAM
from .calibration import compute_confidence_calibration, compute_temperature_scaling, estimate_uncertainty
from .model_manager import ModelManager

__all__ = [
    "MedicalXRayClassifier",
    "get_model",
    "GradCAM",
    "compute_confidence_calibration",
    "compute_temperature_scaling",
    "estimate_uncertainty",
    "ModelManager"
]

