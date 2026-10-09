"""
PyTorch Neural Network Architectures for Medical Image Screening.
Supports EfficientNet-B0 and ResNet-50 backbones configured for multi-label screening.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DISEASE_CLASSES


class MedicalXRayClassifier(nn.Module):
    def __init__(self, architecture="efficientnet_b0", num_classes=len(DISEASE_CLASSES), pretrained=True):
        super(MedicalXRayClassifier, self).__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes

        if "efficientnet" in self.architecture:
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            
            # Replace classifier head for multi-label binary cross-entropy sigmoid output
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, num_classes)
            )
            self.target_layer_name = "features.7"

        elif "resnet" in self.architecture:
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            
            # Replace final FC layer
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )
            self.target_layer_name = "layer4"
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def get_target_layer(self):
        """Returns target convolutional layer for Grad-CAM activation mapping."""
        if "efficientnet" in self.architecture:
            return self.backbone.features[7]
        elif "resnet" in self.architecture:
            return self.backbone.layer4[2]
        return None

    def forward(self, x):
        logits = self.backbone(x)
        # Multi-label probability output via Sigmoid activation
        probabilities = torch.sigmoid(logits)
        return probabilities, logits


def get_model(architecture="efficientnet_b0"):
    """
    Factory function to initialize model instance with evaluation mode.
    """
    model = MedicalXRayClassifier(architecture=architecture, pretrained=True)
    model.eval()
    return model
