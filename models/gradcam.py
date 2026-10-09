"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for PyTorch models.
Visualizes network attention regions influencing multi-label medical predictions.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.get_target_layer()

        self.gradients = None
        self.activations = None

        # Register forward and backward hooks on target layer
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates Grad-CAM heatmap for specified class index or top predicted class.
        Args:
            input_tensor: PyTorch tensor (1, 3, H, W)
            class_idx: int (0 to num_classes-1)
        Returns:
            heatmap: numpy array (H, W) normalized [0, 1]
            target_class_idx: int
        """
        # Enable gradient computation for backpropagation through model
        input_tensor.requires_grad_()
        
        # Forward pass
        probs, logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(probs[0]).item()

        # Zero existing gradients
        self.model.zero_grad()

        # Target class score for backprop
        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback synthetic heatmap if hooks failed
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5, class_idx

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)

        # Weighted sum of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # Shape: (1, 1, H_conv, W_conv)
        cam = F.relu(cam)  # Apply ReLU to focus on positive contributions

        # Interpolate heatmap to original input spatial dimensions
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)

        # Normalize heatmap to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx

    @staticmethod
    def overlay_heatmap(original_img_np, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlays normalized heatmap on top of RGB numpy image.
        """
        h, w = original_img_np.shape[:2]
        heatmap_resized = cv2.resize((heatmap * 255).astype(np.uint8), (w, h))

        # Colorize heatmap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Alpha blend original image and heatmap
        overlay = cv2.addWeighted(original_img_np.astype(np.uint8), 1 - alpha, heatmap_colored, alpha, 0)
        return overlay, heatmap_colored
