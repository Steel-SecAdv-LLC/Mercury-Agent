# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Feature extraction backbones for visual anomaly detection.

Provides unified interface for extracting multi-scale features from
pre-trained networks including ResNet, WideResNet, EfficientNet, ViT, and DINOv2.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Backbone configuration: output channels for each layer
BACKBONE_CONFIGS: dict[str, dict[str, Any]] = {
    "resnet18": {
        "layer1": 64,
        "layer2": 128,
        "layer3": 256,
        "layer4": 512,
    },
    "resnet50": {
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048,
    },
    "wide_resnet50_2": {
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048,
    },
    "efficientnet_b4": {
        "layer1": 32,
        "layer2": 56,
        "layer3": 160,
        "layer4": 448,
    },
}


def get_backbone(
    backbone_name: str,
    pretrained: bool = True,
) -> nn.Module:
    """Get a pre-trained backbone network.

    Args:
        backbone_name: Name of backbone architecture
        pretrained: Whether to load ImageNet pre-trained weights

    Returns:
        Pre-trained backbone module

    Raises:
        ValueError: If backbone_name is not supported
    """
    try:
        from torchvision import models
    except ImportError as e:
        raise ImportError("torchvision is required for backbone models") from e

    weights = "IMAGENET1K_V1" if pretrained else None

    backbone_map = {
        "resnet18": lambda: models.resnet18(weights=weights),
        "resnet50": lambda: models.resnet50(weights=weights),
        "wide_resnet50_2": lambda: models.wide_resnet50_2(weights=weights),
    }

    if backbone_name in backbone_map:
        return backbone_map[backbone_name]()

    # Try timm for additional architectures
    try:
        import timm

        if backbone_name in timm.list_models():
            return timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
            )
    except ImportError:
        logger.warning("timm not available for extended backbone support")

    raise ValueError(
        f"Unsupported backbone: {backbone_name}. " f"Supported: {list(backbone_map.keys())}"
    )


class FeatureExtractor(nn.Module):
    """Multi-scale feature extractor from pre-trained backbones.

    Extracts features from multiple intermediate layers of a backbone
    network for use in anomaly detection algorithms.

    Attributes:
        backbone_name: Name of the backbone architecture
        layers: List of layer names to extract features from
        feature_dims: Output dimensions for each extracted layer

    Example:
        >>> extractor = FeatureExtractor('wide_resnet50_2', ['layer2', 'layer3'])
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features = extractor(images)
        >>> print({k: v.shape for k, v in features.items()})
        {'layer2': torch.Size([4, 512, 28, 28]), 'layer3': torch.Size([4, 1024, 14, 14])}
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: list[str] | None = None,
        device: torch.device | str = "cpu",
        pretrained: bool = True,
    ):
        """Initialize feature extractor.

        Args:
            backbone_name: Pre-trained backbone architecture name
            layers: Layers to extract features from (default: layer2, layer3)
            device: Computation device
            pretrained: Whether to use ImageNet pre-trained weights
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.layers = layers or ["layer2", "layer3"]
        self.device = torch.device(device)

        # Load backbone
        self.backbone = get_backbone(backbone_name, pretrained=pretrained)
        self.backbone.eval()

        # Get layer info
        if backbone_name in BACKBONE_CONFIGS:
            self.feature_dims = {
                layer: BACKBONE_CONFIGS[backbone_name].get(layer, 512) for layer in self.layers
            }
        else:
            # Infer dimensions from forward pass
            self.feature_dims = self._infer_dims()

        # Register hooks for feature extraction
        self._features: dict[str, torch.Tensor] = {}
        self._register_hooks()

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.to(self.device)

    def _register_hooks(self) -> None:
        """Register forward hooks to capture intermediate features."""
        for layer_name in self.layers:
            layer = self._get_layer(layer_name)
            if layer is not None:
                layer.register_forward_hook(self._hook_fn(layer_name))
            else:
                logger.warning(f"Layer {layer_name} not found in backbone")

    def _get_layer(self, layer_name: str) -> nn.Module | None:
        """Get a layer from the backbone by name."""
        try:
            return getattr(self.backbone, layer_name)
        except AttributeError:
            # Try nested access
            parts = layer_name.split(".")
            module = self.backbone
            for part in parts:
                module = getattr(module, part, None)  # type: ignore[assignment, unused-ignore]
                if module is None:
                    return None
            return module

    def _hook_fn(self, layer_name: str) -> Any:
        """Create a hook function for a specific layer."""

        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self._features[layer_name] = output

        return hook

    def _infer_dims(self) -> dict[str, int]:
        """Infer feature dimensions from a forward pass."""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.backbone(dummy_input)
        return {name: feat.shape[1] for name, feat in self._features.items()}

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Extract multi-scale features from input images.

        Args:
            x: Input images [B, 3, H, W]
            return_dict: If True, return dict of layer features; else concatenate

        Returns:
            Dict mapping layer names to feature tensors, or concatenated features
        """
        self._features.clear()

        # Forward through backbone
        with torch.no_grad():
            _ = self.backbone(x)

        if return_dict:
            return {k: v.clone() for k, v in self._features.items()}

        # Concatenate features at same spatial resolution
        features_list = list(self._features.values())
        if len(features_list) == 1:
            return features_list[0]

        # Resize all features to smallest spatial size
        min_h = min(f.shape[2] for f in features_list)
        min_w = min(f.shape[3] for f in features_list)

        resized = []
        for feat in features_list:
            if feat.shape[2] != min_h or feat.shape[3] != min_w:
                feat = nn.functional.interpolate(
                    feat, size=(min_h, min_w), mode="bilinear", align_corners=False
                )
            resized.append(feat)

        return torch.cat(resized, dim=1)

    def get_total_dim(self) -> int:
        """Get total feature dimension when concatenating all layers."""
        return sum(self.feature_dims.values())
