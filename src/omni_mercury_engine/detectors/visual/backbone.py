# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Feature extraction backbones for visual anomaly detection.

Provides unified interface for extracting multi-scale features from
pre-trained networks including ResNet, WideResNet, EfficientNet, and ViT.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

#: Frozen pretrained backbones, keyed by ``(name, pretrained, timm out-indices)``,
#: so repeated detector construction reuses the loaded weights instead of
#: re-reading them from disk.  Every cache hit returns a ``deepcopy``, so no
#: training state can leak between instances; only ``pretrained=True`` models
#: are cached, and trainable students bypass the cache entirely (they are
#: built from scratch and must never share weights with a frozen teacher).
_BACKBONE_CACHE: dict[tuple[str, bool, tuple[int, ...] | None], nn.Module] = {}

#: timm ``features_only`` stage index for each named layer of the supported
#: timm backbones.  efficientnet_b4's stages 1-4 carry the channel counts
#: documented in ``BACKBONE_CONFIGS``; the ViT has no pyramid, so layer1-4
#: name four evenly spaced transformer blocks (all 768 channels at 1/16).
_TIMM_FEATURE_INDICES: dict[str, dict[str, int]] = {
    "efficientnet_b4": {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4},
    "vit_base_patch16_224": {"layer1": 2, "layer2": 5, "layer3": 8, "layer4": 11},
}

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


class TimmFeatureBackbone(nn.Module):
    """A timm ``features_only`` model behind the named layer-dict contract.

    timm feature extractors return a *list* of feature maps and have no
    ``layer1``-``layer4`` attributes, so the hook-based extraction used for
    torchvision ResNets silently produced an empty feature dict (and an
    ``IndexError`` downstream).  This wrapper maps each supported layer name
    to its real timm feature stage (``_TIMM_FEATURE_INDICES``) and returns a
    ``{layer_name: tensor}`` dict from ``forward``.
    """

    def __init__(self, backbone_name: str, layers: list[str], pretrained: bool) -> None:
        """Build the timm feature model for the requested layers.

        Args:
            backbone_name: timm model name (must be in ``_TIMM_FEATURE_INDICES``)
            layers: Layer names (``layer1``..``layer4``) to serve
            pretrained: Whether to load pretrained weights

        Raises:
            ValueError: If a requested layer has no mapped feature stage.
        """
        super().__init__()
        import timm

        index_map = _TIMM_FEATURE_INDICES[backbone_name]
        unknown = [layer for layer in layers if layer not in index_map]
        if unknown:
            raise ValueError(
                f"Backbone {backbone_name} has no feature stage for {unknown}; "
                f"supported layer names: {sorted(index_map)}"
            )
        out_indices = sorted({index_map[layer] for layer in layers})
        model: Any = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.model = model
        self._positions = {layer: out_indices.index(index_map[layer]) for layer in layers}
        channels = model.feature_info.channels()
        self.feature_dims: dict[str, int] = {
            layer: int(channels[position]) for layer, position in self._positions.items()
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the named multi-scale feature dict for ``x``."""
        maps = self.model(x)
        return {layer: maps[position] for layer, position in self._positions.items()}


def _build_backbone(
    backbone_name: str,
    pretrained: bool,
    layers: list[str] | None,
) -> nn.Module:
    """Construct the backbone without consulting the cache."""
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

    # timm architectures with a verified layer-name mapping get the wrapper
    # that honours the ``{layer_name: tensor}`` contract.
    if backbone_name in _TIMM_FEATURE_INDICES:
        try:
            return TimmFeatureBackbone(
                backbone_name, layers or ["layer2", "layer3"], pretrained=pretrained
            )
        except ImportError as e:
            raise ValueError(
                f"Backbone {backbone_name} requires timm, which is not installed"
            ) from e

    # Other timm names: returned raw as before (hook-based extraction works
    # only for models that expose the requested layer attributes).
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

    if backbone_name == "dinov2_vitb14":
        raise ValueError(
            "Backbone 'dinov2_vitb14' (BackboneType.DINO_V2) is not a timm model "
            "name and is not wired for feature extraction. For a DINOv2 backbone "
            "use a real timm name such as 'vit_base_patch14_dinov2' (note the "
            "patch-14 input sizing it requires); for the supported options use "
            f"{sorted([*backbone_map, *_TIMM_FEATURE_INDICES])}."
        )

    raise ValueError(
        f"Unsupported backbone: {backbone_name}. "
        f"Supported: {sorted([*backbone_map, *_TIMM_FEATURE_INDICES])}"
    )


def get_backbone(
    backbone_name: str,
    pretrained: bool = True,
    layers: list[str] | None = None,
    use_cache: bool = True,
) -> nn.Module:
    """Get a pre-trained backbone network.

    Args:
        backbone_name: Name of backbone architecture
        pretrained: Whether to load ImageNet pre-trained weights
        layers: Layer names the caller will extract (used to pick timm
            feature stages; ignored for torchvision models)
        use_cache: Whether a frozen pretrained model may be served from the
            module cache (as an independent ``deepcopy``). Trainable
            students must pass ``False``; non-pretrained models are never
            cached because random initialization is cheap and must stay
            per-instance.

    Returns:
        Pre-trained backbone module

    Raises:
        ValueError: If backbone_name is not supported
    """
    if not (use_cache and pretrained):
        return _build_backbone(backbone_name, pretrained, layers)

    out_key: tuple[int, ...] | None = None
    if backbone_name in _TIMM_FEATURE_INDICES:
        index_map = _TIMM_FEATURE_INDICES[backbone_name]
        requested = layers or ["layer2", "layer3"]
        out_key = tuple(sorted({index_map[layer] for layer in requested if layer in index_map}))
    key = (backbone_name, pretrained, out_key)
    if key not in _BACKBONE_CACHE:
        _BACKBONE_CACHE[key] = _build_backbone(backbone_name, pretrained, layers)
    return copy.deepcopy(_BACKBONE_CACHE[key])


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
        trainable: bool = False,
    ):
        """Initialize feature extractor.

        Args:
            backbone_name: Pre-trained backbone architecture name
            layers: Layers to extract features from (default: layer2, layer3)
            device: Computation device
            pretrained: Whether to use ImageNet pre-trained weights
            trainable: If True, the backbone's parameters keep their
                gradients and ``forward`` runs with autograd enabled, so the
                extractor can be trained (e.g. a distillation student).  The
                default keeps the historical frozen/no-grad behaviour for
                teacher paths.  Trainable extractors never come from the
                module-level backbone cache.
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.layers = layers or ["layer2", "layer3"]
        self.device = torch.device(device)
        self.trainable = trainable

        # Load backbone (cache only serves frozen pretrained models)
        self.backbone = get_backbone(
            backbone_name,
            pretrained=pretrained,
            layers=self.layers,
            use_cache=not trainable,
        )
        # Start in eval mode either way: a trainable student is switched to
        # train mode by its trainer's ``fit`` (e.g. ``.train()`` before the
        # epoch loop), and eval keeps BatchNorm usable on single images.
        self.backbone.eval()

        # Get layer info: the timm wrapper knows its real dims; otherwise use
        # the static table or infer from a forward pass.
        self._features: dict[str, torch.Tensor] = {}
        wrapper_dims = getattr(self.backbone, "feature_dims", None)
        if isinstance(wrapper_dims, dict):
            self.feature_dims = {layer: int(wrapper_dims[layer]) for layer in self.layers}
        elif backbone_name in BACKBONE_CONFIGS:
            self.feature_dims = {
                layer: BACKBONE_CONFIGS[backbone_name].get(layer, 512) for layer in self.layers
            }
        else:
            # Infer dimensions from forward pass
            self.feature_dims = self._infer_dims()

        # Register hooks for feature extraction
        self._register_hooks()

        # Freeze backbone unless it is a trainable student
        if not trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.to(self.device)

    def _register_hooks(self) -> None:
        """Register forward hooks to capture intermediate features."""
        if isinstance(self.backbone, TimmFeatureBackbone):
            return  # the wrapper returns the named feature dict directly
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
            out = self.backbone(dummy_input)
        if isinstance(out, dict):
            return {name: int(feat.shape[1]) for name, feat in out.items()}
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

        # Forward through backbone.  Frozen teacher paths stay under
        # ``no_grad``; a trainable student must keep the autograd graph, or
        # its optimizer receives no gradients and training is a no-op.
        if self.trainable:
            out = self.backbone(x)
        else:
            with torch.no_grad():
                out = self.backbone(x)
        if isinstance(out, dict):
            self._features.update(out)

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


class MultiScaleFeatureAggregator(nn.Module):
    """Aggregate multi-scale features into a unified representation.

    Combines features from different layers using adaptive pooling and optional learned projections.
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        output_dim: int = 512,
        pool_size: tuple[int, int] = (1, 1),
    ):
        """Initialize aggregator.

        Args:
            feature_dims: Dict mapping layer names to feature dimensions
            output_dim: Output embedding dimension
            pool_size: Spatial pooling size before projection
        """
        super().__init__()

        self.feature_dims = feature_dims
        self.output_dim = output_dim

        # Adaptive pooling for each scale
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

        # Projections for each layer
        total_dim = sum(feature_dims.values()) * pool_size[0] * pool_size[1]
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate multi-scale features.

        Args:
            features: Dict mapping layer names to feature tensors [B, C, H, W]

        Returns:
            Aggregated features [B, output_dim]

        Raises:
            KeyError: a layer in ``feature_dims`` is missing from ``features``;
                the projection is sized for all of them.
        """
        missing = [name for name in self.feature_dims if name not in features]
        if missing:
            raise KeyError(f"features missing layers {missing}")
        pooled = [self.pool(features[name]).flatten(1) for name in self.feature_dims]
        concatenated = torch.cat(pooled, dim=1)
        return self.projection(concatenated)


class PatchEmbedding(nn.Module):
    """Extract patch-level embeddings from feature maps.

    Converts spatial feature maps into sequences of patch embeddings for memory bank and attention-
    based methods.
    """

    def __init__(
        self,
        feature_dim: int,
        patch_size: int = 3,
        stride: int = 1,
        output_dim: int | None = None,
    ):
        """Initialize patch embedding.

        Args:
            feature_dim: Input feature dimension
            patch_size: Size of patches to extract
            stride: Stride for patch extraction
            output_dim: Output dimension (None = same as input)
        """
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride
        self.output_dim = output_dim or feature_dim

        # Unfold for patch extraction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)

        # Optional projection
        patch_dim = feature_dim * patch_size * patch_size
        if output_dim and output_dim != patch_dim:
            self.projection = nn.Linear(patch_dim, output_dim)
        else:
            self.projection = None  # type: ignore[assignment, unused-ignore]
            self.output_dim = patch_dim

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Extract patch embeddings from feature map.

        Args:
            features: Feature map [B, C, H, W]

        Returns:
            Tuple of:
                - Patch embeddings [B, num_patches, output_dim]
                - Patch grid size (h_patches, w_patches)
        """
        _batch_size, _channels, height, width = features.shape

        # Calculate output grid size
        h_patches = (height - self.patch_size) // self.stride + 1
        w_patches = (width - self.patch_size) // self.stride + 1

        # Extract patches: [B, C*patch_size*patch_size, num_patches]
        patches = self.unfold(features)

        # Reshape to [B, num_patches, patch_dim]
        patches = patches.permute(0, 2, 1)

        # Optional projection
        if self.projection is not None:
            patches = self.projection(patches)

        return patches, (h_patches, w_patches)
