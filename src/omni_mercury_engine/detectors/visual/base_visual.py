# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Base classes for visual anomaly detection.

Provides unified interface for all visual anomaly detection algorithms,
ensuring consistent API across PatchCore, PaDiM, STFPM, and other methods.
"""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, ClassVar, Self, cast

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.core.base import BaseDetector, DetectorMetrics
from omni_mercury_engine.core.conformal_prediction import (
    split_calibration_indices,
    split_conformal_threshold,
)
from omni_mercury_engine.core.exceptions import DataException, InputModalityError
from omni_mercury_engine.security.safe_torch import safe_torch_load

logger = logging.getLogger(__name__)

#: Fixed seed of the deterministic fusion projection: two detector instances
#: (or a save/load round trip) must map features to the fusion space with the
#: same random matrix, bit for bit.
_FUSION_PROJECTION_SEED = 0x4D455243  # "MERC"


class IncompatibleCheckpointError(RuntimeError):
    """A checkpoint this build cannot score correctly, refused on ``load``.

    Raised when a visual-detector checkpoint was written under a different
    scoring / fitted-state layout than the running code (or carries fitted
    statistics on a feature grid this detector does not extract).  Loading
    it would not crash where it matters -- it would score on another scale
    than its threshold was calibrated on -- so ``load`` refuses and says
    what changed.  A ``RuntimeError`` subclass, like every other ``load``
    refusal.
    """


class ConfigAliasMixin:
    """Alias fields of a dataclass config that mirror a canonical field.

    A config keeps released alias names next to their canonical fields
    (``PatchCoreConfig.k_nearest`` for ``num_neighbors``).  Resolving the
    pair by comparing values against the defaults cannot tell "explicitly
    set to the default" from "unset", so ``dataclasses.replace(config,
    num_neighbors=9)`` back to the default was ignored and a value mutated
    back to the default reverted on an ``asdict`` -> reconstruct round trip.
    This mixin resolves the pair from what the constructor was actually
    GIVEN:

    * ``__new__`` records the names of the arguments passed to the
      constructor, so a canonical field given explicitly is known as such
      whatever its value;
    * an alias field declares ``None`` as its "unset" default.  In
      :meth:`_resolve_aliases` (called from ``__post_init__``) a canonical
      field that was given always wins; otherwise a given alias sets it;
    * after construction the alias reads back the effective (canonical)
      value, and assigning either name updates both, so the pair can never
      disagree.

    ``asdict`` -> reconstruct and ``dataclasses.replace`` pass every field,
    so the canonical value always survives them.  The one thing a caller
    cannot do is ``replace(config, <alias>=...)``: ``replace`` also passes
    the old canonical value, which wins -- change the canonical name
    instead.

    (``detectors.vlm.base_vlm`` carries the same mixin: the ``visual`` and
    ``vlm`` extras install independently, so neither package imports the
    other.)
    """

    #: alias field name -> the canonical field it mirrors.
    _FIELD_ALIASES: ClassVar[dict[str, str]] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """Record which constructor arguments were given, then build."""
        self = super().__new__(cls)
        init_names = [f.name for f in fields(cast("Any", cls)) if f.init]
        given = frozenset(init_names[: len(args)]) | frozenset(kwargs)
        object.__setattr__(self, "_given_fields", given)
        object.__setattr__(self, "_aliases_live", False)
        return self

    def _was_given(self, name: str) -> bool:
        """Whether ``name`` was passed to the constructor (not defaulted)."""
        return name in self.__dict__.get("_given_fields", frozenset())

    def _resolve_aliases(self) -> None:
        """Reconcile every alias pair; call from ``__post_init__``."""
        for alias, canonical in self._FIELD_ALIASES.items():
            value = getattr(self, alias)
            if value is not None and not self._was_given(canonical):
                object.__setattr__(self, canonical, value)
            object.__setattr__(self, alias, getattr(self, canonical))
        object.__setattr__(self, "_aliases_live", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign ``name``; once constructed, keep its alias twin in step."""
        object.__setattr__(self, name, value)
        if not self.__dict__.get("_aliases_live", False):
            return
        for alias, canonical in self._FIELD_ALIASES.items():
            if name == alias:
                object.__setattr__(self, canonical, value)
            elif name == canonical:
                object.__setattr__(self, alias, value)


class BackboneType(Enum):
    """Supported backbone architectures for feature extraction.

    ``DINO_V2`` is retained as a released member, but its value
    ``"dinov2_vitb14"`` is not a timm model name: constructing it raises a
    clear error naming the real timm DINOv2 variants (see
    :func:`~omni_mercury_engine.detectors.visual.backbone.create_backbone`).
    It is kept rather than removed so existing references do not break.
    """

    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    WIDE_RESNET50_2 = "wide_resnet50_2"
    EFFICIENTNET_B4 = "efficientnet_b4"
    VIT_BASE = "vit_base_patch16_224"
    DINO_V2 = "dinov2_vitb14"


@dataclass
class VisualDetectorConfig:
    """Configuration for visual anomaly detectors.

    Attributes:
        backbone: Pre-trained backbone architecture for feature extraction
        image_size: Input image dimensions (height, width)
        layers: Feature extraction layers to use from backbone
        device: Computation device (cuda/cpu/mps)
        batch_size: Batch size for inference
        threshold: Anomaly score threshold for binary classification.
            ``None`` (the default) calibrates it during ``fit`` from
            held-out normal images by split conformal prediction, so the
            decision is on the detector's own score scale.  A float is used
            verbatim and ``fit`` then trains on every image.
        false_alarm_rate: Target probability that a normal image is flagged
            when the threshold is calibrated (split-conformal level alpha).
        calibration_fraction: Share of the normal images held out from
            training to calibrate the threshold (at least one image each
            for training and calibration).
        calibration_seed: Seed of the deterministic train/calibration split.
        normalize_features: Whether ``extract_features`` L2-normalizes the
            fusion embedding it returns.
        pool_strategy: How ``extract_features`` pools patch/spatial features
            into the fusion embedding: 'avg' and 'adaptive' average (the
            historical behaviour), 'max' takes the maximum.
    """

    backbone: str = "resnet18"
    image_size: tuple[int, int] = (224, 224)
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    threshold: float | None = None
    normalize_features: bool = True
    pool_strategy: str = "adaptive"
    false_alarm_rate: float = 0.05
    calibration_fraction: float = 0.2
    calibration_seed: int = 0


class BaseVisualDetector(BaseDetector, nn.Module):
    """Abstract base class for visual anomaly detectors.

    .. note:: **Transparent abstract contract (ROADMAP deferred item #4).**
        ``fit`` / ``detect`` / ``extract_features`` are genuine
        ``@abstractmethod`` declarations, so :class:`BaseVisualDetector`
        cannot be instantiated directly (a ``TypeError`` is raised) — there
        is no ``NotImplementedError`` stub on the public path. The native
        SOTA detectors (PatchCore, PaDiM, STFPM, ReverseDistillation, CFlow)
        are the concrete implementations, consistent with the 2026-05
        strategic decision to keep native detectors.

    Extends BaseDetector with visual-specific functionality including
    image preprocessing, feature extraction, and anomaly map generation.

    All visual detectors must implement:
        - fit(): Learn normal feature distribution
        - detect(): Compute anomaly scores and maps
        - extract_features(): Extract backbone features

    Attributes:
        config: Detector configuration
        backbone: Pre-trained feature extractor
        device: Computation device
    """

    #: Version of this detector's scoring rule and fitted-state layout.  It
    #: is stamped into every checkpoint and checked by :meth:`load`: bump it
    #: (and describe the change in :attr:`scoring_changes`) whenever scores,
    #: or the meaning of the saved state, stop being comparable with what an
    #: earlier build wrote.  A checkpoint without a stamp predates the
    #: stamp and is read as version 1.
    scoring_version: ClassVar[int] = 1
    #: ``version -> what changed at that version``, quoted by the refusal.
    scoring_changes: ClassVar[dict[int, str]] = {}

    def __init__(self, config: VisualDetectorConfig | dict[str, Any] | None = None) -> None:
        """Initialize visual detector.

        Args:
            config: Detector configuration (VisualDetectorConfig or dict)
        """
        # Initialize nn.Module first (required for PyTorch)
        nn.Module.__init__(self)

        # Handle config conversion
        if config is None:
            self.visual_config = VisualDetectorConfig()
        elif isinstance(config, dict):
            self.visual_config = VisualDetectorConfig(**config)
        else:
            self.visual_config = config

        # Expose config reference for internal use
        self._visual_config_ref = self.visual_config

        # Initialize BaseDetector attributes
        # Note: We don't call BaseDetector.__init__() because we inherit from nn.Module
        # and need to avoid MRO conflicts. Instead, we manually set the required attributes.
        # ``inf`` until ``fit`` calibrates it: no decision before calibration.
        configured = self.visual_config.threshold
        self.threshold = math.inf if configured is None else float(configured)
        #: How ``threshold`` was set: ``{"method": "fixed", ...}`` for a
        #: configured value, ``{"method": "split_conformal", ...}`` with the
        #: calibration size, the requested and the achieved false-alarm
        #: bound when it was calibrated by ``fit``; ``None`` before ``fit``.
        self.calibration: dict[str, Any] | None = None
        self._is_fitted = False
        self._name = self.__class__.__name__
        # BaseDetector's contract (its ``metrics`` property) promises a
        # real DetectorMetrics, so honour it even though we skip
        # BaseDetector.__init__ to avoid the nn.Module MRO conflict.
        self._metrics = DetectorMetrics()

        self.device = torch.device(self.visual_config.device)
        self._backbone: nn.Module | None = None
        self._feature_dim: int = 0

        # Deterministic fusion-projection matrices, keyed by (in_dim, out_dim).
        # Plain tensors (not Parameters/buffers), so they never enter
        # ``state_dict``: they are reproducible from _FUSION_PROJECTION_SEED.
        self._fusion_projections: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        # Image normalization (ImageNet stats) - use _norm prefix to avoid conflicts with subclass attributes
        self.register_buffer("_norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def visual_detector_config(self) -> VisualDetectorConfig:
        """Get the visual detector configuration."""
        return self._visual_config_ref

    @property
    def config(self) -> VisualDetectorConfig:
        """Get the detector configuration (alias for backward compatibility)."""
        return self._visual_config_ref

    @config.setter
    def config(self, value: VisualDetectorConfig | dict[str, Any]) -> None:
        """Set the detector configuration.

        Every reference the detector holds to its previous config follows
        the new one.  Each detector also keeps the config under its own typed
        name (``patchcore_config``, ``padim_config`` ...) and READS ITS
        SETTINGS THROUGH IT; left behind (as it was), a loaded detector
        reported the checkpoint's config while scoring with the
        constructor's.

        Args:
            value: VisualDetectorConfig instance or dict.
        """
        previous = getattr(self, "_visual_config_ref", None)
        if isinstance(value, dict):
            # Rebuild the detector's own config dataclass: a saved
            # ``PatchCoreConfig`` dict carries subclass fields
            # (``coreset_sampling_ratio`` ...) that the base
            # ``VisualDetectorConfig`` rejects, which made every concrete
            # visual detector impossible to ``load``.
            config_cls = type(previous or VisualDetectorConfig())
            value = config_cls(**value)
        elif not isinstance(value, VisualDetectorConfig):
            return
        if previous is not None:
            for name, held in list(vars(self).items()):
                if held is previous:
                    setattr(self, name, value)
        self._visual_config_ref = value
        self.visual_config = value

    @property
    def backbone(self) -> nn.Module:
        """Get the backbone feature extractor."""
        if self._backbone is None:
            raise RuntimeError("Backbone not initialized. Call _init_backbone() first.")
        return self._backbone

    @property
    def feature_dim(self) -> int:
        """Get the feature dimension after extraction."""
        return self._feature_dim

    def _init_backbone(self) -> None:
        """Initialize the backbone network.

        Override in subclasses if needed.
        """
        from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor

        self._backbone = FeatureExtractor(
            backbone_name=self.visual_config.backbone,
            layers=self.visual_config.layers,
            device=self.device,
        )
        self._backbone.eval()
        self._backbone.to(self.device)
        # What was actually built: ``load`` may replace the config, never the
        # extractor, so a checkpoint must have been fitted on this one.
        self._built_architecture = (
            str(self.visual_config.backbone),
            tuple(self.visual_config.layers),
        )

        # Compute feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, *self.visual_config.image_size).to(self.device)
            features = self._backbone(dummy)
            if isinstance(features, dict):
                self._feature_dim = sum(f.shape[1] for f in features.values())
            else:
                self._feature_dim = features.shape[1]

    def preprocess(self, images: torch.Tensor | np.ndarray[Any, Any]) -> torch.Tensor:
        """Preprocess images for feature extraction.

        Scaling to [0, 1] is explicit and per-sample, never keyed on the
        batch maximum (which wrongly divided already-normalized images that
        shared a batch with 8-bit ones): ``uint8`` input (NumPy or torch) is
        always divided by 255 -- decided on the input's own dtype by
        :meth:`_as_float_images`, before any float cast; float (and other
        integer) input is taken as-is unless a sample's own maximum exceeds
        1.5, in which case that sample alone is treated as 0-255 data and
        divided by 255.

        Args:
            images: Input images [B, C, H, W] or [B, H, W, C], values in [0, 255] or [0, 1]

        Returns:
            Normalized tensor [B, 3, H, W] on device

        Raises:
            ValueError: If the input is not a 4-D image batch with 1 or 3
                channels (off-modality input, e.g. a generic 2-D feature
                array).
        """
        images = self._as_float_images(images)

        if images.dim() != 4 or (images.shape[1] not in (1, 3) and images.shape[-1] not in (1, 3)):
            raise InputModalityError(
                f"{self._name} expects a batch of images with shape "
                f"[B, C, H, W] or [B, H, W, C] where C is 1 or 3 channels; "
                f"got input of shape {tuple(images.shape)}"
            )

        # Disambiguate channel layout. Prefer channel-first ([B, C, H, W]) when
        # the leading axis is already a valid 1/3-channel count; only treat the
        # input as channel-last ([B, H, W, C]) — and permute — when the leading
        # axis is not a valid channel count but the trailing one is. This avoids
        # mis-permuting a genuine channel-first tensor whose width happens to be
        # 1 or 3 (e.g. [B, 3, H, 3]).
        if images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
            images = images.permute(0, 3, 1, 2)

        # Scale to [0, 1].  uint8 input was already divided by 255 above (it
        # is by definition 0-255, and now lies in [0, 1], which this rule
        # leaves alone); float input is taken as-is unless a sample's own max
        # exceeds 1.5 (heuristic for float 0-255 input), decided per sample
        # so one 8-bit image cannot rescale the already-normalized images in
        # the same batch.
        sample_max = images.reshape(images.shape[0], -1).amax(dim=1)
        divisor = torch.where(
            sample_max > 1.5,
            torch.full_like(sample_max, 255.0),
            torch.ones_like(sample_max),
        )
        images = images / divisor.view(-1, 1, 1, 1)

        # Resize if needed
        if images.shape[-2:] != self.visual_config.image_size:
            images = torch.nn.functional.interpolate(
                images,
                size=self.visual_config.image_size,
                mode="bilinear",
                align_corners=False,
            )

        # Apply ImageNet normalization
        images = images.to(self.device)
        images = (images - self._norm_mean) / self._norm_std  # type: ignore[operator, unused-ignore]

        return images

    @staticmethod
    def _as_float_images(images: torch.Tensor | np.ndarray[Any, Any]) -> torch.Tensor:
        """Convert a raw image batch to a float tensor, 8-bit input scaled to [0, 1].

        The one place the input dtype is read.  ``uint8`` -- a NumPy array or
        a torch tensor alike -- is 0-255 by definition and is divided by 255
        here, while the dtype is still known.  Every ``fit`` / ``detect`` /
        ``extract_features`` entry point calls this first and none casts on
        its own: a bare ``torch.from_numpy(data).float()`` erased the dtype,
        left NumPy ``uint8`` input to the per-sample ``max > 1.5`` rule of
        :meth:`preprocess`, and a near-black 8-bit frame (values 0/1) was
        then read as an already-normalized full-contrast image.

        Float input is returned unscaled (the per-sample rule in
        :meth:`preprocess` owns it); other integer dtypes are cast to float
        and treated like float input.

        Args:
            images: Raw input batch.

        Returns:
            A floating-point tensor; ``uint8`` input scaled to [0, 1].
        """
        if isinstance(images, np.ndarray):
            if images.dtype == np.uint8:
                return torch.from_numpy(images).float() / 255.0
            return torch.from_numpy(images).float()
        if images.dtype == torch.uint8:
            return images.float() / 255.0
        return images if torch.is_floating_point(images) else images.float()

    @staticmethod
    def _input_spatial_size(images: torch.Tensor | np.ndarray[Any, Any]) -> tuple[int, int]:
        """Spatial ``(H, W)`` of a raw 4-D image batch, before preprocessing.

        Applies the same channel-layout rule as :meth:`preprocess`: the batch
        is channel-first unless axis 1 is not a valid 1/3-channel count while
        the trailing axis is -- then it is channel-last and the spatial size
        is axes 1 and 2.  Detectors must size their anomaly maps from this,
        never from ``shape[-2:]`` of the raw tensor, which reads ``(W, C)``
        for channel-last input.

        Args:
            images: Raw input batch, before :meth:`preprocess`.

        Returns:
            The input's spatial size ``(H, W)``.
        """
        shape = tuple(int(s) for s in images.shape)
        if len(shape) == 4 and shape[1] not in (1, 3) and shape[-1] in (1, 3):
            return shape[1], shape[2]
        return shape[-2], shape[-1]

    def _fusion_projection_params(
        self, in_dim: int, out_dim: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The deterministic linear map used to reach the fusion dimension.

        Initialized like ``nn.Linear`` but from a ``torch.Generator`` seeded
        with a fixed constant, so every instance (and every save/load round
        trip) produces the identical matrix.  Kept as plain tensors: it is
        derived, not learned, so it stays out of ``state_dict``.
        """
        key = (int(in_dim), int(out_dim))
        if key not in self._fusion_projections:
            generator = torch.Generator(device="cpu").manual_seed(_FUSION_PROJECTION_SEED)
            bound = 1.0 / math.sqrt(in_dim)
            weight = torch.empty(out_dim, in_dim).uniform_(-bound, bound, generator=generator)
            bias = torch.empty(out_dim).uniform_(-bound, bound, generator=generator)
            self._fusion_projections[key] = (weight, bias)
        weight, bias = self._fusion_projections[key]
        return weight.to(device), bias.to(device)

    def _project_features_for_fusion(
        self, features: torch.Tensor, output_dim: int = 128
    ) -> torch.Tensor:
        """Map pooled features to the fusion dimension for the ML pipeline.

        Applies the deterministic projection when the dimensions differ and,
        when ``visual_config.normalize_features`` is set (the default),
        L2-normalizes the result.

        Args:
            features: Pooled per-image features ``[N, D]``.
            output_dim: Fusion embedding width (128 for the fusion pipeline).

        Returns:
            Fusion features ``[N, output_dim]``.
        """
        if features.shape[1] != output_dim:
            weight, bias = self._fusion_projection_params(
                int(features.shape[1]), output_dim, features.device
            )
            features = features @ weight.t() + bias
        if self.visual_config.normalize_features:
            features = nn.functional.normalize(features, p=2, dim=1)
        return features

    def _pool_for_fusion(self, features: torch.Tensor, dim: int | list[int]) -> torch.Tensor:
        """Pool features along ``dim`` according to ``pool_strategy``.

        'avg' and 'adaptive' average (the historical behaviour); 'max' takes
        the maximum.

        Raises:
            DataException: If ``pool_strategy`` names an unknown strategy.
        """
        strategy = self.visual_config.pool_strategy
        if strategy == "max":
            return features.amax(dim=dim)
        if strategy in ("avg", "adaptive"):
            return features.mean(dim=dim)
        raise DataException(
            f"{self._name}: unknown pool_strategy {strategy!r}; expected 'avg', 'max' or 'adaptive'"
        )

    def postprocess_anomaly_map(
        self,
        anomaly_map: torch.Tensor,
        original_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Postprocess anomaly map to original image size.

        Args:
            anomaly_map: Raw anomaly map from detector
            original_size: Target size (H, W) for resizing

        Returns:
            Smoothed and resized anomaly map
        """
        # Gaussian smoothing for cleaner maps
        if anomaly_map.dim() == 3:
            anomaly_map = anomaly_map.unsqueeze(1)

        kernel_size = 33
        sigma = 4.0
        kernel = self._get_gaussian_kernel(kernel_size, sigma).to(anomaly_map.device)
        anomaly_map = torch.nn.functional.conv2d(anomaly_map, kernel, padding=kernel_size // 2)

        # Resize to original size
        if original_size is not None:
            anomaly_map = torch.nn.functional.interpolate(
                anomaly_map,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        return anomaly_map.squeeze(1)

    def postprocess(
        self,
        anomaly_map: torch.Tensor,
        original_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Alias for postprocess_anomaly_map for API compatibility.

        Args:
            anomaly_map: Raw anomaly map from detector
            original_size: Target size (H, W) for resizing

        Returns:
            Smoothed and resized anomaly map
        """
        return self.postprocess_anomaly_map(anomaly_map, original_size)

    @staticmethod
    def _get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel for smoothing."""
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
        kernel = gauss.outer(gauss)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    @abstractmethod
    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> BaseVisualDetector:
        """Fit detector to normal (non-anomalous) images.

        Args:
            data: Normal images [N, C, H, W] or [N, H, W, C]

        Returns:
            Self for method chaining

        Abstract contract — every native visual detector implements this.
        """
        ...

    @abstractmethod
    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies in images.

        Args:
            data: Test images [N, C, H, W] or [N, H, W, C]

        Returns:
            Dict containing:
                - scores: Image-level anomaly scores [N]
                - anomaly_maps: Pixel-level anomaly maps [N, H, W]
                - is_anomaly: Binary anomaly flags [N]
                - features: Extracted features for fusion [N, D]

        Abstract contract — every native visual detector implements this.
        """
        ...

    @abstractmethod
    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion pipeline.

        Args:
            data: Input images [N, C, H, W]

        Returns:
            Feature tensor [N, feature_dim] normalized for fusion

        Abstract contract — every native visual detector implements this.
        """
        ...

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for integration with PyTorch pipelines.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Detection results dict
        """
        return self.detect(x)

    def _split_for_calibration(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split normal images into training and calibration subsets.

        A fixed configured ``threshold`` needs no calibration: every image
        trains.  Otherwise a seeded ``calibration_fraction`` of the batch is
        held out so the threshold is calibrated on images the model never
        trained on -- scoring the training images themselves would put the
        threshold inside the in-sample error (a memory bank scores its own
        patches near zero), and normal images would then be flagged.

        Args:
            data: Raw image batch, before :meth:`preprocess`.

        Returns:
            ``(training_images, calibration_images)``; the second is ``None``
            when no calibration will run.

        Raises:
            DataException: If calibration is required and fewer than two
                images were given.
        """
        cfg = self.visual_config
        if cfg.threshold is not None:
            self.threshold = float(cfg.threshold)
            self.calibration = {"method": "fixed", "threshold": self.threshold}
            return data, None
        if data.dim() != 4:
            # Not an image batch: ``preprocess`` refuses it with the typed
            # modality error, which is the error the caller should see.
            return data, None
        n = int(data.shape[0])
        if n < 2:
            raise DataException(
                f"{self._name} calibrates its decision threshold on held-out normal "
                f"images and needs at least 2 to fit (got {n}). Pass more normal images, "
                "or configure an explicit `threshold`."
            )
        if not 0.0 < cfg.false_alarm_rate < 1.0:
            raise DataException(f"false_alarm_rate must lie in (0, 1); got {cfg.false_alarm_rate}")
        try:
            fit_idx, cal_idx = split_calibration_indices(
                n, cfg.calibration_fraction, cfg.calibration_seed
            )
        except ValueError as e:
            raise DataException(str(e)) from e
        index = torch.as_tensor(np.concatenate([fit_idx, cal_idx]), device=data.device)
        n_fit = len(fit_idx)
        return data[index[:n_fit]], data[index[n_fit:]]

    def _calibrate_threshold(self, calibration: torch.Tensor | None) -> None:
        """Set ``threshold`` by split conformal prediction on held-out normals.

        With ``n`` calibration scores and target level ``alpha``, the
        threshold is the ``k``-th smallest score, ``k = min(ceil((n + 1)(1 -
        alpha)), n)``.  For a new normal image exchangeable with the
        calibration images, ``P(score > threshold) <= (n + 1 - k)/(n + 1)``
        (Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2021).  That
        achieved bound is recorded in :attr:`calibration`; when too few
        images exist to reach ``alpha`` it is larger than ``alpha`` and a
        warning says so -- the guarantee is reported, never overstated.

        Args:
            calibration: Held-out normal images, or ``None`` (no-op).

        Raises:
            DataException: If a calibration score is not finite.
        """
        if calibration is None:
            return
        # Raw scores only: no decision is made while the threshold is unset.
        self.threshold = math.inf
        scores = np.asarray(self.detect(calibration)["scores"], dtype=np.float64).reshape(-1)
        try:
            result = split_conformal_threshold(scores, self.visual_config.false_alarm_rate)
        except ValueError as e:
            raise DataException(f"{self._name}: cannot calibrate: {e}") from e
        self.threshold = result.threshold
        self.calibration = result.as_record()
        if result.false_alarm_bound > result.false_alarm_rate:
            logger.warning(
                "%s: %d calibration image(s) bound the false-alarm rate by %.3f, above the "
                "requested %.3f; fit on more normal images to tighten it",
                self._name,
                result.n_calibration,
                result.false_alarm_bound,
                result.false_alarm_rate,
            )

    def _fitted_state(self) -> dict[str, Any]:
        """Fitted state that ``state_dict`` does not carry (plain tensor attributes).

        Subclasses holding their fitted model outside registered modules and
        buffers (a memory bank, Gaussian statistics, a random projection)
        return it here so :meth:`save` persists it; without this a saved
        detector reloads with weights but no fitted model.
        """
        return {}

    def _build_architecture_for_load(self) -> None:
        """Create lazily-built submodules so ``load_state_dict`` has targets.

        Detectors that build modules on the first ``fit`` batch build them
        here from the configured ``image_size`` (``fit`` sees preprocessed,
        i.e. resized, images, so the shapes are identical).
        """

    def _restore_fitted_state(self, state: dict[str, Any]) -> None:
        """Reinstate :meth:`_fitted_state` after the weights are loaded."""

    def _check_scoring_version(self, checkpoint: dict[str, Any], path: str) -> None:
        """Refuse a checkpoint written under another scoring version.

        Args:
            checkpoint: The loaded checkpoint payload.
            path: Where it was read from (for the message).

        Raises:
            IncompatibleCheckpointError: If the stamp differs from
                :attr:`scoring_version`.
        """
        current = int(self.scoring_version)
        stamped = "scoring_version" in checkpoint
        saved = checkpoint.get("scoring_version", 1)
        if isinstance(saved, int) and not isinstance(saved, bool) and saved == current:
            return
        if not isinstance(saved, int) or isinstance(saved, bool) or saved > current:
            raise IncompatibleCheckpointError(
                f"Checkpoint at '{path}' carries {self._name} scoring version {saved!r}, "
                f"newer than (or unknown to) this build, which scores at version {current}. "
                "It was written by a newer release: load it with that release, or refit "
                "the detector on normal images with this one."
            )
        changes = " ".join(
            f"[v{version}] {self.scoring_changes[version]}"
            for version in range(saved + 1, current + 1)
            if version in self.scoring_changes
        )
        origin = (
            f"scoring version {saved}"
            if stamped
            else f"no scoring version (the original format, version {saved})"
        )
        raise IncompatibleCheckpointError(
            f"Checkpoint at '{path}' was written by {self._name} with {origin}; this build "
            f"scores at version {current}. What changed: {changes or 'see the release notes.'} "
            "Loading it would score on a different scale than the saved state and threshold "
            "were produced on, so it is refused. A refit is required: fit the detector on "
            "normal images with this build and save a new checkpoint."
        )

    def _validate_checkpoint(self, checkpoint: dict[str, Any], path: str) -> None:
        """Refuse fitted state this detector cannot score.

        Called by :meth:`load` after the version check and before anything
        is restored, so a refusal leaves the detector untouched.  The base
        check: the checkpoint's ``backbone`` and ``layers`` must be the ones
        this detector's feature extractor was built with -- ``load`` replaces
        the config but cannot swap the extractor, and fitted state is only
        meaningful on the features it was fitted on.  Subclasses extend it
        (and call ``super()``).

        Args:
            checkpoint: The loaded checkpoint payload.
            path: Where it was read from (for the message).

        Raises:
            IncompatibleCheckpointError: If the checkpoint was fitted on
                another backbone or other layers.
        """
        built = getattr(self, "_built_architecture", None)
        config = checkpoint.get("config")
        if built is None or config is None:
            return
        if isinstance(config, dict):
            backbone, layers = config.get("backbone"), config.get("layers")
        else:
            backbone, layers = getattr(config, "backbone", None), getattr(config, "layers", None)
        if backbone is None or layers is None:
            return
        saved = (str(backbone), tuple(layers))
        if saved != built:
            raise IncompatibleCheckpointError(
                f"Checkpoint at '{path}' was fitted by {self._name} on backbone "
                f"{saved[0]!r} with layers {list(saved[1])}, but this detector was built "
                f"with backbone {built[0]!r} and layers {list(built[1])}. load() restores "
                "fitted state into the feature extractor the detector was constructed "
                "with; it cannot swap it. Construct the detector with the checkpoint's "
                "backbone and layers, then load."
            )

    def _load_probe(self) -> torch.Tensor:
        """A zero batch at the configured input size, for shape inference."""
        height, width = self.visual_config.image_size
        return torch.zeros(1, 3, int(height), int(width), device=self.device)

    def save(self, path: str) -> None:
        """Save detector state to file.

        The config dataclass is flattened to a primitive dict before
        ``torch.save`` so the checkpoint round-trips cleanly under
        ``weights_only=True`` on load -- PyTorch's safe-load mode
        refuses arbitrary classes, and a stored ``VisualDetectorConfig``
        instance would not be on the safe-globals list. The on-disk
        schema is now ``{"config": dict, "state_dict": ..., "is_fitted":
        bool}`` regardless of any future config field additions.

        ``scoring_version`` stamps the scoring rule / fitted-state layout
        the state was produced under (see :attr:`scoring_version`), so
        :meth:`load` can refuse a checkpoint the running code would
        mis-score.
        """
        torch.save(
            {
                "scoring_version": int(self.scoring_version),
                "config": asdict(self.visual_config),
                "state_dict": self.state_dict(),
                "is_fitted": self._is_fitted,
                "threshold": self.threshold,
                "calibration": self.calibration,
                "fitted_state": self._fitted_state(),
            },
            path,
        )

    def load(self, path: str) -> BaseVisualDetector:
        """Load detector state from file.

        Args:
            path: Path to saved detector state file

        Security Note:
            Loading is hard-pinned to ``weights_only=True`` so the
            checkpoint cannot execute arbitrary code via pickle. The
            previous ``allow_unsafe=True`` escape hatch has been
            removed; legacy checkpoints that cannot load with
            ``weights_only=True`` must be re-saved by the operator (or
            converted via ``tools/migrate_pkl.py`` if they originated
            as raw pickles).

            Round-trip contract: ``save()`` writes the config as a
            primitive dict and ``load()`` reconstructs the dataclass
            via keyword expansion. Checkpoints saved by an older
            code path that stored the full ``VisualDetectorConfig``
            object will fail under ``weights_only=True`` and must be
            re-saved.

        Compatibility: the checkpoint's ``scoring_version`` stamp must equal
        this detector's :attr:`scoring_version` (a checkpoint with no stamp
        is the original format, version 1).  A mismatch is refused before
        anything is restored -- the detector is left exactly as it was --
        with an error naming what changed; such a checkpoint cannot be
        migrated, the detector must be refitted.

        Raises:
            IncompatibleCheckpointError: If the checkpoint was written under
                another scoring version, or holds fitted state this detector
                cannot score (:meth:`_validate_checkpoint`).
            RuntimeError: If the checkpoint cannot load with
                ``weights_only=True``.
            TypeError: If the saved config dict references fields
                that the current ``VisualDetectorConfig`` no longer
                accepts (schema drift).
        """
        try:
            checkpoint = safe_torch_load(path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint at '{path}' cannot be loaded safely (weights_only=True). "
                "Re-save the checkpoint with the current code path; the unsafe "
                "fallback was removed because pickle has no role in the Mercury "
                f"Agent runtime. Original error: {e}"
            ) from e

        # Refuse an incompatible checkpoint BEFORE touching this detector.
        self._check_scoring_version(checkpoint, path)
        self._validate_checkpoint(checkpoint, path)

        config_payload = checkpoint["config"]
        if isinstance(config_payload, (VisualDetectorConfig, dict)):
            # Route through the ``config`` property setter rather than
            # assigning ``self.visual_config`` directly. The setter
            # also maintains ``self._visual_config_ref`` (which backs
            # the ``config`` / ``visual_detector_config`` accessors).
            # A direct attribute assignment would leave those two
            # references out of sync after a load -- ``detector.config``
            # would still return the pre-load config while
            # ``detector.visual_config`` carried the freshly loaded
            # checkpoint config. The setter accepts both the dataclass
            # instance (legacy on-disk shape) and a dict (current
            # ``asdict``-serialised shape).
            self.config = config_payload
        else:
            raise RuntimeError(
                f"Checkpoint at '{path}' has 'config' of type "
                f"{type(config_payload).__name__}; expected dict (from "
                "dataclasses.asdict) or VisualDetectorConfig."
            )
        self._build_architecture_for_load()
        state_dict = checkpoint["state_dict"]
        # Checkpoints written while ``extract_features`` lazily registered an
        # unseeded ``_fusion_projection`` Linear carry its two keys; the
        # projection is now deterministic (recreated from a fixed seed) and
        # lives outside ``state_dict``.  Exactly those keys are dropped --
        # every other key still loads strictly.
        for key in [k for k in state_dict if k.startswith("_fusion_projection.")]:
            del state_dict[key]
        self.load_state_dict(state_dict)
        self._restore_fitted_state(checkpoint.get("fitted_state") or {})
        self._is_fitted = checkpoint["is_fitted"]
        # A calibrated threshold lives on the detector, not in its config;
        # checkpoints written before calibration existed carry neither key
        # and keep the configured threshold.
        if "threshold" in checkpoint:
            self.threshold = checkpoint["threshold"]
            self.calibration = checkpoint.get("calibration")
        else:
            configured = self.visual_config.threshold
            self.threshold = math.inf if configured is None else float(configured)
        return self
