# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dual-Student Knowledge Distillation for Anomaly Detection.

Implementation based on recent SOTA papers (2024):
- "Dual-Student Knowledge Distillation for Visual Anomaly Detection"
- "Generalist Multi-Class AD via Distillation to Two Heterogeneous Students"

Key Innovation:
Two students with inverted architectures learning from one teacher.
- Student 1 (Encoder-Decoder): Specialized for patch-level defects
- Student 2 (Encoder-Encoder): Optimized for semantic anomalies

The dual structure enhances normal consistency while introducing
diversity for anomaly representation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from omni_mercury_engine.core.conformal_prediction import (
    split_calibration_indices,
    split_conformal_threshold,
)
from omni_mercury_engine.core.exceptions import (
    DataException,
    InputModalityError,
    InvalidObservationError,
)

logger = logging.getLogger(__name__)


@dataclass
class DualStudentConfig:
    """Configuration for Dual-Student distillation.

    Attributes:
        backbone: Backbone architecture for teacher/students
        hidden_dim: Hidden dimension for students
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        temperature: RESERVED. Classical softened-logit distillation
            (Hinton et al. 2015) divides teacher/student LOGITS by T before
            the softmax; this distiller matches teacher FEATURES with a
            cosine loss, so there are no logits and no principled place for
            a temperature. The knob is kept for config compatibility and
            deliberately unwired (maintainer decision); it is not validated
            and changing it changes nothing.
        alpha: Convex weight tilting training between the two students:
            ``total_loss = alpha * loss1 + (1 - alpha) * loss2``, where
            student 1 (encoder-decoder) is the patch-level/reconstruction
            branch and student 2 (encoder-encoder) the semantic branch.
            1.0 trains only student 1, 0.0 only student 2; the default 0.5
            weights both equally (the historical unweighted-sum behaviour
            up to the overall scale).
        imagenet_normalize: Standardise inputs with the ImageNet statistics
            the pre-trained teacher was trained on (images in [0, 1]).
        threshold: Decision threshold on the anomaly score.  ``None`` (the
            default) calibrates it in ``fit`` on held-out normal images by
            split conformal prediction at ``false_alarm_rate``; a float is
            used verbatim and every image trains.
        false_alarm_rate: Target probability that a normal image is flagged.
        calibration_fraction: Share of the normal images held out to
            calibrate the threshold.
        calibration_seed: Seed of the train/calibration split.
    """

    backbone: str = "resnet18"
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 100
    temperature: float = 4.0
    alpha: float = 0.5
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    imagenet_normalize: bool = True
    threshold: float | None = None
    false_alarm_rate: float = 0.05
    calibration_fraction: float = 0.2
    calibration_seed: int = 0

    def __post_init__(self) -> None:
        """Validate the student weighting.

        Raises:
            ValueError: If ``alpha`` is not a finite number in [0, 1] -- a
                weight outside the convex range would train a student with a
                negative loss coefficient (gradient ascent) silently.
        """
        alpha = float(self.alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"alpha must be a finite weight in [0, 1], got {self.alpha!r}; "
                "1.0 trains only the patch-level student, 0.0 only the semantic student"
            )
        self.alpha = alpha


class EncoderDecoderStudent(nn.Module):
    """Encoder-Decoder student for patch-level anomaly detection.

    Learns to reconstruct teacher features, specialized for detecting local/patch-level defects.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
    ):
        """Initialize encoder-decoder student.

        Args:
            in_channels: Input feature channels
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Reconstructed features [B, C, H, W]
        """
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded: torch.Tensor = self.decoder(bottleneck)
        return decoded


class EncoderEncoderStudent(nn.Module):
    """Encoder-Encoder student for semantic anomaly detection.

    Uses skip connections to preserve semantic information, specialized for detecting
    semantic/contextual anomalies.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
    ):
        """Initialize encoder-encoder student.

        Args:
            in_channels: Input feature channels
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # First encoder block
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Second encoder block with skip
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Attention for feature recombination
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.output = nn.Conv2d(hidden_dim, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Processed features [B, C, H, W]
        """
        # First encoding
        enc1 = self.enc1(x)

        # Second encoding with residual
        enc2 = self.enc2(enc1) + enc1

        # Attention modulation
        attn = self.attention(enc2)
        attn = attn.view(-1, enc2.shape[1], 1, 1)
        modulated = enc2 * attn

        # Output
        output: torch.Tensor = self.output(modulated)
        return output


class DualStudentDistillation(nn.Module):
    """Dual-Student Knowledge Distillation for anomaly detection.

    Trains two student networks with different architectures
    to learn from a pre-trained teacher. Anomalies are detected
    based on discrepancy between teacher and student outputs.

    Example:
        >>> distiller = DualStudentDistillation()
        >>> distiller.fit(normal_images)
        >>> anomaly_scores = distiller.detect(test_images)
    """

    _norm_mean: torch.Tensor
    _norm_std: torch.Tensor

    def __init__(self, config: DualStudentConfig | dict[str, Any] | None = None) -> None:
        """Initialize dual-student distillation.

        Args:
            config: Distillation configuration
        """
        super().__init__()

        if config is None:
            self.config = DualStudentConfig()
        elif isinstance(config, dict):
            self.config = DualStudentConfig(**config)
        else:
            self.config = config

        self.device = torch.device(self.config.device)

        # Teacher (pre-trained, frozen)
        self.teacher: nn.Module | None = None

        # Students
        self.student1: EncoderDecoderStudent | None = None
        self.student2: EncoderEncoderStudent | None = None

        self._is_fitted = False
        self._feature_dim: int = 0
        # ``inf`` until fit calibrates it: no decision before calibration.
        self.threshold = math.inf if self.config.threshold is None else float(self.config.threshold)
        #: How ``threshold`` was set (``fixed`` or ``split_conformal`` with
        #: the achieved false-alarm bound); ``None`` before ``fit``.
        self.calibration: dict[str, Any] | None = None
        self.register_buffer("_norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prepare(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """``data`` as a finite float32 image batch [N, 3, H, W] on the device.

        Raises:
            InputModalityError: If ``data`` is not a 3-channel image batch.
            InvalidObservationError: If it contains non-finite pixels (they
                used to score NaN and read as "not anomalous").
        """
        batch = torch.as_tensor(np.asarray(data) if not torch.is_tensor(data) else data)
        batch = batch.to(device=self.device, dtype=torch.float32)
        if batch.dim() != 4 or batch.shape[1] != 3:
            raise InputModalityError(
                "DualStudentDistillation expects an image batch [N, 3, H, W]; "
                f"got shape {tuple(batch.shape)}"
            )
        if not bool(torch.isfinite(batch).all()):
            raise InvalidObservationError("DualStudentDistillation: non-finite pixel values")
        if self.config.imagenet_normalize:
            batch = (batch - self._norm_mean.to(batch.device)) / self._norm_std.to(batch.device)
        return batch

    def _discrepancy_maps(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position ``1 - cos(teacher, student)`` for each student, [B, H, W].

        The discrepancy the students are trained to minimise
        (:meth:`_distillation_loss`); scoring anything else -- the squared
        error of the raw features, as before -- measures the student output
        norm the cosine loss never constrains, and ranked real defects below
        normal images.
        """
        assert self.teacher is not None
        assert self.student1 is not None
        assert self.student2 is not None
        with torch.no_grad():
            teacher_agg = self._aggregate_features(self.teacher(batch))
            t_norm = nn.functional.normalize(teacher_agg, p=2, dim=1)
            maps = []
            for student in (self.student1, self.student2):
                s_norm = nn.functional.normalize(student(teacher_agg), p=2, dim=1)
                maps.append(1.0 - (t_norm * s_norm).sum(dim=1))
        return maps[0], maps[1]

    def _scores(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diff1, diff2 = self._discrepancy_maps(batch)
        anomaly_maps = torch.max(diff1, diff2)
        return anomaly_maps.flatten(1).max(dim=1)[0], diff1, diff2

    @property
    def feature_dim(self) -> int:
        """Aggregated teacher feature channels (0 until networks are initialized)."""
        return self._feature_dim

    def _initialize_networks(self, sample_input: torch.Tensor) -> None:
        """Initialize networks based on input dimensions.

        Args:
            sample_input: Sample input for dimension inference
        """
        # Initialize teacher
        from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor

        self.teacher = FeatureExtractor(
            backbone_name=self.config.backbone,
            layers=["layer2", "layer3"],
            device=self.device,
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Get feature dimensions
        with torch.no_grad():
            features = self.teacher(sample_input)
            total_channels = sum(f.shape[1] for f in features.values())

        self._feature_dim = total_channels

        # Initialize students
        self.student1 = EncoderDecoderStudent(total_channels, self.config.hidden_dim).to(
            self.device
        )

        self.student2 = EncoderEncoderStudent(total_channels, self.config.hidden_dim).to(
            self.device
        )

        logger.info(f"Initialized dual-student networks (feature_dim={total_channels})")

    def _aggregate_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate multi-scale teacher features.

        Args:
            features: Dict of layer features

        Returns:
            Aggregated features [B, C_total, H, W]
        """
        feature_list = list(features.values())

        # Find minimum spatial size
        min_h = min(f.shape[2] for f in feature_list)
        min_w = min(f.shape[3] for f in feature_list)

        resized = []
        for feat in feature_list:
            if feat.shape[2] != min_h or feat.shape[3] != min_w:
                feat = nn.functional.interpolate(
                    feat, size=(min_h, min_w), mode="bilinear", align_corners=False
                )
            resized.append(feat)

        return torch.cat(resized, dim=1)

    def _distillation_loss(
        self,
        teacher_feat: torch.Tensor,
        student_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss.

        Uses cosine similarity for better gradient flow.

        Args:
            teacher_feat: Teacher features [B, C, H, W]
            student_feat: Student features [B, C, H, W]

        Returns:
            Loss tensor
        """
        # Normalize along channel dimension
        t_norm = nn.functional.normalize(teacher_feat, p=2, dim=1)
        s_norm = nn.functional.normalize(student_feat, p=2, dim=1)

        # Cosine similarity loss (1 - cosine similarity)
        similarity = (t_norm * s_norm).sum(dim=1)
        loss = (1 - similarity).mean()

        return loss

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> DualStudentDistillation:
        """Train students on normal data.

        Datasets smaller than ``batch_size`` train as a single full batch;
        they are never silently skipped by ``drop_last``.

        Args:
            data: Normal images [N, C, H, W]

        Returns:
            Self for method chaining

        Raises:
            ValueError: If ``data`` contains no samples.
        """
        if len(data) == 0:
            raise ValueError("fit() requires at least one sample; got an empty batch")

        data = self._prepare(data)
        calibration: torch.Tensor | None = None
        if self.config.threshold is not None:
            self.threshold = float(self.config.threshold)
            self.calibration = {"method": "fixed", "threshold": self.threshold}
        else:
            try:
                fit_idx, cal_idx = split_calibration_indices(
                    len(data), self.config.calibration_fraction, self.config.calibration_seed
                )
            except ValueError as e:
                raise DataException(
                    f"DualStudentDistillation calibrates its threshold on held-out normal "
                    f"images: {e}. Pass more normal images, or configure an explicit threshold."
                ) from e
            calibration = data[torch.as_tensor(cal_idx, device=data.device)]
            data = data[torch.as_tensor(fit_idx, device=data.device)]

        # Initialize networks
        if self.student1 is None:
            self._initialize_networks(data[:1])

        logger.info(f"Training dual-student on {len(data)} images")

        # Assert networks are initialized
        assert self.student1 is not None
        assert self.student2 is not None
        assert self.teacher is not None

        # Setup optimizer
        params = list(self.student1.parameters()) + list(self.student2.parameters())
        optimizer = optim.Adam(params, lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)

        # Data loader. Cap the batch size at the dataset size: with
        # drop_last=True a smaller dataset would otherwise yield zero
        # batches and fit() would mark itself fitted without training.
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(data)),
            shuffle=True,
            drop_last=True,
        )

        # Training loop
        self.student1.train()
        self.student2.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for (batch,) in loader:
                batch = batch.to(self.device)

                # Get teacher features
                with torch.no_grad():
                    teacher_features = self.teacher(batch)
                    teacher_agg = self._aggregate_features(teacher_features)

                # Get student outputs
                student1_out = self.student1(teacher_agg)
                student2_out = self.student2(teacher_agg)

                # Distillation losses
                loss1 = self._distillation_loss(teacher_agg, student1_out)
                loss2 = self._distillation_loss(teacher_agg, student2_out)

                # Combined loss: alpha tilts training between the patch-level
                # student (loss1) and the semantic student (loss2). At the
                # default 0.5 this is the historical loss1 + loss2 up to a
                # constant 0.5 factor, which Adam's per-parameter scaling
                # absorbs.
                alpha = self.config.alpha
                total_loss = alpha * loss1 + (1.0 - alpha) * loss2

                optimizer.zero_grad()
                total_loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.6f}")

        self.student1.eval()
        self.student2.eval()
        self._is_fitted = True

        if calibration is not None:
            result = split_conformal_threshold(
                self._scores(calibration)[0].cpu().numpy(), self.config.false_alarm_rate
            )
            self.threshold = result.threshold
            self.calibration = result.as_record()
            if result.false_alarm_bound > result.false_alarm_rate:
                logger.warning(
                    "DualStudentDistillation: %d calibration image(s) bound the false-alarm "
                    "rate by %.3f, above the requested %.3f",
                    result.n_calibration,
                    result.false_alarm_bound,
                    result.false_alarm_rate,
                )

        logger.info("Dual-student training complete")
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using dual-student discrepancy.

        Args:
            data: Test images [N, C, H, W]

        Returns:
            Detection results dict
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before detect()")

        assert self.student1 is not None
        assert self.student2 is not None
        self.student1.eval()
        self.student2.eval()

        scores, diff1, diff2 = self._scores(self._prepare(data))
        anomaly_maps = torch.max(diff1, diff2)
        return {
            "scores": scores.cpu().numpy(),
            "anomaly_maps": anomaly_maps.cpu().numpy(),
            # Against the calibrated threshold.  Before, the decision was
            # ``score > batch mean``: a single image was never flagged and
            # an all-normal batch always flagged some.
            "is_anomaly": (scores > self.threshold).cpu().numpy(),
            "threshold": self.threshold,
            "student1_maps": diff1.cpu().numpy(),
            "student2_maps": diff2.cpu().numpy(),
        }

    def predict(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Registry contract: the same per-image scores and decisions as :meth:`detect`."""
        result = self.detect(data)
        return {
            "anomaly_scores": result["scores"],
            "is_anomaly": result["is_anomaly"],
            "threshold": result["threshold"],
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-image measured discrepancy summary.

        Returns:
            ``[N, 5]``: anomaly score, then mean and max of each student's
            discrepancy map (student 1 mean, student 1 max, student 2 mean,
            student 2 max).
        """
        result = self.detect(data)
        maps1, maps2 = result["student1_maps"], result["student2_maps"]
        n = len(result["scores"])
        return np.stack(
            [
                result["scores"],
                maps1.reshape(n, -1).mean(axis=1),
                maps1.reshape(n, -1).max(axis=1),
                maps2.reshape(n, -1).mean(axis=1),
                maps2.reshape(n, -1).max(axis=1),
            ],
            axis=1,
        ).astype(np.float64)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for PyTorch integration."""
        return self.detect(x)
