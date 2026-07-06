# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""SR-CNN: supervised CNN over Spectral-Residual saliency for time-series anomalies.

SR-CNN (Ren et al., *Time-Series Anomaly Detection Service at Microsoft*, KDD
2019) pairs the training-free Spectral-Residual (SR) saliency transform with a
small convolutional discriminator. SR alone thresholds saliency heuristically;
SR-CNN instead *learns* the saliency-to-anomaly mapping from data using a
self-supervised trick: synthetic point anomalies are injected into a copy of the
normal training series, the SR saliency map is recomputed, and a 1-D CNN is
trained to label the injected points from their local saliency window. At
inference the CNN scores each point's saliency window, turning SR's raw saliency
into a calibrated ``[0, 1]`` probability.

This detector wraps that pipeline in the
:class:`~omni_mercury_engine.core.base.BaseDetector` contract and reuses the SR
transform from
:class:`~omni_mercury_engine.detectors.spectral_residual.SpectralResidualDetector`
so the saliency definition stays identical across the SR and SR-CNN detectors. It
requires PyTorch (registered as a lazy, torch-gated BASE detector); when PyTorch
is absent the module simply is not imported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors.spectral_residual import SpectralResidualDetector

__all__ = ["SRCNNDetector"]


class _SRCNN(nn.Module):
    """1-D CNN mapping a saliency window to its centre-point anomaly logit."""

    def __init__(self, window: int, hidden: int) -> None:
        """Build the two-layer temporal CNN head.

        Args:
            window: Saliency window length fed to the network.
            hidden: Channel width of the convolutional layers.
        """
        super().__init__()
        # Max-pool (not average): the SR saliency of an anomaly is a *peak* in
        # the window; averaging washes it out, whereas max-pool preserves the
        # strongest saliency response the discriminator keys on.
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the centre-point logit for each ``(batch, 1, window)`` input."""
        return self.net(x).squeeze(-1)


class SRCNNDetector(BaseDetector):
    """Spectral-Residual + CNN discriminator anomaly detector (Ren et al., 2019).

    :meth:`fit` computes the SR saliency of a synthetic-anomaly-augmented copy of
    the training series and trains a 1-D CNN to classify each point's saliency
    window; :meth:`detect` scores each point of a new series by the CNN's
    sigmoid output on its saliency window.
    """

    def __init__(
        self,
        window: int = 33,
        hidden: int = 16,
        epochs: int = 60,
        lr: float = 1e-2,
        inject_ratio: float = 0.05,
        inject_scale: float = 6.0,
        seed: int = 0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the SR-CNN detector.

        Args:
            window: Odd saliency-window length centred on the scored point.
                Coerced to the next odd integer if even. Must be >= 3.
            hidden: Convolutional channel width. Must be >= 1.
            epochs: Training epochs for the discriminator. Must be >= 1.
            lr: Adam learning rate. Must be > 0.
            inject_ratio: Fraction of points that receive a synthetic anomaly
                during self-supervised training. Must be in ``(0, 1)``.
            inject_scale: Std-multiple magnitude of injected synthetic spikes.
                Must be > 0.
            seed: Torch/NumPy seed for deterministic training.
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If any parameter is out of its valid range.
        """
        super().__init__(config)
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 < inject_ratio < 1.0:
            raise ValueError(f"inject_ratio must be in (0, 1), got {inject_ratio}")
        if inject_scale <= 0.0:
            raise ValueError(f"inject_scale must be > 0, got {inject_scale}")
        self.window = int(window) | 1  # force odd so the window has a centre
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.inject_ratio = float(inject_ratio)
        self.inject_scale = float(inject_scale)
        self.seed = int(seed)
        self._sr = SpectralResidualDetector()
        self._model: _SRCNN | None = None

    def is_fitted(self) -> bool:
        """Return ``True`` once the CNN discriminator has been trained."""
        return self._is_fitted

    @staticmethod
    def _to_1d(data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        arr = np.nan_to_num(np.asarray(data, dtype=np.float64)).ravel()
        if arr.size == 0:
            raise ValueError("input series is empty")
        return arr

    def _windows(self, saliency: np.ndarray[Any, Any]) -> torch.Tensor:
        """Reflect-pad the saliency map and extract one window per point."""
        half = self.window // 2
        padded = np.pad(saliency, half, mode="reflect") if saliency.size > 1 else saliency
        if padded.size < self.window:
            padded = np.pad(padded, (0, self.window - padded.size), mode="edge")
        strided = np.lib.stride_tricks.sliding_window_view(padded, self.window)[: saliency.size]
        return torch.from_numpy(np.ascontiguousarray(strided, dtype=np.float32)).unsqueeze(1)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> SRCNNDetector:
        """Inject synthetic anomalies, learn SR scale, and train the CNN.

        Args:
            data: Training series of normal behaviour (1-D).

        Returns:
            ``self``.
        """
        series = self._to_1d(data)
        self._sr.fit(series)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        augmented = series.copy()
        labels = np.zeros(series.size, dtype=np.float32)
        std = float(series.std()) + 1e-9
        n_inject = max(1, int(series.size * self.inject_ratio))
        idx = rng.choice(series.size, size=n_inject, replace=False)
        augmented[idx] += rng.choice([-1.0, 1.0], size=n_inject) * self.inject_scale * std
        labels[idx] = 1.0

        saliency = self._sr._saliency_map(augmented)
        features = self._windows(saliency)
        targets = torch.from_numpy(labels)

        model = _SRCNN(self.window, self.hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # Balance the rare positive class in the BCE objective. Cast the NumPy
        # scalar to float first so the ratio is plain-float arithmetic (avoids a
        # numpy ``__truediv__`` overload ambiguity under the strict type gate).
        n_pos = float(labels.sum())
        pos_weight = torch.tensor([(float(labels.size) - n_pos) / max(n_pos, 1.0)])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
        model.eval()
        self._model = model
        self._is_fitted = True
        return self

    def _point_scores(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Per-point CNN anomaly probabilities in ``[0, 1]``."""
        if self._model is None:
            raise RuntimeError("call fit() before scoring")
        saliency = self._sr._saliency_map(series)
        with torch.no_grad():
            logits = self._model(self._windows(saliency))
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.clip(probs.astype(np.float64), 0.0, 1.0)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-point anomaly scores in ``[0, 1]`` from the SR-CNN discriminator."""
        series = self._to_1d(data)
        scores = self._point_scores(series)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores.astype(np.float32),
            "is_anomaly": scores > self.threshold,
            "confidence": scores.astype(np.float32),
            "metadata": {"window": self.window},
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-point fusion feature: the CNN anomaly probability.

        Args:
            data: Input series (1-D).

        Returns:
            ``(n_points, 1)`` float32 anomaly probabilities.
        """
        series = self._to_1d(data)
        return self._point_scores(series).astype(np.float32).reshape(-1, 1)
