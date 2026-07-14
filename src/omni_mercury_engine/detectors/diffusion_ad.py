# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""DDPM-AD: denoising-diffusion reconstruction anomaly detector for time series.

Denoising Diffusion Probabilistic Models (Ho et al., 2020) learn to invert a
gradual Gaussian-noising process. Trained only on *normal* windows, the model
learns to predict the noise added to in-distribution signals; an anomalous window
lies off that learned manifold, so the model's noise-prediction error is
systematically larger. That reconstruction error is the anomaly signal
(diffusion-based anomaly detection, e.g. Wyatt et al. *AnoDDPM* 2022, adapted to
1-D streams).

This detector trains a compact conditional denoiser on standardised sliding
windows of the training series and scores each point by the average noise-
prediction error of its centred window across several diffusion timesteps,
squashed into a calibrated ``[0, 1]`` probability. It wraps the pipeline in the
:class:`~omni_mercury_engine.core.base.BaseDetector` contract and requires PyTorch
(registered as a lazy, torch-gated BASE detector); when PyTorch is absent the
module is simply not imported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors._calibration import (
    bound_finite_config,
    finite_scores,
    squash_scale,
)
from omni_mercury_engine.detectors._torch_perf import single_threaded_torch

__all__ = ["DiffusionReconstructionDetector"]

_LN2 = float(np.log(2.0))


class _Denoiser(nn.Module):
    """MLP that predicts the noise added to a window, conditioned on timestep."""

    def __init__(self, window: int, hidden: int, n_steps: int) -> None:
        """Build the timestep-conditioned denoising MLP.

        Args:
            window: Window length (input and output dimension).
            hidden: Hidden width.
            n_steps: Number of diffusion steps (timestep-embedding cardinality).
        """
        super().__init__()
        self.embed = nn.Embedding(n_steps, hidden)
        self.net = nn.Sequential(
            nn.Linear(window + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, window),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise for noisy windows ``x`` at integer timesteps ``t``."""
        h = torch.cat([x, self.embed(t)], dim=-1)
        return self.net(h)


class DiffusionReconstructionDetector(BaseDetector):
    """DDPM reconstruction-error anomaly detector over 1-D sliding windows.

    :meth:`fit` trains a conditional denoiser on standardised normal windows;
    :meth:`detect` scores each point by the mean noise-prediction error of its
    centred window across a set of diffusion timesteps, squashed to ``[0, 1]``.
    """

    def __init__(
        self,
        window: int = 16,
        hidden: int = 32,
        n_steps: int = 50,
        epochs: int = 40,
        lr: float = 5e-3,
        eval_steps: tuple[int, ...] = (5, 15, 25),
        calibration_quantile: float = 0.98,
        seed: int = 0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the diffusion reconstruction detector.

        Args:
            window: Odd sliding-window length centred on the scored point (coerced
                to the next odd integer). Must be >= 3.
            hidden: Denoiser hidden width. Must be >= 1.
            n_steps: Diffusion timesteps in the linear beta schedule. Must be >= 2.
            epochs: Training epochs. Must be >= 1.
            lr: Adam learning rate. Must be > 0.
            eval_steps: Timesteps at which reconstruction error is averaged at
                scoring time. Each must be in ``[0, n_steps)``.
            calibration_quantile: Training-error quantile placed at the 0.5
                boundary; ``1 - calibration_quantile`` is the normal-regime FPR.
                Must be in ``(0, 1)``.
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
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not eval_steps or any(not 0 <= s < n_steps for s in eval_steps):
            raise ValueError(f"eval_steps must be non-empty and in [0, {n_steps})")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.window = int(window) | 1
        self.hidden = int(hidden)
        self.n_steps = int(n_steps)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.eval_steps = tuple(int(s) for s in eval_steps)
        self.calibration_quantile = float(calibration_quantile)
        self.seed = int(seed)
        self._model: _Denoiser | None = None
        self._mean = 0.0
        self._std = 1.0
        self._scale = 1.0
        betas = np.linspace(1e-4, 0.05, self.n_steps, dtype=np.float64)
        self._alpha_bar = np.cumprod(1.0 - betas)

    def is_fitted(self) -> bool:
        """Return ``True`` once the denoiser has been trained."""
        return self._is_fitted

    def _to_1d(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        # Label non-finite corrections with this detector's name so the
        # ``omni_detector_nonfinite_corrected`` metric/log attributes them here
        # rather than to the generic "tier" default.
        arr = bound_finite_config(self, np.asarray(data, dtype=np.float64)).ravel()
        if arr.size == 0:
            raise ValueError("input series is empty")
        return arr

    def _windows(self, series: np.ndarray[Any, Any]) -> torch.Tensor:
        """Standardise and extract one reflect-padded window per point."""
        norm = (series - self._mean) / self._std
        half = self.window // 2
        padded = np.pad(norm, half, mode="reflect") if norm.size > 1 else norm
        if padded.size < self.window:
            padded = np.pad(padded, (0, self.window - padded.size), mode="edge")
        strided = np.lib.stride_tricks.sliding_window_view(padded, self.window)[: series.size]
        return torch.from_numpy(np.ascontiguousarray(strided, dtype=np.float32))

    def _recon_error(self, windows: torch.Tensor) -> np.ndarray[Any, Any]:
        """Mean noise-prediction error per window across ``eval_steps``."""
        assert self._model is not None
        n = windows.shape[0]
        generator = torch.Generator().manual_seed(self.seed)
        errors = torch.zeros(n)
        with single_threaded_torch(), torch.no_grad():
            for step in self.eval_steps:
                abar = float(self._alpha_bar[step])
                noise = torch.randn(windows.shape, generator=generator)
                noisy = (abar**0.5) * windows + ((1.0 - abar) ** 0.5) * noise
                t = torch.full((n,), step, dtype=torch.long)
                pred = self._model(noisy, t)
                errors += ((pred - noise) ** 2).mean(dim=-1)
        return (errors / len(self.eval_steps)).cpu().numpy().astype(np.float64)

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        return squash_scale(raw, self.calibration_quantile)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> DiffusionReconstructionDetector:
        """Train the denoiser on normal windows and set the squash scale.

        Args:
            data: Training series of normal behaviour (1-D).

        Returns:
            ``self``.
        """
        series = self._to_1d(data)
        self._mean = float(series.mean())
        self._std = float(series.std()) + 1e-9
        torch.manual_seed(self.seed)
        windows = self._windows(series)

        model = _Denoiser(self.window, self.hidden, self.n_steps)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        generator = torch.Generator().manual_seed(self.seed)
        n = windows.shape[0]
        model.train()
        # Tiny full-batch denoiser: pin to one intra-op thread (see
        # detectors/_torch_perf.py) — the fork/join overhead of the default
        # pool dominates at this tensor size.
        with single_threaded_torch():
            for _ in range(self.epochs):
                optimizer.zero_grad()
                t = torch.randint(0, self.n_steps, (n,), generator=generator)
                abar = torch.from_numpy(self._alpha_bar[t.numpy()].astype(np.float32)).unsqueeze(-1)
                noise = torch.randn(windows.shape, generator=generator)
                noisy = abar.sqrt() * windows + (1.0 - abar).sqrt() * noise
                pred = model(noisy, t)
                loss = loss_fn(pred, noise)
                loss.backward()
                optimizer.step()
        model.eval()
        self._model = model
        self._scale = self._squash_scale(self._recon_error(windows))
        self._is_fitted = True
        return self

    def _point_scores(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Per-point squashed reconstruction-error scores in ``[0, 1]``."""
        if self._model is None:
            raise RuntimeError("call fit() before scoring")
        raw = self._recon_error(self._windows(series))
        # finite_scores (not a bare np.clip, which passes NaN through): a huge
        # in-cap input can overflow the float32 window cast / reconstruction error
        # to inf and yield a NaN score, so guarantee a finite [0, 1] score and
        # attribute any correction here -- consistent with the rest of the tier.
        return finite_scores(1.0 - np.exp(-raw / self._scale), detector=self.name)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-point anomaly scores in ``[0, 1]`` from diffusion reconstruction error."""
        series = self._to_1d(data)
        scores = self._point_scores(series)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores.astype(np.float32),
            "is_anomaly": scores > self.threshold,
            "confidence": scores.astype(np.float32),
            "metadata": {"window": self.window, "n_steps": self.n_steps},
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-point fusion feature: the squashed reconstruction error.

        Args:
            data: Input series (1-D).

        Returns:
            ``(n_points, 1)`` float32 reconstruction-error scores.
        """
        series = self._to_1d(data)
        return self._point_scores(series).astype(np.float32).reshape(-1, 1)
