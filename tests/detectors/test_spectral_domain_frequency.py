# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract + behavioural tests for ``SpectralDomainFrequency``.

This detector ships registered in ``DETECTOR_MANIFEST`` (BASE) but previously
had no dedicated test. These exercise the real fit -> extract_features -> detect
path on synthetic time-domain signals, plus the BaseDetector contract, so a
shipped detector is not relying on incidental coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors.spectral_domain_frequency import SpectralDomainFrequency


def _signals(n: int = 12, t: int = 512, fs: float = 1000.0, freq: float = 50.0, seed: int = 0):
    """``(n, t)`` time-domain sinusoids at ``freq`` Hz with light noise."""
    rng = np.random.RandomState(seed)
    tt = np.arange(t) / fs
    base = np.sin(2.0 * np.pi * freq * tt)
    return np.stack([base + rng.normal(0.0, 0.1, t) for _ in range(n)]).astype(np.float64)


def test_is_basedetector_and_instantiable_no_args() -> None:
    """cls() with no args must work — DETECTOR_MANIFEST auto-discovery relies on it."""
    det = SpectralDomainFrequency()
    assert isinstance(det, BaseDetector)


def test_fit_then_extract_features_is_per_sample() -> None:
    X = _signals()
    det = SpectralDomainFrequency().fit(X)
    assert det.is_fitted()
    feats = det.extract_features(X)
    # Per-sample leading dim is the contract _extract_fusion_features requires.
    assert feats.shape[0] == X.shape[0]
    assert feats.ndim == 2


def test_detect_returns_contract_keys() -> None:
    X = _signals()
    out = SpectralDomainFrequency().fit(X).detect(X)
    assert "anomaly_score" in out
    assert "is_anomaly" in out
    assert np.isfinite(float(out["anomaly_score"]))


def test_single_signal_1d_input_is_accepted() -> None:
    X = _signals()
    det = SpectralDomainFrequency().fit(X)
    feats = det.extract_features(_signals(n=1)[0])  # 1-D (T,)
    assert feats.shape[0] == 1
