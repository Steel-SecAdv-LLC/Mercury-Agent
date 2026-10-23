# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fitting the spectral vibration detector on a single signal.

The reference spread was the standard deviation across the fit batch, so one
signal gave a spread of 0 (+1e-8): every other signal, including another
recording of the same process, scored 1.0 and was flagged.  One signal's
spread is now taken between equal blocks of it, each analysed as a whole
signal is, at an fft_size halved until every block holds two frames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.detectors.spectral_vibration import (
    SpectralVibrationDetector,
    VibrationSignatureType,
)

if TYPE_CHECKING:
    from pathlib import Path

FS = 1000.0


def _normal(rng: np.random.Generator, n: int) -> np.ndarray:
    """A 100 Hz machine tone in sensor noise."""
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * 100 * t) + 0.1 * rng.standard_normal(n)


def _faulted(rng: np.random.Generator, n: int) -> np.ndarray:
    """The same machine with a 42 Hz fault component."""
    t = np.arange(n) / FS
    return _normal(rng, n) + 0.3 * np.sin(2 * np.pi * 42 * t)


@pytest.fixture(autouse=True)
def _seeded_networks() -> None:
    # The detector's networks are untrained random projections.
    torch.manual_seed(0)


@pytest.mark.parametrize(
    ("length", "fft_size"),
    [(8192, 512), (1000, 64)],
    ids=["long", "short"],
)
def test_one_signal_separates_normal_from_faulted(length: int, fft_size: int) -> None:
    rng = np.random.default_rng(1)
    detector = SpectralVibrationDetector({"sample_rate": FS}).fit(_normal(rng, length))

    normal = [detector.detect(_normal(rng, length)) for _ in range(5)]
    faulted = [detector.detect(_faulted(rng, length)) for _ in range(5)]

    # Before: every normal recording scored 1.0, as the faulted ones do.
    assert max(r["anomaly_score"] for r in normal) < 0.5
    assert not any(r["is_anomaly"] for r in normal)
    assert min(r["anomaly_score"] for r in faulted) > 0.5
    assert {r["fft_size"] for r in normal + faulted} == {fft_size}


def test_a_batch_is_fitted_at_the_configured_fft_size() -> None:
    rng = np.random.default_rng(2)
    detector = SpectralVibrationDetector({"sample_rate": FS})
    detector.fit(_normal(rng, 1000))
    assert detector._spectral_config.fft_size == 64

    detector.fit(np.stack([_normal(rng, 1000) for _ in range(4)]))
    assert detector._spectral_config.fft_size == 1024
    assert detector.detect(_normal(rng, 1000))["fft_size"] == 1024


def test_too_short_a_signal_is_refused_and_leaves_no_state() -> None:
    detector = SpectralVibrationDetector({"sample_rate": FS})
    # 8 blocks of two 32-sample frames at overlap 0.5 need 8 * 48 = 384.
    with pytest.raises(DetectorException, match="needs 384"):
        detector.fit(_normal(np.random.default_rng(3), 383))
    assert not detector.is_fitted()
    assert detector._spectral_config.fft_size == 1024
    assert detector._signature_database[VibrationSignatureType.NORMAL] == []

    detector.fit(_normal(np.random.default_rng(3), 384))
    assert detector._spectral_config.fft_size == 32


def test_reduced_fft_size_survives_save_and_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    fitted = SpectralVibrationDetector({"sample_rate": FS}).fit(_normal(rng, 1000))
    probe = _faulted(rng, 1000)
    path = tmp_path / "spectral.pt"
    fitted.save(str(path))

    # A fresh instance is built at the configured 1024; the checkpoint's
    # network weights are shaped for 64 and load only once it is restored.
    restored = SpectralVibrationDetector({"sample_rate": FS})
    restored.load(str(path))

    assert restored._spectral_config.fft_size == 64
    np.testing.assert_array_equal(restored.detect(probe)["scores"], fitted.detect(probe)["scores"])
