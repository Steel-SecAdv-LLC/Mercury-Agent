# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the geological trainers on real-data-sized inputs.

Two latent defects lived on the real-data training paths, which the synthetic
fallback masked and the (network-gated) real-data path never exercised:

1. ``train_waveform_analyzer`` / ``train_seismic_analyzer`` shuffled and batched
   with the ``n_samples`` *fallback-generation* parameter (default 1000) rather
   than the actual number of loaded samples, so ``torch.randperm(n_samples)``
   indexed the shorter real tensors out of bounds (and divided accuracy by the
   wrong denominator).
2. ``train_seismic_analyzer`` unpacked ``model(...)`` into two values, but
   ``SeismicWaveAnalyzer.forward`` returns a 4-tuple ``(prob, magnitude,
   p_wave, s_wave)`` -- so it raised ``ValueError`` on the first batch,
   independent of data size.

These tests feed each trainer a real-data-sized batch whose length differs from
``n_samples`` and assert training runs to completion.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import omni_mercury_engine.detectors.geological.disaster_detectors as dd


class TestWaveformTrainerRealSize:
    def test_trains_when_real_sample_count_differs_from_n_samples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = dd.generate_synthetic_tsunami_data(37)  # 37 != n_samples default
        monkeypatch.setattr(dd, "load_dart_buoy_data", lambda: real)

        model = dd.WaveformFFTAnalyzer()
        history = dd.train_waveform_analyzer(
            model,
            n_epochs=1,
            batch_size=16,
            n_samples=1000,
            use_real_data=True,
        )

        assert len(history["loss"]) == 1
        assert 0.0 <= history["accuracy"][-1] <= 1.0
        assert np.isfinite(history["loss"][-1])


class TestSeismicTrainerRealSize:
    def test_trains_when_real_sample_count_differs_from_n_samples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = dd.generate_synthetic_earthquake_data(29)  # 29 != n_samples default
        monkeypatch.setattr(dd, "load_usgs_earthquake_catalog", lambda: real)

        model = dd.SeismicWaveAnalyzer()
        # Would raise ValueError (4-tuple unpacked into 2) without the fix, and
        # IndexError (randperm(1000) on 29 rows) without the n_train fix.
        history = dd.train_seismic_analyzer(
            model,
            n_epochs=1,
            batch_size=16,
            n_samples=1000,
            use_real_data=True,
        )

        assert len(history["loss"]) == 1
        assert 0.0 <= history["accuracy"][-1] <= 1.0
        assert np.isfinite(history["loss"][-1])
