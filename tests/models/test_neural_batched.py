# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The batched neural model reproduces the per-row model it replaced.

``_ReferenceNeural`` is the original implementation, one Python iteration per
row, kept here as the oracle. The batched model must match it on every shape
and dtype, carry streaming state across calls the same way, and fail on
non-finite input the same way. ``_row_histogram`` is checked against
``np.histogram`` itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.models.neural import NeuralCognitiveModel, _row_histogram


class _ReferenceNeural(NeuralCognitiveModel):
    """The per-row implementation the batched one replaced."""

    def _hippocampal_memory(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        memory_features = np.zeros((data.shape[0], 16), dtype=np.float32)
        for i in range(data.shape[0]):
            pattern = data[i]
            self.memory_buffer.append(pattern)
            same_shape = [p for p in self.memory_buffer if p.shape == pattern.shape]
            buffer_array = np.array(same_shape)
            similarities = np.dot(buffer_array, pattern) / (
                np.linalg.norm(buffer_array, axis=1) * np.linalg.norm(pattern) + 1e-8
            )
            memory_features[i, :8] = np.histogram(similarities, bins=8)[0].astype(np.float32) / len(
                same_shape
            )
            memory_features[i, 8:] = [
                np.mean(similarities),
                np.std(similarities),
                np.max(similarities),
                np.min(similarities),
                np.median(similarities),
                len(self.memory_buffer) / self.memory_capacity,
                np.sum(similarities > 0.7),
                np.sum(similarities < 0.3),
            ]
        return memory_features

    def _prefrontal_executive(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        executive_features = np.zeros((data.shape[0], 16), dtype=np.float32)
        for i in range(data.shape[0]):
            pattern = data[i]
            executive_features[i, :8] = [
                np.mean(pattern),
                np.std(pattern),
                np.max(pattern),
                np.min(pattern),
                np.median(pattern),
                np.percentile(pattern, 25),
                np.percentile(pattern, 75),
                np.ptp(pattern),
            ]
            if len(pattern) >= 2:
                diffs = np.diff(pattern)
                std_diffs = np.std(diffs)
                executive_features[i, 8:] = [
                    np.mean(diffs),
                    std_diffs,
                    np.max(np.abs(diffs)),
                    np.sum(diffs > 0) / len(diffs),
                    np.sum(diffs < 0) / len(diffs),
                    np.mean(np.abs(diffs)),
                    np.sum(np.abs(diffs) > std_diffs),
                    len(pattern),
                ]
            else:
                executive_features[i, 8:15] = 0.0
                executive_features[i, 15] = len(pattern)
        return executive_features

    def _amygdala_processing(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        emotional_features = np.zeros((data.shape[0], 16), dtype=np.float32)
        for i in range(data.shape[0]):
            pattern = data[i]
            fft_result = np.fft.fft(pattern)
            power_spectrum = np.abs(fft_result) ** 2
            emotional_features[i, :8] = np.histogram(power_spectrum, bins=8)[0].astype(
                np.float32
            ) / len(pattern)
            emotional_features[i, 8:] = [
                np.mean(power_spectrum),
                np.std(power_spectrum),
                np.max(power_spectrum),
                np.sum(power_spectrum > np.mean(power_spectrum)),
                np.mean(np.abs(fft_result)),
                np.std(np.abs(fft_result)),
                np.sum(pattern > 0) / len(pattern),
                np.sum(pattern < 0) / len(pattern),
            ]
        return emotional_features


def _both(**config: Any) -> tuple[NeuralCognitiveModel, _ReferenceNeural]:
    return NeuralCognitiveModel(config), _ReferenceNeural(config)


def _assert_close(new: np.ndarray[Any, Any], ref: np.ndarray[Any, Any]) -> None:
    assert new.shape == ref.shape
    np.testing.assert_allclose(new, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
@pytest.mark.parametrize("shape", [(1, 8), (7, 5), (150, 8), (2500, 3), (40, 64), (30, 2)])
def test_matches_the_per_row_model(dtype: Any, shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(shape[0] * 31 + shape[1])
    if np.issubdtype(dtype, np.integer):
        data = rng.integers(-500, 500, size=shape).astype(dtype)
    else:
        data = (rng.normal(size=shape) * 20).astype(dtype)
    new, ref = _both()
    _assert_close(new.extract_features(data), ref.extract_features(data))


@pytest.mark.parametrize("amplitude", [1.0, 3000.0, 1e5])
def test_a_spike_row_does_not_discard_the_batch(amplitude: float) -> None:
    """A single spike has a flat spectrum; ``np.histogram`` refuses to bin it.

    The per-row model raised ``ValueError`` there, so the engine dropped the
    neural feature group for the whole batch. The batched model keeps every
    other row's features unchanged and bins the spike row like a constant one.
    """
    batch = np.random.default_rng(13).normal(size=(64, 8)).astype(np.float32)
    spiked = batch.copy()
    spiked[10] = 0.0
    spiked[10, 3] = amplitude
    with pytest.raises(ValueError, match="Too many bins"):
        _ReferenceNeural().extract_features(spiked)

    clean = NeuralCognitiveModel().extract_features(batch)
    features = NeuralCognitiveModel().extract_features(spiked)
    others = np.arange(len(batch)) != 10
    # Executive and emotional blocks are per-row: nothing else may move.
    np.testing.assert_array_equal(features[others, 16:], clean[others, 16:])
    emotional_histogram = features[10, 32:40]
    assert emotional_histogram[4] == 1.0 and emotional_histogram.sum() == 1.0


def test_float32_matches_to_single_precision() -> None:
    data = np.random.default_rng(5).normal(size=(400, 8)).astype(np.float32)
    new, ref = _both()
    got, want = new.extract_features(data), ref.extract_features(data)
    # float32 similarities may land either side of a bin edge by one ULP.
    assert np.mean(~np.isclose(got, want, rtol=1e-4, atol=1e-5)) < 5e-4


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((12, 6)),
        np.ones((12, 6)),
        np.tile(np.arange(6.0), (12, 1)),
        np.arange(40.0).reshape(20, 2),
        np.arange(5.0).reshape(5, 1),
        np.array([5.0]),
        np.array([[1e-23, 2e-23, 3e-23, 1e-23]]),
    ],
    ids=["zeros", "ones", "identical-rows", "ramp", "single-column", "1d", "denormal-power"],
)
def test_degenerate_inputs(data: np.ndarray[Any, Any]) -> None:
    new, ref = _both()
    _assert_close(new.extract_features(data), ref.extract_features(data))


def test_predict_matches() -> None:
    data = np.random.default_rng(6).normal(size=(120, 8))
    new, ref = _both()
    got, want = new.predict(data), ref.predict(data)
    for key in ("anomaly_scores", "memory_scores", "executive_scores", "emotional_scores"):
        _assert_close(got[key], want[key])


def test_streaming_state_carries_over_like_the_reference() -> None:
    rng = np.random.default_rng(7)
    new, ref = _both(memory_capacity=40)
    for rows in (30, 1, 55, 200, 3):
        batch = rng.normal(size=(rows, 8))
        _assert_close(new._hippocampal_memory(batch), ref._hippocampal_memory(batch))
        assert len(new.memory_buffer) == len(ref.memory_buffer)
        for got, want in zip(new.memory_buffer, ref.memory_buffer, strict=True):
            np.testing.assert_array_equal(got, want)


def test_mixed_length_stream_matches() -> None:
    """Patterns of another length hold window slots but are never compared."""
    rng = np.random.default_rng(8)
    new, ref = _both(memory_capacity=25)
    for width in (5, 7, 7, 5, 6):
        batch = rng.normal(size=(9, width))
        _assert_close(new.extract_features(batch), ref.extract_features(batch))


def test_buffer_does_not_alias_the_callers_array() -> None:
    model = NeuralCognitiveModel()
    batch = np.random.default_rng(9).normal(size=(10, 4))
    model._hippocampal_memory(batch)
    stored = [p.copy() for p in model.memory_buffer]
    batch[:] = 0.0
    for kept, original in zip(model.memory_buffer, stored, strict=True):
        np.testing.assert_array_equal(kept, original)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_input_fails_like_the_reference(bad: float) -> None:
    data = np.random.default_rng(10).normal(size=(20, 6))
    data[3, 2] = bad
    new, ref = _both()
    with pytest.raises(ValueError):
        ref.extract_features(data)
    with pytest.raises(ValueError):
        new.extract_features(data)


def test_empty_batch() -> None:
    assert NeuralCognitiveModel()._hippocampal_memory(np.zeros((0, 4))).shape == (0, 16)


class TestRowHistogram:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_matches_numpy_on_random_rows(self, dtype: Any) -> None:
        rng = np.random.default_rng(11)
        values = np.concatenate(
            [
                rng.normal(size=(300, 33)),
                rng.integers(0, 4, size=(300, 33)),  # heavy ties
                rng.exponential(size=(300, 33)) * 1e-30,  # tiny range
                np.round(rng.normal(size=(300, 33)), 1),
            ]
        ).astype(dtype)
        got = _row_histogram(values, 8)
        want = np.stack([np.histogram(row, bins=8)[0] for row in values])
        np.testing.assert_array_equal(got, want)

    def test_matches_numpy_on_edge_values(self) -> None:
        # Values sitting exactly on, and one ULP either side of, the bin edges.
        edges = np.linspace(0.0, 1.0, 9)
        row = np.concatenate([edges, np.nextafter(edges, 2), np.nextafter(edges, -1)])
        values = np.stack([row, row * 3 - 1, np.full_like(row, 2.5)])
        want = np.stack([np.histogram(r, bins=8)[0] for r in values])
        np.testing.assert_array_equal(_row_histogram(values, 8), want)

    def test_mask_selects_the_rows_own_entries(self) -> None:
        rng = np.random.default_rng(12)
        values = rng.normal(size=(50, 20))
        valid = rng.random((50, 20)) < 0.6
        valid[:, 0] = True
        got = _row_histogram(values, 8, valid)
        want = np.stack([np.histogram(v[m], bins=8)[0] for v, m in zip(values, valid, strict=True)])
        np.testing.assert_array_equal(got, want)

    def test_non_finite_range_raises(self) -> None:
        values = np.array([[0.0, 1.0, np.nan]])
        with pytest.raises(ValueError):
            _row_histogram(values, 8)
