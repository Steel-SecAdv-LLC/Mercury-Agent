# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Mercury-native matrix profile matches a brute-force reference and STUMPY.

The native backend (numpy STOMP) is what the MatrixProfileDetector runs on
when STUMPY is not installed, so these tests pin that it is the same
computation: profile values against an O(l^2 m) brute force on every input
family that has bitten (flat runs, large offsets), the STOMP refresh path,
index semantics by distance (ties may pick either neighbour), the
constant-window convention, and the adapter's backend selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.ml.matrix_profile import (
    MatrixProfileResult,
    matrix_profile,
    sliding_dot_product,
)

WINDOW = 12


def _brute_force(
    series: np.ndarray[Any, Any], m: int, normalize: bool
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Distance matrix and profile by explicit z-normalization, O(l^2 m)."""
    t = np.asarray(series, dtype=np.float64)
    length = t.shape[0] - m + 1
    zone = int(np.ceil(m / 4))
    windows = np.lib.stride_tricks.sliding_window_view(t, m).astype(np.float64)
    if normalize:
        mu = windows.mean(axis=1, keepdims=True)
        sd = windows.std(axis=1, keepdims=True)
        # A constant window is one whose values are all identical (ptp == 0):
        # the two-pass std of identical float64 values can come out ~1e-16
        # from mean rounding, so `sd == 0` would miss exact flats.
        constant = np.ptp(windows, axis=1) == 0.0
        z = np.where(sd > 0, (windows - mu) / np.where(sd == 0, 1.0, sd), 0.0)
        dist = np.sqrt(np.maximum(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1), 0.0))
        dist[np.ix_(constant, constant)] = 0.0
        dist[np.ix_(constant, ~constant)] = np.sqrt(m)
        dist[np.ix_(~constant, constant)] = np.sqrt(m)
    else:
        dist = np.sqrt(np.maximum(((windows[:, None, :] - windows[None, :, :]) ** 2).sum(-1), 0.0))
    for i in range(length):
        dist[i, max(0, i - zone) : i + zone + 1] = np.inf
    profile = dist.min(axis=1)
    return dist, profile


def _case(name: str) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(7)
    cases = {
        "random": rng.normal(size=220),
        "offset_1e8": 1e8 + rng.normal(size=200),
        "flat_runs": np.concatenate(
            [rng.normal(size=70), np.full(40, 3.7), rng.normal(size=70), np.full(30, 3.7)]
        ),
        "sine": np.sin(np.linspace(0, 12 * np.pi, 240)) + 0.05 * rng.normal(size=240),
    }
    return np.asarray(cases[name], dtype=np.float64)


@pytest.mark.parametrize("name", ["random", "offset_1e8", "flat_runs", "sine"])
@pytest.mark.parametrize("normalize", [True, False])
def test_profile_matches_brute_force(name: str, normalize: bool) -> None:
    series = _case(name)
    dist, profile = _brute_force(series, WINDOW, normalize)
    result = matrix_profile(series, WINDOW, normalize=normalize)
    finite = np.isfinite(profile)
    assert np.array_equal(np.isfinite(result.profile), finite)
    np.testing.assert_allclose(result.profile[finite], profile[finite], atol=1e-9)
    # Index semantics by distance: a tie may pick either neighbour, so pin
    # that the distance AT the chosen index equals the row minimum.
    rows = np.flatnonzero(finite)
    chosen = dist[rows, result.index[rows]]
    np.testing.assert_allclose(chosen, profile[rows], atol=1e-9)


@pytest.mark.parametrize("name", ["random", "flat_runs"])
def test_left_right_indices_match_brute_force(name: str) -> None:
    series = _case(name)
    dist, _ = _brute_force(series, WINDOW, normalize=True)
    result = matrix_profile(series, WINDOW)
    length = dist.shape[0]
    for i in range(length):
        left_min = dist[i, :i].min() if i > 0 else np.inf
        right_min = dist[i, i + 1 :].min() if i < length - 1 else np.inf
        if np.isfinite(left_min):
            assert result.left_index[i] >= 0
            assert result.left_index[i] < i
            np.testing.assert_allclose(dist[i, result.left_index[i]], left_min, atol=1e-9)
        else:
            assert result.left_index[i] == -1
        if np.isfinite(right_min):
            assert result.right_index[i] > i
            np.testing.assert_allclose(dist[i, result.right_index[i]], right_min, atol=1e-9)
        else:
            assert result.right_index[i] == -1


def test_stomp_refresh_path_matches_brute_force(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows after an FFT refresh (every _REFRESH_INTERVAL rows) stay exact."""
    import omni_mercury_engine.ml.matrix_profile as mp_module

    monkeypatch.setattr(mp_module, "_REFRESH_INTERVAL", 7)
    series = _case("random")
    _, profile = _brute_force(series, WINDOW, normalize=True)
    result = mp_module.matrix_profile(series, WINDOW)
    finite = np.isfinite(profile)
    np.testing.assert_allclose(result.profile[finite], profile[finite], atol=1e-9)


def test_constant_window_convention() -> None:
    """Flat vs flat is 0; flat vs varying is sqrt(m)."""
    rng = np.random.default_rng(3)
    series = np.concatenate([np.full(40, 2.0), rng.normal(size=80), np.full(40, 2.0)])
    result = matrix_profile(series, WINDOW)
    windows = np.lib.stride_tricks.sliding_window_view(series, WINDOW)
    flat = windows.std(axis=1) == 0.0
    # Interior flat windows find the other flat run at distance 0.
    assert np.allclose(result.profile[np.flatnonzero(flat)], 0.0, atol=1e-9)
    # A varying window whose nearest neighbour is flat would read sqrt(m);
    # none here should beat a real match, so all finite values stay <= 2*sqrt(m).
    finite = np.isfinite(result.profile)
    assert np.all(result.profile[finite] <= 2.0 * np.sqrt(WINDOW) + 1e-9)


def test_planted_discord_is_the_profile_peak() -> None:
    rng = np.random.default_rng(11)
    series = np.sin(np.linspace(0, 40 * np.pi, 600)) + 0.05 * rng.normal(size=600)
    series[300:315] += 2.5
    result = matrix_profile(series, 16)
    peak = int(np.argmax(np.where(np.isfinite(result.profile), result.profile, -np.inf)))
    assert 284 <= peak <= 315  # a window overlapping the planted anomaly


def test_sliding_dot_product_matches_direct() -> None:
    rng = np.random.default_rng(5)
    query, series = rng.normal(size=9), rng.normal(size=100)
    direct = np.array([query @ series[j : j + 9] for j in range(92)])
    np.testing.assert_allclose(sliding_dot_product(query, series), direct, atol=1e-10)


def test_input_contract() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="1-D"):
        matrix_profile(rng.normal(size=(40, 2)), WINDOW)
    with pytest.raises(ValueError, match="window_size"):
        matrix_profile(rng.normal(size=40), 2)
    with pytest.raises(ValueError, match="longer than"):
        matrix_profile(rng.normal(size=WINDOW), WINDOW)
    bad = rng.normal(size=40)
    bad[7] = np.nan
    with pytest.raises(ValueError, match="finite"):
        matrix_profile(bad, WINDOW)


def test_result_shapes_and_types() -> None:
    series = _case("random")
    result = matrix_profile(series, WINDOW)
    assert isinstance(result, MatrixProfileResult)
    length = series.shape[0] - WINDOW + 1
    for arr in (result.profile, result.index, result.left_index, result.right_index):
        assert arr.shape == (length,)
    assert result.profile.dtype == np.float64
    assert result.index.dtype == np.int64
    assert result.window_size == WINDOW
    # First window has no left neighbour, last has no right neighbour.
    assert result.left_index[0] == -1
    assert result.right_index[-1] == -1


def test_matches_stumpy_when_installed() -> None:
    stumpy = pytest.importorskip("stumpy")
    for name in ("random", "flat_runs", "sine"):
        series = _case(name)
        reference = stumpy.stump(series, m=WINDOW)
        result = matrix_profile(series, WINDOW)
        ref_profile = reference[:, 0].astype(np.float64)
        finite = np.isfinite(result.profile) & np.isfinite(ref_profile)
        np.testing.assert_allclose(result.profile[finite], ref_profile[finite], atol=1e-5)


def test_adapter_uses_native_backend_when_stumpy_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detector runs the real native computation, not a mock, without stumpy."""
    import builtins

    from omni_mercury_engine.models.foundation.matrix_profile import (
        MatrixProfileConfig,
        MatrixProfileDetector,
    )

    real_import = builtins.__import__

    def _no_stumpy(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "stumpy" or name.startswith("stumpy."):
            raise ImportError("stumpy blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_stumpy)
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name, *a, **k: (
            (_ for _ in ()).throw(ImportError("blocked"))
            if name.startswith("stumpy")
            else real_import(name)
        ),
    )

    detector = MatrixProfileDetector(MatrixProfileConfig(window_size=16, use_gpu=False))
    rng = np.random.default_rng(11)
    series = np.sin(np.linspace(0, 40 * np.pi, 600)) + 0.05 * rng.normal(size=600)
    series[300:315] += 2.5
    result = detector.detect(series)
    # Initialization ran inside detect with stumpy blocked, so the detector
    # must have selected the native backend, not merely fallen through.
    assert detector._stumpy_available is False
    assert set(result) >= {"scores", "is_anomaly", "discords", "matrix_profile"}
    native = matrix_profile(series, 16)
    np.testing.assert_allclose(result["matrix_profile"], native.profile, atol=1e-9)
    top = result["discord_indices"][0]
    assert 284 <= top <= 315
