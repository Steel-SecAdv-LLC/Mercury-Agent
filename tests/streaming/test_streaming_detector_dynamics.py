# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The streaming detector must score the newest point with real dynamics.

Scoring a lone point via ``detect(point.reshape(1, -1))`` makes the kinematic
component fall back to its ``n < 2`` constant of 0.5, so a third of every
streamed score carried no information. The newest point must instead be scored
as the last element of its trailing window, where velocity/acceleration/jerk
are real.
"""

from __future__ import annotations

import numpy as np
import pytest

_torch_available = False
try:
    import torch  # noqa: F401

    _torch_available = True
except ImportError:
    pass

if not _torch_available:
    pytest.skip("torch required for StreamingDetector", allow_module_level=True)

from omni_mercury_engine.streaming.streaming_detector import StreamingDetector


def _stream_with_velocity_change() -> np.ndarray:
    """A steady warmup followed by a region of real acceleration."""
    rng = np.random.default_rng(3)
    steady = rng.normal(0.0, 0.05, 70)
    # A smoothly accelerating ramp: velocity rises, so acceleration/jerk are
    # nonzero and the kinematic component has something real to report.
    t = np.arange(50)
    accelerating = 0.001 * t**2 + rng.normal(0.0, 0.05, 50)
    tail = accelerating[-1] + rng.normal(0.0, 0.05, 30)
    return np.concatenate([steady, accelerating, tail]).astype(np.float64)


def test_newest_point_kinematic_contribution_is_not_constant() -> None:
    """The newest-point kinematic score must vary, not sit pinned at 0.5.

    Fails today: a single-row ``detect`` makes ``kinematic_scores[-1]`` the
    ``n < 2`` constant 0.5 for every streamed point.
    """
    sd = StreamingDetector(window_size=200, min_samples=40, refit_interval=10_000)
    kin_newest: list[float] = []
    for value in _stream_with_velocity_change():
        result = sd.ingest([float(value)])
        if result is not None and "kinematic_scores" in result:
            kin_newest.append(float(np.asarray(result["kinematic_scores"])[-1]))

    kin = np.asarray(kin_newest)
    assert kin.size > 20, "stream did not produce enough scored points"
    assert not np.allclose(kin, 0.5), "kinematic contribution is pinned at the n<2 constant 0.5"
    assert float(np.std(kin)) > 1e-3, "newest-point kinematic contribution is constant"


def test_newest_point_score_responds_to_dynamics() -> None:
    """The newest-point kinematic score must rise when the dynamics change.

    The accelerating region (indices ~70-120) must drive the newest-point
    kinematic score meaningfully above its steady-warmup level.
    """
    sd = StreamingDetector(window_size=200, min_samples=40, refit_interval=10_000)
    series = _stream_with_velocity_change()
    kin_by_index: dict[int, float] = {}
    for i, value in enumerate(series):
        result = sd.ingest([float(value)])
        if result is not None and "kinematic_scores" in result:
            kin_by_index[i] = float(np.asarray(result["kinematic_scores"])[-1])

    steady = np.array([v for i, v in kin_by_index.items() if 40 <= i < 68])
    accel = np.array([v for i, v in kin_by_index.items() if 72 <= i < 120])
    assert steady.size and accel.size
    assert accel.max() > steady.mean() + 0.05, (
        f"accel kinematic max {accel.max():.3f} did not exceed steady "
        f"mean {steady.mean():.3f}; newest point is not responding to dynamics"
    )
