# Copyright (C) 2025 Steel Security Advisors LLC
"""Test equation profiles."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.equation_profiles import (
    BASELINE_PROFILE_ID,
    PHI_FIBRING_PROFILE_ID,
    QUIET_HORIZON_PROFILE_ID,
    available_equation_profiles,
    components_from_score_channels,
    score_runtime_equation_profile,
)


def test_runtime_profiles_are_bounded_and_distinct() -> None:
    raw = np.array([0.2, 0.5, 0.8])
    r = np.array([0.3, 0.6, 0.9])
    h = np.array([0.4, 0.5, 0.7])
    o = np.array([0.2, 0.7, 0.8])

    baseline, baseline_meta = score_runtime_equation_profile(
        raw, r, h, o, eta=0.96, profile_id=BASELINE_PROFILE_ID
    )
    quiet, quiet_meta = score_runtime_equation_profile(
        raw, r, h, o, eta=0.96, profile_id=QUIET_HORIZON_PROFILE_ID
    )

    assert BASELINE_PROFILE_ID in available_equation_profiles()
    assert QUIET_HORIZON_PROFILE_ID in available_equation_profiles()
    assert np.all((baseline >= 0.0) & (baseline <= 1.0))
    assert np.all((quiet >= 0.0) & (quiet <= 1.0))
    assert baseline_meta["applied"] is True
    assert quiet_meta["profile_id"] == QUIET_HORIZON_PROFILE_ID
    assert not np.allclose(baseline, quiet)


def test_none_profile_preserves_raw_scores() -> None:
    raw = np.array([0.1, 0.9])
    scored, metadata = score_runtime_equation_profile(raw, raw, raw, raw, profile_id=None)

    assert np.allclose(scored, raw)
    assert metadata == {"profile_id": None, "applied": False}


def test_unknown_profile_is_rejected() -> None:
    raw = np.array([0.1])
    with pytest.raises(ValueError, match="unknown equation profile"):
        score_runtime_equation_profile(raw, raw, raw, raw, profile_id="missing")


def test_components_from_score_channels_maps_detector_families() -> None:
    raw = np.array([0.5, 0.6])
    r, h, o = components_from_score_channels(
        {
            "statistical": np.array([0.2, 0.4]),
            "neurosymbolic": np.array([0.9, 0.8]),
            "spatial": np.array([0.1, 0.3]),
        },
        raw_scores=raw,
    )

    assert np.allclose(h, [0.2, 0.4])
    assert np.allclose(r, [0.1, 0.3])
    assert np.allclose(o, np.array([[0.5, 0.6], [0.9, 0.8]]).mean(axis=0))


def test_phi_fibring_uses_golden_ratio_base_split_when_independent() -> None:
    """Independent neural/equation signals get the canonical phi-weighted split."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.0, 1.0, 256)
    indep = rng.uniform(0.0, 1.0, 256)

    scored, meta = score_runtime_equation_profile(
        raw, indep, indep, indep, eta=0.96, profile_id=PHI_FIBRING_PROFILE_ID
    )

    assert PHI_FIBRING_PROFILE_ID in available_equation_profiles()
    assert meta["decorrelation_applied"] is False
    # phi/(1+phi) : 1/(1+phi) == 0.618... : 0.382...
    assert meta["neural_weight"] == pytest.approx(0.6180339887, abs=1e-6)
    assert meta["equation_weight"] == pytest.approx(0.3819660113, abs=1e-6)
    assert meta["neural_weight"] + meta["equation_weight"] == pytest.approx(1.0)
    assert np.all((scored >= 0.0) & (scored <= 1.0))


def test_phi_fibring_decorrelates_redundant_equation_signal() -> None:
    """A near-duplicate equation signal is down-weighted (no double-counting)."""
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.0, 1.0, 256)
    redundant = np.clip(raw + rng.normal(0.0, 0.02, 256), 0.0, 1.0)

    scored, meta = score_runtime_equation_profile(
        raw, redundant, redundant, redundant, eta=0.96, profile_id=PHI_FIBRING_PROFILE_ID
    )

    assert meta["correlation"] is not None and abs(meta["correlation"]) >= 0.85
    assert meta["decorrelation_applied"] is True
    # the renormalised weights still sum to 1 and stay a convex blend
    assert meta["neural_weight"] + meta["equation_weight"] == pytest.approx(1.0)
    # the redundant equation component is shrunk below its phi base weight
    assert meta["equation_weight"] < 0.3819660113
    assert np.all((scored >= 0.0) & (scored <= 1.0))


def test_phi_fibring_is_bounded_and_distinct_from_frozen_baseline() -> None:
    """With independent signals the phi split differs from the frozen baseline."""
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.0, 1.0, 128)
    # Independent components: no decorrelation fires, so the golden-ratio split
    # (0.618/0.382) must produce a different blend than the baseline (0.70/0.30).
    r = rng.uniform(0.0, 1.0, 128)
    h = rng.uniform(0.0, 1.0, 128)
    o = rng.uniform(0.0, 1.0, 128)

    baseline, _ = score_runtime_equation_profile(
        raw, r, h, o, eta=0.96, profile_id=BASELINE_PROFILE_ID
    )
    phi, phi_meta = score_runtime_equation_profile(
        raw, r, h, o, eta=0.96, profile_id=PHI_FIBRING_PROFILE_ID
    )

    assert phi_meta["profile_id"] == PHI_FIBRING_PROFILE_ID
    assert phi_meta["decorrelation_applied"] is False
    assert np.all((phi >= 0.0) & (phi <= 1.0))
    assert not np.allclose(baseline, phi)
