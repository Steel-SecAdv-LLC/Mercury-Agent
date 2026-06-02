from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.equation_profiles import (
    BASELINE_PROFILE_ID,
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
