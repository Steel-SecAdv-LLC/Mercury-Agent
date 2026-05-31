"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Deterministic tests for GCP ingestion + pre-registered statistics (WS-D).

No network: the trusted-allowlist gate rejects the GCP host before any socket,
so ``fetch_egg_stream`` returns ``reachable=False`` reproducibly. The
statistics are checked against their closed-form null behaviour. No psi claim is
made or tested -- only that the plumbing and the null statistics are correct.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.models.gcp_ingest import (
    BITS_PER_TRIAL,
    NULL_MEAN,
    NULL_STD,
    egg_sums_to_z,
    fetch_egg_stream,
    network_variance,
    stouffer_z,
    synthetic_null_streams,
)


def test_ingestion_reports_unreachable_honestly() -> None:
    res = fetch_egg_stream(2020, 1, 1)
    assert res.reachable is False
    assert res.egg_sums is None
    assert res.reason  # a concrete reason string
    assert "noosphere" in res.provenance["url"]


def test_null_constants() -> None:
    assert NULL_MEAN == BITS_PER_TRIAL / 2.0
    assert abs(NULL_STD - (BITS_PER_TRIAL / 4.0) ** 0.5) < 1e-9


def test_synthetic_null_is_deterministic_and_shaped() -> None:
    a = synthetic_null_streams(120, 32, seed=7)
    b = synthetic_null_streams(120, 32, seed=7)
    assert a.shape == (120, 32)
    assert np.array_equal(a, b)


def test_network_variance_matches_chi_square_df() -> None:
    """Under the null, mean per-second network variance ~= egg count."""
    z = egg_sums_to_z(synthetic_null_streams(5000, 50, seed=0))
    assert abs(float(network_variance(z).mean()) - 50.0) < 3.0


def test_stouffer_z_is_null_on_random_streams() -> None:
    """A true-random stream must not be flagged: |Z| stays small."""
    zvals = [
        stouffer_z(egg_sums_to_z(synthetic_null_streams(300, 64, seed=k))) for k in range(8)
    ]
    # All within a generous null band; not a single |Z| > 3.5 across 8 seeds.
    assert all(abs(z) < 3.5 for z in zvals)
    assert abs(float(np.mean(zvals))) < 1.5
