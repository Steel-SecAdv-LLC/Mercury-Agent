"""Item E: certificates are threaded through return values, not engine state.

The engine used to stash ``self._last_detector_certificates`` during
``_extract_detector_features`` and read it back later in
``detect_with_fusion`` — a staleness / re-entrancy hazard: two interleaved
``detect`` calls could read each other's certificates.  The certificate dict is
now returned from ``_extract_detector_features`` and threaded straight into the
result, so there is no shared mutable certificate state to cross-contaminate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

if TYPE_CHECKING:
    from omni_mercury_engine.engine import OmniMercuryEngine

pytestmark = pytest.mark.timeout(120)


def _engine_with_single_stat_detector() -> OmniMercuryEngine:
    """A fusion engine whose only detector is a fitted, cert-enabled stat det.

    Restricting the detector set keeps the test fast and deterministic while
    still exercising the real ``_extract_detector_features`` threading path.
    """
    from omni_mercury_engine.engine import OmniMercuryEngine

    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    rng = np.random.default_rng(0)
    det = MercuryAnomalyDetector({"info_geometry_certificate_enabled": True})
    det.fit(rng.normal(size=(128, 4)))
    engine.detectors = {"statistical": det}
    return engine


def test_no_stale_certificate_state_on_engine() -> None:
    engine = _engine_with_single_stat_detector()
    assert not hasattr(engine, "_last_detector_certificates")


def test_interleaved_extract_does_not_cross_contaminate_certificates() -> None:
    engine = _engine_with_single_stat_detector()
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=(24, 4))
    b = rng.normal(4.0, 1.0, size=(24, 4))  # far off-manifold -> different prices

    # Interleave A, B, A.  Each call must return its OWN certificate.
    _, _, cert_a1 = engine._extract_detector_features(a)
    _, _, cert_b = engine._extract_detector_features(b)
    _, _, cert_a2 = engine._extract_detector_features(a)

    assert set(cert_a1) == {"statistical"} == set(cert_b)
    price_a = np.asarray(cert_a1["statistical"]["price"], dtype=float)
    price_b = np.asarray(cert_b["statistical"]["price"], dtype=float)
    price_a2 = np.asarray(cert_a2["statistical"]["price"], dtype=float)

    # B's certificate reflects B's input, not A's; re-running A reproduces A.
    assert not np.allclose(price_a, price_b)
    assert np.array_equal(price_a, price_a2)
    # And no shared mutable state was introduced by the calls.
    assert not hasattr(engine, "_last_detector_certificates")
