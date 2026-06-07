from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from omni_mercury_engine.core.governed_fusion import (
    JointCertificate,
    pgd_flip_distance,
)
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def test_certificate_radius_is_sound_for_mahalanobis_boundary() -> None:
    cert = JointCertificate(
        loc=np.zeros(2, dtype=np.float64),
        precision=np.eye(2, dtype=np.float64),
        p_tau=2.0,
    )
    point = np.asarray([[3.0, 0.0]], dtype=np.float64)
    radius = float(cert.certified_radius(point)[0])

    rng = np.random.default_rng(7)
    for _ in range(120):
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        candidate = point + direction.reshape(1, -1) * (0.99 * radius)
        assert cert.price(candidate)[0] > cert.p_tau

    assert pgd_flip_distance(cert, point[0], steps=120, step_size=0.02) >= 0.95 * radius
    assert int(cert.witness_channel(point)[0]) == 0


def test_certificate_absence_preserves_detector_verdict_and_scores() -> None:
    rng = np.random.default_rng(11)
    train = rng.normal(size=(64, 3))
    probe = train[:16].copy()

    with_cert = (
        MercuryAnomalyDetector({"fusion_certificates_enabled": True}).fit(train).detect(probe)
    )
    without_cert = (
        MercuryAnomalyDetector({"fusion_certificates_enabled": False}).fit(train).detect(probe)
    )

    assert "fusion_certificate" in with_cert
    assert "fusion_certificate" not in without_cert
    assert_allclose(with_cert["scores"], without_cert["scores"], rtol=0.0, atol=0.0)
    assert np.array_equal(with_cert["is_anomaly"], without_cert["is_anomaly"])
