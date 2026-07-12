# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test governed fusion substrate."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from omni_mercury_engine.core.governed_fusion import (
    InfoGeometryCertificate,
    pgd_flip_distance,
)
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def test_certificate_radius_is_sound_for_mahalanobis_boundary() -> None:
    cert = InfoGeometryCertificate(
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


def test_certificate_radius_sound_against_real_component_score() -> None:
    """Perturbing the input within rho never flips the REAL component score.

    This is the transparent soundness probe (Item 1): it certifies the
    info-geometry *component's* price level-set, so it perturbs the input and
    checks ``_compute_info_geometry_score`` — the component's actual decision
    signal — does not cross the component's own operating threshold inside the
    certified radius.
    """
    rng = np.random.default_rng(3)
    train = rng.normal(size=(96, 3))
    probe = np.vstack([train[:20], rng.normal(2.5, 1.0, size=(8, 3))])

    det = MercuryAnomalyDetector({"info_geometry_certificate_enabled": True}).fit(train)
    cert = det.detect(probe)["info_geometry_certificate"]

    thr = float(cert["component_threshold_score"])
    radii = np.asarray(cert["certified_l2_radius"], dtype=np.float64)
    clean = det._compute_info_geometry_score(probe)
    side_clean = clean > thr

    checked = 0
    for i in range(len(probe)):
        r = float(radii[i])
        if r <= 1e-9:
            continue
        for _ in range(16):
            d = rng.normal(size=int(probe.shape[1]))
            d /= np.linalg.norm(d)
            pert = probe[i] + d * (0.99 * r)
            score = float(det._compute_info_geometry_score(pert.reshape(1, -1))[0])
            assert (score > thr) == bool(side_clean[i])
            checked += 1
    assert checked > 0  # the certified radius was non-trivial for some rows


def test_certificate_default_off_is_byte_exact_reduction() -> None:
    rng = np.random.default_rng(11)
    train = rng.normal(size=(64, 3))
    probe = train[:16].copy()

    enabled = (
        MercuryAnomalyDetector({"info_geometry_certificate_enabled": True}).fit(train).detect(probe)
    )
    default = MercuryAnomalyDetector().fit(train).detect(probe)  # default OFF

    assert "info_geometry_certificate" in enabled
    assert "info_geometry_certificate" not in default
    assert "fusion_certificate" not in default  # old overclaiming key is gone
    assert_allclose(enabled["scores"], default["scores"], rtol=0.0, atol=0.0)
    assert np.array_equal(enabled["is_anomaly"], default["is_anomaly"])


def test_certificate_certifies_component_not_fused_verdict() -> None:
    """The payload is explicitly scoped to the component, not the fusion."""
    rng = np.random.default_rng(5)
    train = rng.normal(size=(80, 4))
    det = MercuryAnomalyDetector({"info_geometry_certificate_enabled": True}).fit(train)
    cert = det.detect(train[:24])["info_geometry_certificate"]
    assert "component_verdict" in cert
    assert "NOT the fused/gated verdict" in cert["certifies"]
    # component_verdict is the info-geometry component's own level-set decision.
    info_geo = det._compute_info_geometry_score(train[:24])
    thr = float(cert["component_threshold_score"])
    assert np.array_equal(np.asarray(cert["component_verdict"], dtype=bool), info_geo > thr)
