# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test isotope predictor smoke."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_isotope_predictor_importable() -> None:
    from omni_mercury_engine.models.isotope_predictor import IsotopePredictor

    assert IsotopePredictor is not None


def test_nuclear_forensics_analyzer_importable() -> None:
    from omni_mercury_engine.models.isotope_predictor import NuclearForensicsAnalyzer

    assert NuclearForensicsAnalyzer is not None


def test_radiological_threat_assessor_importable() -> None:
    from omni_mercury_engine.models.isotope_predictor import RadiologicalThreatAssessor

    assert RadiologicalThreatAssessor is not None


def test_zero_u235_ratio_does_not_crash() -> None:
    """An explicit U235/U238 == 0 (fully depleted / no-U-235 measurement) drives
    the theoretical U-234 co-enrichment to zero; inferring the production method
    must not divide by zero.
    """
    from omni_mercury_engine.models.isotope_predictor import IsotopePredictor

    predictor = IsotopePredictor()
    result = predictor.predict_isotope_anomaly(
        {"isotope_ratios": {"U235_U238": 0.0, "U234_U238": 0.000055}}
    )
    assert result.nuclear_forensics["production_method"] == "chemical_or_unknown"


def test_off_contract_ml_features_do_not_crash() -> None:
    """An off-width ratio_features vector must not reach an unguarded matmul and
    raise a shape RuntimeError; the untrained ML lane returns the neutral prior.
    """
    import numpy as np

    from omni_mercury_engine.models.isotope_predictor import IsotopePredictor

    predictor = IsotopePredictor()
    result = predictor.predict_isotope_anomaly(
        {"isotope_ratios": {"U235_U238": 0.03}, "ratio_features": np.zeros(10)}
    )
    assert result.isotope_type == "natural_isotope"
    assert result.confidence == 0.0


def test_untrained_ml_lane_returns_neutral_not_fabricated() -> None:
    """The untrained IsotopeRatioAnalyzer must not fabricate isotope-type/threat
    classifications from random weights: even a contract-width feature vector
    yields the deterministic neutral prior, and the deterministic forensics lane
    carries the real enrichment determination.
    """
    import numpy as np

    from omni_mercury_engine.models.isotope_predictor import IsotopePredictor

    predictor = IsotopePredictor()
    ml = predictor.predict_isotope_anomaly(
        {"isotope_ratios": {"U235_U238": 0.03}, "ratio_features": np.zeros(64)}
    )
    assert ml.isotope_type == "natural_isotope"
    assert ml.confidence == 0.0

    # Deterministic forensics still flags highly-enriched material as anomalous.
    enriched = predictor.predict_isotope_anomaly(
        {"isotope_ratios": {"U235_U238": 0.9, "U234_U238": 0.007}}
    )
    assert enriched.nuclear_forensics["enrichment_category"] == "highly_enriched"
    assert enriched.anomaly_detected is True
