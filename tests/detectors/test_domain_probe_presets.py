"""Option E: domain probe preset routing tests."""
import numpy as np
from omni_mercury_engine.detectors.math_arrest.arrest import (
    AnomalyMathArrest,
    PROBE_PRESETS,
)
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def test_all_domain_presets_use_registered_probe_names() -> None:
    """Every probe name in every domain preset must exist in _PROBE_REGISTRY."""
    from omni_mercury_engine.detectors.math_arrest.arrest import _PROBE_REGISTRY

    domain_keys = [
        "infrastructure", "medical", "humanitarian", "security",
        "environmental", "financial", "tabular",
    ]
    for domain in domain_keys:
        assert domain in PROBE_PRESETS, f"Domain preset '{domain}' missing from PROBE_PRESETS"
        for probe_name in PROBE_PRESETS[domain]:
            assert probe_name in _PROBE_REGISTRY, (
                f"Domain '{domain}': probe '{probe_name}' not in _PROBE_REGISTRY"
            )


def test_domain_preset_activates_correct_probe_count() -> None:
    """Domain-preset AMA must activate exactly the probes in the preset."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)

    for domain in ["tabular", "security", "humanitarian"]:
        ama = AnomalyMathArrest(probes=domain, geometry_routing=False)
        ama.fit(X)
        expected = len(PROBE_PRESETS[domain])
        actual = len(ama._probes)
        assert actual == expected, (
            f"Domain '{domain}': expected {expected} probes, got {actual}"
        )


def test_mercury_domain_hint_reaches_ama() -> None:
    """When MercuryAnomalyDetector is given a domain hint, AMA must use
    the corresponding preset (not all-21)."""
    rng = np.random.RandomState(7)
    X = rng.randn(200, 8)
    det = MercuryAnomalyDetector(auto_validate=False, domain="tabular")
    det.fit(X)
    assert det._ama_detector is not None
    n_tabular_probes = len(PROBE_PRESETS["tabular"])
    assert len(det._ama_detector._probes) == n_tabular_probes, (
        f"Expected {n_tabular_probes} tabular probes, "
        f"got {len(det._ama_detector._probes)}"
    )
