# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The integrated library heads expose the registry contract without theater.

Five real, tested detectors were importable but reachable from no served
surface: no manifest entry, no registry methods or differently named ones.
Each gained only a *delegating* view of the output it already computed --
``AdaptiveAnomalyDetector.predict`` returns the same scores its ``detect``
builds, ``TopologicalAnomalyDetector.extract_features`` the same
persistent-homology summary its ``score`` reports, and
``InformationGeometryDetector.fit``/``predict`` alias its
``fit_reference_distribution``/``detect_ood``.  These tests pin the
delegation: the new views must equal the original outputs exactly.  On the
previous code every one of these methods is missing (AttributeError).
"""

from __future__ import annotations

import numpy as np
import pytest

RNG = np.random.default_rng(11)


class TestAdaptiveAnomalyPredict:
    def test_predict_serves_the_exact_detect_scores(self) -> None:
        from omni_mercury_engine.detectors.adaptive_detector import AdaptiveAnomalyDetector

        data = RNG.normal(size=(200, 6))
        det = AdaptiveAnomalyDetector()
        det.fit(data)
        result = det.detect(data)
        out = AdaptiveAnomalyDetector().fit(data).predict(data)
        np.testing.assert_array_equal(out["anomaly_scores"], result.scores)
        np.testing.assert_array_equal(out["predictions"], result.predictions)
        assert out["threshold"] == result.threshold
        assert out["profile_used"] == result.profile_used


class TestTopologicalExtractFeatures:
    def test_features_are_the_score_summary_in_order(self) -> None:
        from omni_mercury_engine.detectors.topological_analysis import TopologicalAnomalyDetector

        ref = RNG.normal(size=(80, 3))
        test = RNG.normal(size=(40, 3)) + 2.0
        det = TopologicalAnomalyDetector()
        det.fit(ref)
        s = det.score(test)
        feats = det.extract_features(test)
        expected = np.array(
            [
                s["anomaly_score"],
                s["betti_0"],
                s["betti_1"],
                s["persistence_entropy_h0"],
                s["persistence_entropy_h1"],
                s["landscape_norm_h0"],
                s["landscape_norm_h1"],
                s["wasserstein_h0"],
                s["wasserstein_h1"],
            ]
        )
        np.testing.assert_allclose(feats, expected)

    def test_extract_before_fit_fails_closed(self) -> None:
        from omni_mercury_engine.detectors.topological_analysis import TopologicalAnomalyDetector

        with pytest.raises(RuntimeError, match="fit"):
            TopologicalAnomalyDetector().extract_features(RNG.normal(size=(10, 3)))


class TestInfoGeometryAliases:
    def test_fit_and_predict_delegate_to_the_real_methods(self) -> None:
        from omni_mercury_engine.detectors.info_geometry import InformationGeometryDetector

        ref = RNG.normal(size=(120, 4))
        test = RNG.normal(size=(30, 4)) + 3.0
        a = InformationGeometryDetector()
        assert a.fit(ref) is a
        via_alias = a.predict(test)
        b = InformationGeometryDetector()
        b.fit_reference_distribution(ref)
        direct = b.detect_ood(test)
        np.testing.assert_allclose(via_alias["sample_scores"], direct["sample_scores"])
        np.testing.assert_array_equal(via_alias["anomaly_scores"], via_alias["sample_scores"])
        assert via_alias["is_ood"] == direct["is_ood"]


class TestManifestCarriesTheHeads:
    def test_all_five_heads_are_manifest_entries_with_registry_seam(self) -> None:
        from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

        entries = {e.name: e for e in DETECTOR_MANIFEST}
        for name in (
            "extended_statistical",
            "adaptive_anomaly",
            "topological",
            "multivariate_ts",
            "info_geometry",
        ):
            assert name in entries, f"{name} missing from DETECTOR_MANIFEST"
            assert "registry" in entries[name].seams
