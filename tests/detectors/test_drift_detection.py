# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D16 wiring: BH FDR control and the chi-squared drift detector go live.

Benjamini-Hochberg control existed only in the evaluation harness, and the
chi-squared drift detector was importable but reachable from no live path;
both now serve the drift detectors directly.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestBenjaminiHochbergCorrection:
    """BH FDR control is now a live option on the per-feature KS test (D16)."""

    def test_bh_detects_many_small_shifts_that_bonferroni_misses(self) -> None:
        from omni_mercury_engine.detectors.drift import KolmogorovSmirnovDriftDetector

        rng = np.random.default_rng(42)
        n_features = 24
        reference = rng.normal(0.0, 1.0, (400, n_features))
        # Every feature shifts a little: individually p ~ 0.001-0.02, so the
        # Bonferroni bar (0.05/24 ~ 0.002) misses most, while the BH step-up
        # rejects the family.
        current = rng.normal(0.45, 1.0, (200, n_features))

        bonf = KolmogorovSmirnovDriftDetector(correction="bonferroni").fit(reference)
        bh = KolmogorovSmirnovDriftDetector(correction="bh_fdr").fit(reference)
        bonf_result = bonf.detect(current)
        bh_result = bh.detect(current)
        assert bh_result.is_drift is True
        assert (
            bh_result.details["n_drifting_features"] >= bonf_result.details["n_drifting_features"]
        )

    def test_bh_holds_on_stationary_data(self) -> None:
        from omni_mercury_engine.detectors.drift import KolmogorovSmirnovDriftDetector

        rng = np.random.default_rng(43)
        reference = rng.normal(0.0, 1.0, (400, 24))
        current = rng.normal(0.0, 1.0, (200, 24))
        result = KolmogorovSmirnovDriftDetector(correction="bh_fdr").fit(reference).detect(current)
        assert result.is_drift is False

    def test_unknown_correction_is_refused(self) -> None:
        from omni_mercury_engine.detectors.drift import KolmogorovSmirnovDriftDetector

        with pytest.raises(ValueError, match="correction"):
            KolmogorovSmirnovDriftDetector(correction="fdr")


class TestEnsembleCategoricalWiring:
    """The chi-squared drift detector is now reachable from the ensemble (D16)."""

    def test_categorical_shift_drives_the_ensemble(self) -> None:
        from omni_mercury_engine.detectors.drift import EnsembleDriftDetector

        rng = np.random.default_rng(7)
        n = 500
        numeric_ref = rng.normal(0.0, 1.0, (n, 2))
        cats_ref = rng.choice([0.0, 1.0, 2.0], size=(n, 1), p=[0.6, 0.3, 0.1])
        reference = np.hstack([numeric_ref, cats_ref])
        # Numeric marginals stay put; only the categorical mix changes.
        numeric_cur = rng.normal(0.0, 1.0, (n, 2))
        cats_cur = rng.choice([0.0, 1.0, 2.0], size=(n, 1), p=[0.1, 0.3, 0.6])
        current = np.hstack([numeric_cur, cats_cur])

        with_cats = EnsembleDriftDetector(
            voting="any", categorical_features=[2], covariance=False, principal_axes=False
        ).fit(reference)
        without = EnsembleDriftDetector(voting="any", covariance=False, principal_axes=False).fit(
            reference[:, :2]
        )
        assert with_cats.detect(current).is_drift is True
        assert without.detect(current[:, :2]).is_drift is False
