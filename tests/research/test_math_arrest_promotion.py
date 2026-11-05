# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The math_arrest promotion decision, pinned to its evidence.

``docs/DORMANCY_LEDGER.md`` §2b withheld a default-set slot pending a
marginal-gain ablation against the live ensemble. That ablation is now made
(``benchmarks/math_arrest_ensemble_marginal.py``, 5 recorded ADBench tables x 8
seeds) and it returns **HOLD**: mean ΔAUC +0.0011 against a pre-registered
+0.002 bar, bootstrap 95% CI [-0.00029, +0.00249].

The 40-cell ablation is far too expensive for a test lane, so what is pinned
here is what the verdict rests on and what would have to change to overturn it:

* ``math_arrest`` is reachable -- a manifest entry with a working seam -- and
  **not** in the engine's default set. Both halves matter: the first is what
  §2b achieved, the second is what the ablation refused.
* The feature group is not degenerate on tabular input. This is load-bearing:
  if it were constant, the ablation would have measured a dead group rather
  than fusion, and the HOLD would say nothing about the instrument.

The HOLD is a measurement, not a permanent judgement. §2b records that four of
five tables leave the trained fusion ≤0.013 of headroom, so it means "not
demonstrated on this corpus" rather than "demonstrated absent".
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

#: The ablation's pre-registered bar and its measured outcome, from
#: ``artifacts/math_arrest_ensemble_marginal_8seed.json``.
PRE_REGISTERED_BAR = 0.002
MEASURED_MEAN_DELTA = 0.0011
MEASURED_CELLS = 40


def test_math_arrest_is_reachable_but_not_in_the_default_set() -> None:
    """Both halves of the disposition, together.

    A later commit that promotes it must delete this assertion, which forces
    the ablation to be re-run and §2b re-derived rather than letting the
    default set drift.
    """
    from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
    from omni_mercury_engine.engine import OmniMercuryEngine

    entries = {entry.name for entry in DETECTOR_MANIFEST}
    assert (
        "math_arrest" in entries
    ), "math_arrest left DETECTOR_MANIFEST; §2b's reachability work is undone"

    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    assert "math_arrest" not in engine.detectors, (
        "math_arrest is now in the engine's default detector set. The "
        "marginal-gain ablation measured mean dAUC "
        f"{MEASURED_MEAN_DELTA:+.4f} over {MEASURED_CELLS} cells against a "
        f"pre-registered bar of +{PRE_REGISTERED_BAR}, with a bootstrap 95% CI "
        "of [-0.00029, +0.00249] -- it did not clear the bar. Re-run "
        "benchmarks/math_arrest_ensemble_marginal.py and update "
        "DORMANCY_LEDGER 2b before promoting."
    )


def test_the_opt_in_seam_the_hold_leaves_open_still_works() -> None:
    """HOLD is not removal: the registration seam must stay usable.

    This is the same disposition the revived k-means detector has, and the
    seam is the whole point of 'reachable but not default'.
    """
    from omni_mercury_engine.detectors.math_arrest.detector import MathArrestDetector
    from omni_mercury_engine.engine import OmniMercuryEngine

    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.register_detector("math_arrest", MathArrestDetector())
    assert "math_arrest" in engine.detectors


def test_the_tabular_feature_group_is_not_degenerate() -> None:
    """Why the ablation measured fusion rather than a dead group.

    Every probe is a series equation and ``_to_1d`` reduces a matrix to its
    per-row mean, so a constant feature group on tabular input is the obvious
    failure mode -- and would have invalidated the ablation entirely. Measured:
    21 columns, a median of 542 distinct values over 548 rows, and at most two
    constant columns on any recorded table.
    """
    from omni_mercury_engine.detectors.math_arrest.detector import MathArrestDetector
    from tests.recorded_data import adbench, stratified_split

    features, labels = adbench("cardio")
    train_idx, test_idx = stratified_split(labels, 0)
    detector = MathArrestDetector()
    detector.fit(features[train_idx])
    group = detector.extract_features(features[test_idx])

    assert group.shape == (
        len(test_idx),
        21,
    ), f"feature group is {group.shape}, expected ({len(test_idx)}, 21)"
    distinct = [len(np.unique(np.round(group[:, j], 9))) for j in range(group.shape[1])]
    constant = sum(1 for d in distinct if d == 1)
    assert constant <= 2, (
        f"{constant} of 21 feature columns are constant on tabular input. The "
        "ablation behind the HOLD assumed a live group; a degenerate one means "
        "it measured nothing about the instrument and must be re-run."
    )
    assert int(np.median(distinct)) > len(test_idx) // 2, (
        f"median distinct values {int(np.median(distinct))} over {len(test_idx)} "
        "rows: the group has collapsed toward constant, see above"
    )
