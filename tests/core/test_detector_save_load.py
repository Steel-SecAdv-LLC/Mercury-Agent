# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every learning detector saves what it learned and restores it exactly.

Regression: 33 manifest detectors learned from data (``fit``) but had no
persistence at all -- every deployment refit from scratch, and nothing a
detector had learned could survive a restart.  The shared mechanism is
``BaseDetector.save``/``load`` (primitives-only torch payload, loaded under
``weights_only=True``); this suite drives it over every engine-seam
learning detector in the manifest and fails when a fitted detector cannot
round-trip, or when the restored copy scores differently from the original.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError

if TYPE_CHECKING:
    from pathlib import Path


def _learning_detectors() -> list[Any]:
    params = []
    for entry in DETECTOR_MANIFEST:
        if "engine" not in entry.seams:
            continue
        cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
        if not (isinstance(cls, type) and issubclass(cls, BaseDetector)):
            continue
        if any("fit" in klass.__dict__ for klass in cls.__mro__ if klass is not BaseDetector):
            params.append(pytest.param(entry, id=entry.name))
    return params


TRAIN = np.random.default_rng(3).standard_normal((160, 4))
PROBE = np.random.default_rng(4).standard_normal((24, 4))


@pytest.mark.parametrize("entry", _learning_detectors())
def test_fitted_state_round_trips_exactly(entry: Any, tmp_path: Path) -> None:
    """fit -> save -> load into a fresh instance -> identical verdicts."""
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    detector = cls()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            detector.fit(TRAIN)
        except Exception as exc:  # modality-specific trainers keep their contracts
            pytest.skip(f"{entry.name} does not fit on a generic matrix: {exc}")
        try:
            original = detector.detect(PROBE)
        except (InputModalityError, InvalidObservationError) as exc:
            pytest.skip(f"{entry.name} refuses a generic probe at detect: {exc}")

        path = tmp_path / f"{entry.name}.pt"
        detector.save(str(path))
        restored_detector = cls().load(str(path))
        assert restored_detector.is_fitted()
        restored = restored_detector.detect(PROBE)

    original_scores = np.asarray(
        original.get("scores", [original.get("anomaly_score", original.get("anomaly_prob"))]),
        dtype=np.float64,
    )
    restored_scores = np.asarray(
        restored.get("scores", [restored.get("anomaly_score", restored.get("anomaly_prob"))]),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        restored_scores,
        original_scores,
        rtol=1e-6,
        atol=1e-9,
        err_msg=f"{entry.name}: restored detector scores differ from the fitted original",
    )


def test_checkpoint_refuses_another_detectors_state(tmp_path: Path) -> None:
    from omni_mercury_engine.detectors.digital_twin import DigitalTwinResidualDetector
    from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector

    detector = DigitalTwinResidualDetector()
    detector.fit(TRAIN[:, 0])
    path = tmp_path / "twin.pt"
    detector.save(str(path))
    with pytest.raises(ValueError, match="refusing to restore"):
        TemporalAnomalyDetector().load(str(path))


def test_module_weights_without_a_constructed_module_are_refused() -> None:
    """Weights for an unbuilt (lazy ``None``) module must not be dropped or crash opaquely.

    Regression: a missing attribute silently received the raw wrapper dict,
    and a lazy ``None`` died with ``'NoneType' object has no attribute
    'load_state_dict'``.
    """
    from omni_mercury_engine.detectors.digital_twin import DigitalTwinResidualDetector

    detector = DigitalTwinResidualDetector()
    detector._lazy_model = None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="no constructed module"):
        detector._restore_learned_state({"_lazy_model": {"__module_state__": {}}})
    with pytest.raises(ValueError, match="no constructed module"):
        detector._restore_learned_state({"_never_built": {"__module_state__": {}}})
