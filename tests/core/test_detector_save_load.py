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
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
from tests.core.test_detector_manifest_seams import (
    _FOUNDATION_EXTRA_MODULE,
    DRIVEN_MODALITIES,
    _construct,
    _payload,
)

if TYPE_CHECKING:
    from pathlib import Path

#: Learning detectors whose modality (video frames) needs vision-language
#: weights to ``detect``; each has a dedicated save/load test in
#: ``tests/test_vlm_detectors.py`` instead.
VIDEO_SAVE_LOAD_TESTED_ELSEWHERE = frozenset({"anyanomaly", "lavad"})


def _is_learning(entry: Any) -> bool:
    if "engine" not in entry.seams:
        return False
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    if not (isinstance(cls, type) and issubclass(cls, BaseDetector)):
        return False
    return any("fit" in klass.__dict__ for klass in cls.__mro__ if klass is not BaseDetector)


def _learning_detectors() -> list[Any]:
    return [
        pytest.param(entry, id=entry.name)
        for entry in DETECTOR_MANIFEST
        if _is_learning(entry) and entry.modality in DRIVEN_MODALITIES
    ]


TRAIN = np.random.default_rng(3).standard_normal((160, 4))

#: Detectors whose seam payload is a degenerate fit for a persistence check.
#: SpectralVibrationDetector fitted on one signal sets every reference spread
#: to 0 + 1e-8, so its scores are floating-point noise amplified 1e8-fold
#: (any other signal saturates at 1.0) -- an open defect, reported, not
#: hidden; persistence is checked on its other documented input, a batch of
#: reference signals ``[batch, num_samples]``.
SAVE_LOAD_PAYLOADS: dict[str, Any] = {
    "spectral_vibration": lambda: np.random.default_rng(7).normal(size=(8, 200)),
}


def test_every_learning_detector_is_round_tripped_somewhere() -> None:
    """Only the video entries leave this suite, and only to their own tests."""
    undriven = {
        e.name for e in DETECTOR_MANIFEST if _is_learning(e) and e.modality not in DRIVEN_MODALITIES
    }
    assert undriven == VIDEO_SAVE_LOAD_TESTED_ELSEWHERE


@pytest.mark.parametrize("entry", _learning_detectors())
def test_fitted_state_round_trips_exactly(entry: Any, tmp_path: Path) -> None:
    """fit -> save -> load into a fresh instance -> identical verdicts.

    Each detector is fitted and probed on its declared modality's example
    payload (the seam suite's), so every learning detector is round-tripped.
    """
    if entry.name in _FOUNDATION_EXTRA_MODULE:
        pytest.importorskip(
            _FOUNDATION_EXTRA_MODULE[entry.name],
            reason=f"{entry.name} needs the [foundation] extra, not installed in this lane",
        )
    detector = _construct(entry)
    payload = (
        SAVE_LOAD_PAYLOADS[entry.name]() if entry.name in SAVE_LOAD_PAYLOADS else _payload(entry)
    )
    detector.fit(payload)
    original = detector.detect(payload)

    path = tmp_path / f"{entry.name}.pt"
    detector.save(str(path))
    restored_detector = type(detector)().load(str(path))
    assert restored_detector.is_fitted()
    restored = restored_detector.detect(payload)

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
    # An attribute the class does not declare, standing in for an unbuilt module.
    lazy: Any = detector
    lazy._lazy_model = None
    with pytest.raises(ValueError, match="no constructed module"):
        detector._restore_learned_state({"_lazy_model": {"__module_state__": {}}})
    with pytest.raises(ValueError, match="no constructed module"):
        detector._restore_learned_state({"_never_built": {"__module_state__": {}}})
