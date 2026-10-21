# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A row's score must not depend on rows that come after it in the batch.

The engine runs two passes over every batch, ``extract_features`` and then
``predict``/``detect``. Components with a bounded memory of recent rows (the
neural model's hippocampal buffer, the directive detector's recursive memory)
were cleared before the first pass only, so the second pass scored the first
rows of a batch against the *last* rows of the same batch. Measured before
the fix: with only the final 100 of 300 rows replaced, 99 of the first 100
neural anomaly scores moved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.engine import OmniMercuryEngine

_N = 300
_TAIL = 100


def _batch(tail_seed: int) -> np.ndarray[Any, Any]:
    """Return a batch with a fixed head and a tail drawn from ``tail_seed``."""
    batch = np.random.default_rng(1).normal(size=(_N, 8)).astype(np.float32)
    batch[-_TAIL:] = np.random.default_rng(tail_seed).normal(size=(_TAIL, 8)).astype(np.float32)
    return batch


@pytest.fixture
def engine() -> OmniMercuryEngine:
    return OmniMercuryEngine(mode="fusion", auto_load_checkpoint=False, require_explicit_fit=False)


def test_model_scores_do_not_look_ahead(engine: OmniMercuryEngine) -> None:
    """Replace only the tail: the head's neural scores must not move."""
    _, scores_a = engine._extract_model_features(_batch(10))
    _, scores_b = engine._extract_model_features(_batch(11))
    np.testing.assert_array_equal(
        np.asarray(scores_a["neural"])[: _N - _TAIL],
        np.asarray(scores_b["neural"])[: _N - _TAIL],
    )


def test_model_scores_match_a_clean_predict(engine: OmniMercuryEngine) -> None:
    """The engine's score for a batch equals ``predict`` on a cleared model."""
    batch = _batch(10)
    model: Any = engine.models["neural"]
    _, scores = engine._extract_model_features(batch)
    model.reset_state()
    clean = engine._normalize_scores(model.predict(batch)["anomaly_scores"], _N)
    np.testing.assert_array_equal(np.asarray(scores["neural"]), np.asarray(clean))


def test_detector_memory_does_not_look_ahead(engine: OmniMercuryEngine) -> None:
    """The directive detector's recursive-memory scores obey the same rule."""
    directive = SigmaDirectiveDetector()
    directive.fit(np.random.default_rng(3).normal(size=(64, 8)))
    _, result_a = engine._cached_detector_features("directive", directive, _batch(10))
    _, result_b = engine._cached_detector_features("directive", directive, _batch(11))
    head = directive.memory_depth
    np.testing.assert_array_equal(
        np.asarray(result_a["rmd_scores"])[:head],
        np.asarray(result_b["rmd_scores"])[:head],
    )
