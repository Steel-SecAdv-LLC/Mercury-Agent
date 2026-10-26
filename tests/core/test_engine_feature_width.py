# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A fitted engine refuses input whose feature width it was not fitted on.

Before this, a width mismatch surfaced two ways, both wrong: a 1-D input
reached a torch ``Linear`` and died with a raw ``mat1 and mat2 shapes cannot
be multiplied`` error, and a 2-D input with the wrong column count passed
silently through detectors that tolerate any width, scoring columns against
the wrong fitted statistics. These tests pin the refusal, its message, the
checkpoint round-trip of the fitted width, and that legacy checkpoints
(no recorded width) and unfitted engines stay uncheckeable and unchecked.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.engine import OmniMercuryEngine

if TYPE_CHECKING:
    from pathlib import Path

WIDTH = 8


@pytest.fixture(scope="module")
def fitted_engine() -> OmniMercuryEngine:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine()
        rng = np.random.default_rng(0)
        engine.fit_fusion(rng.normal(size=(60, WIDTH)), epochs=2, calibrate=False)
    return engine


def test_detect_batch_refuses_wrong_width(fitted_engine: OmniMercuryEngine) -> None:
    bad = np.random.default_rng(1).normal(size=(10, WIDTH - 3))
    with pytest.raises(ValueError, match=rf"{WIDTH - 3} feature column\(s\).*fitted on {WIDTH}"):
        fitted_engine.detect_batch(bad)


def test_detect_refuses_wrong_width(fitted_engine: OmniMercuryEngine) -> None:
    bad = np.random.default_rng(2).normal(size=(10, WIDTH + 2))
    with pytest.raises(ValueError, match=f"fitted on {WIDTH}"):
        fitted_engine.detect(bad)


def test_detect_names_the_one_dimensional_ambiguity(
    fitted_engine: OmniMercuryEngine,
) -> None:
    """A 1-D input gets the reshape hint instead of a torch shape error."""
    with pytest.raises(ValueError, match=r"1-D array is ambiguous.*reshape\(1, -1\)"):
        fitted_engine.detect(np.random.default_rng(3).normal(size=5))


def test_detect_with_fusion_reads_one_dimensional_input_as_one_sample(
    fitted_engine: OmniMercuryEngine,
) -> None:
    """detect_with_fusion's documented 1-D convention (one sample) still works."""
    result = fitted_engine.detect_with_fusion(np.random.default_rng(4).normal(size=WIDTH))
    assert "anomaly_prob" in result or "anomaly_probs" in result


def test_matched_width_still_detects(fitted_engine: OmniMercuryEngine) -> None:
    good = np.random.default_rng(5).normal(size=(6, WIDTH))
    results = fitted_engine.detect_batch(good)
    assert len(results) == 6


def test_unfitted_engine_is_not_checked() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine()
        result = engine.detect(np.random.default_rng(6).normal(size=(30, 4)))
    assert "detectors" in result


def test_fitted_width_round_trips_through_the_checkpoint(
    fitted_engine: OmniMercuryEngine, tmp_path: Path
) -> None:
    path = str(tmp_path / "fusion.pt")
    fitted_engine.save_model(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = OmniMercuryEngine()
        loaded.load_model(path)
    assert loaded._fitted_feature_width == WIDTH
    with pytest.raises(ValueError, match=f"fitted on {WIDTH}"):
        loaded.detect_batch(np.random.default_rng(7).normal(size=(5, WIDTH + 1)))


def test_legacy_checkpoint_without_width_disables_the_check(
    fitted_engine: OmniMercuryEngine, tmp_path: Path
) -> None:
    import torch

    path = tmp_path / "legacy.pt"
    fitted_engine.save_model(str(path))
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    del checkpoint["fitted_feature_width"]
    torch.save(checkpoint, str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = OmniMercuryEngine()
        loaded.load_model(str(path))
    assert loaded._fitted_feature_width is None


def test_dict_input_is_not_width_checked(fitted_engine: OmniMercuryEngine) -> None:
    """Domain-keyed dict input has no columnar width; it must not be refused."""
    fitted_engine._check_feature_width({"domain": "medical"}, "detect")
