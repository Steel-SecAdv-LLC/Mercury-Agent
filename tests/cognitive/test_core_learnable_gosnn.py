# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

from omni_mercury_engine.core.learnable_gosnn import (
    LearnableGOSNN,
    ScalarCategory,
    ScalarState,
)


def test_learnable_gosnn_instantiation() -> None:
    g = LearnableGOSNN()
    assert g is not None


def test_scalar_category_importable() -> None:
    assert ScalarCategory is not None


def test_scalar_state_importable() -> None:
    assert ScalarState is not None


def test_importable_from_core() -> None:
    from omni_mercury_engine.core import (
        LearnableGOSNN as LG,
        ScalarCategory as SC,
        ScalarState as SS,
    )

    assert LG is LearnableGOSNN
    assert SC is ScalarCategory
    assert SS is ScalarState
