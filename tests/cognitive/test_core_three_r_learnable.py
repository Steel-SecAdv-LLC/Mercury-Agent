# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test core three r learnable."""

from __future__ import annotations

from omni_mercury_engine.core.three_r.learnable_fusion import (
    Learnable3RConfig,
    Learnable3REngine,
    Learnable3RResult,
)


def test_learnable_3r_config_instantiation() -> None:
    config = Learnable3RConfig()
    assert config is not None


def test_learnable_3r_engine_instantiation() -> None:
    engine = Learnable3REngine()
    assert engine is not None


def test_learnable_3r_result_importable() -> None:
    assert Learnable3RResult is not None


def test_importable_from_three_r() -> None:
    from omni_mercury_engine.core.three_r import (
        Learnable3RConfig as LC,
        Learnable3REngine as LE,
        Learnable3RResult as LR,
    )

    assert LC is Learnable3RConfig
    assert LE is Learnable3REngine
    assert LR is Learnable3RResult
