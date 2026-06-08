# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test core adaptive fusion."""

from __future__ import annotations

import dataclasses

from omni_mercury_engine.core.adaptive_fusion import (
    AttentionVisualization,
    UncertaintyEstimate,
)


def test_uncertainty_estimate_is_dataclass() -> None:
    assert dataclasses.is_dataclass(UncertaintyEstimate)


def test_attention_visualization_is_dataclass() -> None:
    assert dataclasses.is_dataclass(AttentionVisualization)


def test_importable_from_core() -> None:
    from omni_mercury_engine.core import (
        AttentionVisualization as AV,
        UncertaintyEstimate as UE,
    )

    assert AV is AttentionVisualization
    assert UE is UncertaintyEstimate
