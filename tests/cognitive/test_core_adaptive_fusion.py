"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

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
