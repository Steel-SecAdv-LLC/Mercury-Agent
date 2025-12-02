"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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

"""Tests for regenerative architecture module"""

import pytest
import numpy as np
from omni_anomaly_engine.core.regenerative import (
    RegenerativeArchitecture,
    PermaculturePrinciple,
    FeedbackLoop,
)


def test_regenerative_initialization():
    """Test regenerative architecture initialization"""
    regen = RegenerativeArchitecture(enable_closed_loops=True)
    assert regen.enable_closed_loops is True
    assert len(regen.feedback_loops) == 0
    assert len(regen.permaculture_metrics) == 12


def test_create_feedback_loop():
    """Test creating feedback loops"""
    regen = RegenerativeArchitecture()

    loop = regen.create_feedback_loop(
        loop_id="test_loop",
        input_metric="accuracy",
        output_metric="confidence",
        gain=0.5,
        is_positive=True,
    )

    assert isinstance(loop, FeedbackLoop)
    assert loop.loop_id == "test_loop"
    assert "test_loop" in regen.feedback_loops


def test_apply_feedback_loops():
    """Test applying feedback loops to metrics"""
    regen = RegenerativeArchitecture(enable_closed_loops=True)

    regen.create_feedback_loop(
        loop_id="stabilize",
        input_metric="target",
        output_metric="actual",
        gain=0.5,
        is_positive=False,
    )

    metrics = {"target": 1.0, "actual": 0.5}
    updated = regen.apply_feedback_loops(metrics)

    assert "actual" in updated
    assert updated["actual"] != metrics["actual"]


def test_permaculture_principles():
    """Test applying permaculture principles"""
    regen = RegenerativeArchitecture()

    context = {"data": np.random.randn(100, 10)}

    updated = regen.apply_permaculture_principle(PermaculturePrinciple.USE_EDGES, context)

    assert "edge_cases" in updated
    assert "anomaly_potential" in updated
    assert regen.permaculture_metrics[PermaculturePrinciple.USE_EDGES] > 0


def test_net_positive_score():
    """Test net-positive score calculation"""
    regen = RegenerativeArchitecture()

    context = {
        "value_metrics": {"total_value": 10.0},
        "resource_usage": 5.0,
        "accuracy": 0.95,
        "efficiency": 1.0,
        "ethical_score": 0.9,
    }

    score = regen.calculate_net_positive_score(context)

    assert score > 0
