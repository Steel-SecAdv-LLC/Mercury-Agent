# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for UI/UX Anomaly Detection Module."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import time

import numpy as np
import pytest
import torch

from omni_mercury_engine.detectors.uiux_anomaly import (
    AnomalyCategory,
    BehaviorClassificationNetwork,
    ClickAnalysis,
    ClickPatternNetwork,
    InteractionSequenceEncoder,
    InteractionType,
    MouseTrajectoryNetwork,
    SessionAnalysis,
    UIUXAnomalyDetector,
    UserBehaviorClass,
    UserInteraction,
    analyze_navigation_flow,
    compute_click_heatmap,
    compute_fitts_law_time,
    detect_rage_clicks,
)


def create_test_interactions(
    count: int = 50,
    interaction_type: InteractionType = InteractionType.CLICK,
    time_gap: float = 0.5,
    start_time: float = 0.0,
) -> list[UserInteraction]:
    """Helper to create test interactions."""
    interactions = []
    current_time = start_time

    for i in range(count):
        interaction = UserInteraction(
            timestamp=current_time,
            interaction_type=interaction_type,
            x=100 + i * 10,
            y=200 + i * 5,
            element_id=f"element_{i % 10}",
            element_type="button",
            page_url=f"/page_{i % 3}",
            viewport_width=1920,
            viewport_height=1080,
        )
        interactions.append(interaction)
        current_time += time_gap

    return interactions


class TestUIUXAnomalyDetector:
    """Tests for UIUXAnomalyDetector."""

    def test_init_default_config(self) -> None:
        """Test initialization with default configuration."""
        detector = UIUXAnomalyDetector()
        assert detector is not None
        assert detector.threshold == 0.5
        assert not detector.is_fitted()

    def test_init_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = {
            "rage_click_threshold": 0.3,
            "rage_click_count": 5,
            "bot_detection_threshold": 0.8,
            "threshold": 0.6,
        }
        detector = UIUXAnomalyDetector(config)
        assert detector.threshold == 0.6
        assert detector._uiux_config.rage_click_threshold == 0.3
        assert detector._uiux_config.rage_click_count == 5

    def test_fit_single_session(self) -> None:
        """Test fitting on a single session."""
        detector = UIUXAnomalyDetector()
        interactions = create_test_interactions(count=30)

        detector.fit(interactions)
        assert detector.is_fitted()

    def test_fit_multiple_sessions(self) -> None:
        """Test fitting on multiple sessions."""
        detector = UIUXAnomalyDetector()

        sessions = [
            create_test_interactions(count=20, start_time=0),
            create_test_interactions(count=25, start_time=100),
            create_test_interactions(count=30, start_time=200),
        ]

        detector.fit(sessions)
        assert detector.is_fitted()

    def test_fit_empty_data_raises(self) -> None:
        """Test that fitting with empty data raises exception."""
        from omni_mercury_engine.core.exceptions import DetectorException

        detector = UIUXAnomalyDetector()
        with pytest.raises((ValueError, RuntimeError, DetectorException)):
            detector.fit([])

    def test_detect_normal_session(self) -> None:
        """Test detection on a normal session."""
        detector = UIUXAnomalyDetector({"threshold": 0.6})

        # Train on normal interactions
        train_interactions = create_test_interactions(count=50, time_gap=1.0)
        detector.fit(train_interactions)

        # Test with similar interactions
        test_interactions = create_test_interactions(count=30, time_gap=1.0)
        result = detector.detect(test_interactions)

        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "click_analysis" in result
        assert "scroll_analysis" in result
        assert "navigation_analysis" in result
        assert "session_analysis" in result
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_detect_rage_clicks(self) -> None:
        """Test detection of rage clicks."""
        detector = UIUXAnomalyDetector(
            {
                "rage_click_threshold": 0.2,
                "rage_click_count": 3,
            }
        )

        # Train on normal interactions
        train_interactions = create_test_interactions(count=50, time_gap=1.0)
        detector.fit(train_interactions)

        # Test with rage clicks (rapid clicking)
        base_time = 0.0
        rage_interactions = []

        # Normal interactions
        for i in range(10):
            rage_interactions.append(
                UserInteraction(
                    timestamp=base_time + i,
                    interaction_type=InteractionType.CLICK,
                    x=100,
                    y=100,
                    element_id="button",
                )
            )

        # Rage click sequence (5 clicks in 0.5 seconds)
        for i in range(5):
            rage_interactions.append(
                UserInteraction(
                    timestamp=base_time + 10 + i * 0.1,
                    interaction_type=InteractionType.CLICK,
                    x=500,
                    y=300,
                    element_id="slow_button",
                )
            )

        result = detector.detect(rage_interactions)
        click_analysis = result["click_analysis"]
        assert click_analysis.rage_clicks > 0
        assert AnomalyCategory.RAGE_CLICK.value in result.get("anomaly_categories", [])

    def test_detect_rapid_scrolling(self) -> None:
        """Test detection of rapid scrolling."""
        detector = UIUXAnomalyDetector(
            {
                "scroll_velocity_threshold": 1000.0,
            }
        )

        # Train on normal scrolling
        train_interactions = []
        for i in range(30):
            train_interactions.append(
                UserInteraction(
                    timestamp=i * 0.5,
                    interaction_type=InteractionType.SCROLL,
                    scroll_delta=100,
                )
            )
        detector.fit(train_interactions)

        # Test with rapid scrolling
        rapid_scrolls = []
        for i in range(10):
            rapid_scrolls.append(
                UserInteraction(
                    timestamp=i * 0.05,  # Very fast
                    interaction_type=InteractionType.SCROLL,
                    scroll_delta=500,
                )
            )

        result = detector.detect(rapid_scrolls)
        scroll_analysis = result["scroll_analysis"]
        assert scroll_analysis.rapid_scrolls > 0

    def test_detect_navigation_loop(self) -> None:
        """Test detection of navigation loops."""
        detector = UIUXAnomalyDetector(
            {
                "navigation_loop_threshold": 3,
            }
        )

        # Train
        train_interactions = create_test_interactions(count=30)
        detector.fit(train_interactions)

        # Test with navigation loop (visiting same page repeatedly)
        loop_interactions = []
        pages = ["/home", "/products", "/home", "/products", "/home", "/products"]
        for i, page in enumerate(pages):
            loop_interactions.append(
                UserInteraction(
                    timestamp=i * 2.0,
                    interaction_type=InteractionType.PAGE_VIEW,
                    page_url=page,
                )
            )

        result = detector.detect(loop_interactions)
        nav_analysis = result["navigation_analysis"]
        assert nav_analysis.navigation_loops > 0

    def test_detect_bot_behavior(self) -> None:
        """Test detection of bot-like behavior."""
        detector = UIUXAnomalyDetector(
            {
                "bot_detection_threshold": 0.6,
            }
        )

        # Train on human-like interactions (variable timing)
        train_interactions = []
        np.random.seed(42)
        for i in range(30):
            train_interactions.append(
                UserInteraction(
                    timestamp=i * (0.5 + np.random.rand() * 0.5),
                    interaction_type=InteractionType.CLICK,
                    x=100 + np.random.randint(-10, 10),
                    y=100 + np.random.randint(-10, 10),
                )
            )
        detector.fit(train_interactions)

        # Test with bot-like behavior (perfectly regular timing)
        bot_interactions = []
        for i in range(30):
            bot_interactions.append(
                UserInteraction(
                    timestamp=i * 0.5,  # Perfectly regular
                    interaction_type=InteractionType.CLICK,
                    x=100 + i,  # Linear movement
                    y=100 + i,
                )
            )

        result = detector.detect(bot_interactions)
        assert "bot_probability" in result
        # Bot probability should be elevated due to regular timing

    def test_behavior_classification(self) -> None:
        """Test user behavior classification."""
        detector = UIUXAnomalyDetector()

        interactions = create_test_interactions(count=30)
        detector.fit(interactions)

        result = detector.detect(interactions)
        assert "behavior_class" in result
        assert result["behavior_class"] in [e.value for e in UserBehaviorClass]

    def test_extract_features(self) -> None:
        """Test feature extraction for ML fusion."""
        detector = UIUXAnomalyDetector()

        interactions = create_test_interactions(count=30)
        detector.fit(interactions)

        features = detector.extract_features(interactions)
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2

    def test_short_session_handling(self) -> None:
        """Test handling of sessions shorter than minimum length."""
        detector = UIUXAnomalyDetector({"min_session_length": 10})

        interactions = create_test_interactions(count=30)
        detector.fit(interactions)

        # Very short session
        short_interactions = create_test_interactions(count=3)
        result = detector.detect(short_interactions)

        # Should return default result
        assert result["anomaly_score"] == 0.0
        assert result["is_anomaly"] is False

    def test_detect_not_fitted_raises(self) -> None:
        """Test that detection before fitting raises exception."""
        from omni_mercury_engine.core.exceptions import DetectorException

        detector = UIUXAnomalyDetector()
        with pytest.raises((ValueError, RuntimeError, DetectorException)):
            detector.detect(create_test_interactions(count=10))


class TestInteractionSequenceEncoder:
    """Tests for InteractionSequenceEncoder."""

    def test_forward_pass(self) -> None:
        """Test forward pass through encoder."""
        encoder = InteractionSequenceEncoder(
            input_dim=16,
            hidden_dim=32,
            output_dim=16,
        )

        batch_size = 2
        seq_len = 50
        features = torch.randn(batch_size, seq_len, 16)
        type_indices = torch.randint(0, len(InteractionType), (batch_size, seq_len))

        output = encoder(features, type_indices)
        assert output.shape == (batch_size, 16)


class TestMouseTrajectoryNetwork:
    """Tests for MouseTrajectoryNetwork."""

    def test_forward_pass(self) -> None:
        """Test forward pass through network."""
        network = MouseTrajectoryNetwork(
            hidden_dim=16,
            output_dim=8,
        )

        batch_size = 2
        seq_len = 100
        trajectory = torch.randn(batch_size, seq_len, 4)  # x, y, vx, vy

        features, bot_prob = network(trajectory)

        assert features.shape == (batch_size, 8)
        assert bot_prob.shape == (batch_size, 1)
        assert (bot_prob >= 0).all() and (bot_prob <= 1).all()


class TestClickPatternNetwork:
    """Tests for ClickPatternNetwork."""

    def test_forward_pass(self) -> None:
        """Test forward pass through network."""
        network = ClickPatternNetwork(
            input_dim=8,
            hidden_dim=16,
            output_dim=8,
        )

        batch_size = 2
        seq_len = 30
        click_features = torch.randn(batch_size, seq_len, 8)

        features, rage_score, dead_score = network(click_features)

        assert features.shape == (batch_size, 8)
        assert rage_score.shape == (batch_size, 1)
        assert dead_score.shape == (batch_size, 1)


class TestBehaviorClassificationNetwork:
    """Tests for BehaviorClassificationNetwork."""

    def test_forward_pass(self) -> None:
        """Test forward pass through network."""
        network = BehaviorClassificationNetwork(
            click_dim=16,
            mouse_dim=16,
            scroll_dim=8,
            nav_dim=8,
        )

        batch_size = 2
        click_features = torch.randn(batch_size, 16)
        mouse_features = torch.randn(batch_size, 16)
        scroll_features = torch.randn(batch_size, 8)
        nav_features = torch.randn(batch_size, 8)

        logits = network(click_features, mouse_features, scroll_features, nav_features)

        assert logits.shape == (batch_size, len(UserBehaviorClass))


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_fitts_law_time(self) -> None:
        """Test Fitts's Law computation."""
        distance = 200.0
        target_width = 50.0
        a = 0.1
        b = 0.15

        time = compute_fitts_law_time(distance, target_width, a, b)

        # MT = a + b * log2(D/W + 1)
        import math

        expected = a + b * math.log2(distance / target_width + 1)
        assert abs(time - expected) < 1e-10

    def test_compute_fitts_law_zero_width(self) -> None:
        """Test Fitts's Law with zero target width."""
        time = compute_fitts_law_time(100.0, 0.0)
        assert time == float("inf")

    def test_detect_rage_clicks_basic(self) -> None:
        """Test basic rage click detection."""
        base_time = 0.0

        # Create clicks with rage sequence
        clicks = []
        # Normal clicks
        for i in range(5):
            clicks.append(
                UserInteraction(
                    timestamp=base_time + i * 1.0,
                    interaction_type=InteractionType.CLICK,
                )
            )

        # Rage clicks (4 in 0.3 seconds)
        for i in range(4):
            clicks.append(
                UserInteraction(
                    timestamp=base_time + 5.0 + i * 0.1,
                    interaction_type=InteractionType.CLICK,
                )
            )

        sequences = detect_rage_clicks(clicks, time_threshold=0.3, count_threshold=3)

        assert len(sequences) > 0
        # Should detect the rage sequence starting around index 5

    def test_compute_click_heatmap(self) -> None:
        """Test click heatmap computation."""
        clicks = []
        # Clicks clustered in top-left
        for i in range(20):
            clicks.append(
                UserInteraction(
                    timestamp=i,
                    interaction_type=InteractionType.CLICK,
                    x=100 + np.random.randint(0, 50),
                    y=100 + np.random.randint(0, 50),
                    viewport_width=1920,
                    viewport_height=1080,
                )
            )

        heatmap = compute_click_heatmap(clicks, grid_size=10)

        assert heatmap.shape == (10, 10)
        # Top-left should have clicks
        assert heatmap.sum() == 20

    def test_analyze_navigation_flow(self) -> None:
        """Test navigation flow analysis."""
        interactions = [
            UserInteraction(
                timestamp=0, interaction_type=InteractionType.PAGE_VIEW, page_url="/home"
            ),
            UserInteraction(
                timestamp=1, interaction_type=InteractionType.PAGE_VIEW, page_url="/products"
            ),
            UserInteraction(
                timestamp=2, interaction_type=InteractionType.PAGE_VIEW, page_url="/cart"
            ),
            UserInteraction(
                timestamp=3, interaction_type=InteractionType.PAGE_VIEW, page_url="/checkout"
            ),
        ]

        flow = analyze_navigation_flow(interactions)

        assert len(flow["flow"]) == 4
        assert len(flow["unique_pages"]) == 4
        assert "/home" in flow["transition_matrix"]


class TestDataClasses:
    """Tests for data classes."""

    def test_user_interaction_creation(self) -> None:
        """Test UserInteraction creation."""
        interaction = UserInteraction(
            timestamp=time.time(),
            interaction_type=InteractionType.CLICK,
            x=100,
            y=200,
            element_id="submit_btn",
            element_type="button",
            page_url="/checkout",
        )

        assert interaction.interaction_type == InteractionType.CLICK
        assert interaction.x == 100

    def test_click_analysis_creation(self) -> None:
        """Test ClickAnalysis creation."""
        analysis = ClickAnalysis(
            total_clicks=50,
            rage_clicks=2,
            dead_clicks=5,
            double_click_rate=0.1,
            click_accuracy=0.9,
            click_density_map=np.zeros((10, 10)),
            click_timing_stats={"mean": 0.5, "std": 0.2, "min": 0.1, "max": 2.0},
        )

        assert analysis.total_clicks == 50
        assert analysis.rage_clicks == 2

    def test_session_analysis_creation(self) -> None:
        """Test SessionAnalysis creation."""
        analysis = SessionAnalysis(
            session_duration=300.0,
            total_interactions=150,
            engagement_score=0.8,
            frustration_indicators=[AnomalyCategory.RAGE_CLICK],
            behavior_class=UserBehaviorClass.TASK_FOCUSED,
            attention_score=0.9,
            task_completion=0.7,
        )

        assert analysis.session_duration == 300.0
        assert analysis.behavior_class == UserBehaviorClass.TASK_FOCUSED
