"""
Mercury Agent ♱
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

from __future__ import annotations

"""
UI/UX Anomaly Detection Module for Mercury Agent.

Comprehensive internal implementation for detecting anomalies in user
interactions and interface behavior. This module analyzes:

1. User Interaction Patterns:
   - Click patterns (frequency, location, timing)
   - Scroll behavior (speed, direction, patterns)
   - Mouse/cursor movements (velocity, trajectories)
   - Touch gestures (for mobile interfaces)
   - Keyboard input patterns (timing, sequences)

2. Navigation Flow Analysis:
   - Page transition sequences
   - Session path anomalies
   - Abandonment detection
   - Loop detection (user confusion)

3. UI Element Behavior:
   - Response time anomalies
   - Layout shift detection
   - Interaction failure detection
   - Accessibility issue indicators

4. Session Analysis:
   - Engagement metrics
   - Attention patterns
   - Frustration indicators (rage clicks, rapid scrolling)
   - Bot vs human behavior classification

5. Temporal Patterns:
   - Time-of-day effects
   - Session duration anomalies
   - Inter-action timing

This is a complete internal implementation for UI/UX anomaly detection,
providing the capability to identify usability issues, detect unusual
user behavior, and improve user experience through data-driven insights.

Research foundations:
- Human-Computer Interaction (HCI) principles
- Fitts's Law for pointing device movement
- Cognitive load theory
- User behavior modeling
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.utils.constants import MathematicalConstants

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Enumerations
# =============================================================================

PHI = MathematicalConstants.GOLDEN_RATIO.value


class InteractionType(Enum):
    """Types of user interactions."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    SCROLL = "scroll"
    MOUSE_MOVE = "mouse_move"
    KEY_PRESS = "key_press"
    KEY_HOLD = "key_hold"
    TOUCH = "touch"
    SWIPE = "swipe"
    PINCH = "pinch"
    HOVER = "hover"
    FOCUS = "focus"
    BLUR = "blur"
    PAGE_VIEW = "page_view"
    FORM_SUBMIT = "form_submit"
    DRAG = "drag"
    DROP = "drop"


class AnomalyCategory(Enum):
    """Categories of detected UI/UX anomalies."""

    RAGE_CLICK = "rage_click"
    DEAD_CLICK = "dead_click"
    RAPID_SCROLL = "rapid_scroll"
    NAVIGATION_LOOP = "navigation_loop"
    SESSION_ABANDONMENT = "session_abandonment"
    SLOW_INTERACTION = "slow_interaction"
    ERRATIC_MOVEMENT = "erratic_movement"
    UNUSUAL_TIMING = "unusual_timing"
    BOT_BEHAVIOR = "bot_behavior"
    FRUSTRATION_SIGNAL = "frustration_signal"
    ACCESSIBILITY_BARRIER = "accessibility_barrier"
    UI_FAILURE = "ui_failure"
    LAYOUT_CONFUSION = "layout_confusion"
    ATTENTION_LOSS = "attention_loss"


class UserBehaviorClass(Enum):
    """Classification of user behavior patterns."""

    NORMAL = "normal"
    NOVICE = "novice"
    EXPERT = "expert"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    EXPLORING = "exploring"
    TASK_FOCUSED = "task_focused"
    DISTRACTED = "distracted"
    AUTOMATED = "automated"
    MALICIOUS = "malicious"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class UserInteraction:
    """Single user interaction event.

    Attributes:
        timestamp: Unix timestamp of interaction
        interaction_type: Type of interaction
        x: X coordinate (if applicable)
        y: Y coordinate (if applicable)
        element_id: ID of interacted element
        element_type: Type of UI element (button, link, etc.)
        page_url: Current page URL
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height
        scroll_delta: Scroll amount (if scroll event)
        key_code: Key code (if keyboard event)
        duration: Duration of interaction (if applicable)
        metadata: Additional metadata
    """

    timestamp: float
    interaction_type: InteractionType
    x: float | None = None
    y: float | None = None
    element_id: str | None = None
    element_type: str | None = None
    page_url: str | None = None
    viewport_width: int = 1920
    viewport_height: int = 1080
    scroll_delta: float = 0.0
    key_code: int | None = None
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UIUXConfig:
    """Configuration for UI/UX anomaly detection.

    Attributes:
        rage_click_threshold: Clicks within this time window = rage click
        rage_click_count: Number of clicks to trigger rage click
        dead_click_timeout: Time after click with no response = dead click
        scroll_velocity_threshold: Scroll speed threshold for rapid scroll
        navigation_loop_threshold: Visits to same page indicating confusion
        session_timeout: Session timeout in seconds
        mouse_velocity_threshold: Threshold for erratic mouse movement
        timing_z_score_threshold: Z-score threshold for timing anomalies
        bot_detection_threshold: Threshold for bot behavior classification
        min_session_length: Minimum interactions for analysis
        fitts_law_a: Fitts's Law parameter a (intercept)
        fitts_law_b: Fitts's Law parameter b (slope)
        threshold: Overall anomaly detection threshold
    """

    rage_click_threshold: float = 0.5  # seconds
    rage_click_count: int = 3
    dead_click_timeout: float = 3.0  # seconds
    scroll_velocity_threshold: float = 3000.0  # pixels/second
    navigation_loop_threshold: int = 3
    session_timeout: float = 1800.0  # 30 minutes
    mouse_velocity_threshold: float = 5000.0  # pixels/second
    timing_z_score_threshold: float = 3.0
    bot_detection_threshold: float = 0.7
    min_session_length: int = 5
    fitts_law_a: float = 0.1  # Fitts's Law intercept
    fitts_law_b: float = 0.15  # Fitts's Law slope
    threshold: float = 0.5
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClickAnalysis:
    """Analysis results for click patterns.

    Attributes:
        total_clicks: Total number of clicks
        rage_clicks: Number of rage click events
        dead_clicks: Number of dead click events
        double_click_rate: Rate of double clicks
        click_accuracy: Accuracy based on element targeting
        click_density_map: Heatmap-like density distribution
        click_timing_stats: Statistics on inter-click timing
    """

    total_clicks: int
    rage_clicks: int
    dead_clicks: int
    double_click_rate: float
    click_accuracy: float
    click_density_map: np.ndarray
    click_timing_stats: dict[str, float]


@dataclass
class ScrollAnalysis:
    """Analysis results for scroll behavior.

    Attributes:
        total_scrolls: Total scroll events
        rapid_scrolls: Number of rapid scroll events
        scroll_reversals: Number of direction changes
        average_velocity: Average scroll velocity
        scroll_depth: Maximum scroll depth reached
        reading_patterns: Detected reading pattern score
    """

    total_scrolls: int
    rapid_scrolls: int
    scroll_reversals: int
    average_velocity: float
    scroll_depth: float
    reading_patterns: float


@dataclass
class NavigationAnalysis:
    """Analysis results for navigation patterns.

    Attributes:
        pages_visited: Number of unique pages
        navigation_loops: Number of detected loops
        backtrack_rate: Rate of going back
        path_efficiency: How direct was the navigation path
        abandonment_risk: Likelihood of session abandonment
        confusion_score: Score indicating user confusion
    """

    pages_visited: int
    navigation_loops: int
    backtrack_rate: float
    path_efficiency: float
    abandonment_risk: float
    confusion_score: float


@dataclass
class SessionAnalysis:
    """Complete session analysis results.

    Attributes:
        session_duration: Total session duration
        total_interactions: Total number of interactions
        engagement_score: Overall engagement score
        frustration_indicators: List of frustration signals
        behavior_class: Classified user behavior
        attention_score: Estimated attention level
        task_completion: Estimated task completion status
    """

    session_duration: float
    total_interactions: int
    engagement_score: float
    frustration_indicators: list[AnomalyCategory]
    behavior_class: UserBehaviorClass
    attention_score: float
    task_completion: float


@dataclass
class UIUXAnomalyResult:
    """Complete UI/UX anomaly detection result.

    Attributes:
        anomaly_score: Overall anomaly score [0, 1]
        is_anomaly: Boolean anomaly flag
        anomaly_categories: Detected anomaly categories
        click_analysis: Click pattern analysis
        scroll_analysis: Scroll behavior analysis
        navigation_analysis: Navigation pattern analysis
        session_analysis: Complete session analysis
        mouse_trajectory_score: Mouse movement anomaly score
        timing_anomaly_score: Timing pattern anomaly score
        bot_probability: Probability of bot behavior
        recommendations: Suggested UX improvements
    """

    anomaly_score: float
    is_anomaly: bool
    anomaly_categories: list[AnomalyCategory]
    click_analysis: ClickAnalysis
    scroll_analysis: ScrollAnalysis
    navigation_analysis: NavigationAnalysis
    session_analysis: SessionAnalysis
    mouse_trajectory_score: float
    timing_anomaly_score: float
    bot_probability: float
    recommendations: list[str]


# =============================================================================
# Neural Network Components
# =============================================================================


class InteractionSequenceEncoder(nn.Module):
    """Encoder for sequences of user interactions.

    Uses LSTM with attention to learn representations of interaction patterns.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
    ) -> None:
        """Initialize interaction sequence encoder.

        Args:
            input_dim: Input feature dimension per interaction
            hidden_dim: LSTM hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of LSTM layers
        """
        super().__init__()

        # Embedding for interaction types
        self.type_embedding = nn.Embedding(len(InteractionType), 8)

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_dim + 8,  # Features + type embedding
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        type_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            features: Interaction features [batch, seq_len, input_dim]
            type_indices: Interaction type indices [batch, seq_len]

        Returns:
            Sequence embedding [batch, output_dim]
        """
        # Embed interaction types
        type_emb = self.type_embedding(type_indices)

        # Concatenate features and type embedding
        combined = torch.cat([features, type_emb], dim=-1)

        # LSTM encoding
        lstm_out, _ = self.lstm(combined)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Mean pooling
        pooled = attn_out.mean(dim=1)

        return self.output_proj(pooled)


class MouseTrajectoryNetwork(nn.Module):
    """Neural network for analyzing mouse trajectory patterns.

    Detects abnormal movement patterns, bot-like behavior, and
    frustration indicators from cursor trajectories.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        output_dim: int = 16,
    ) -> None:
        """Initialize mouse trajectory network.

        Args:
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super().__init__()

        # 1D convolutions for trajectory patterns
        self.conv1 = nn.Conv1d(4, hidden_dim, kernel_size=5, padding=2)  # x, y, vx, vy
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        # Global pooling and output
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2, output_dim),
            nn.ReLU(),
        )

        # Bot detection head
        self.bot_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Analyze mouse trajectory.

        Args:
            trajectory: Mouse trajectory [batch, seq_len, 4] (x, y, vx, vy)

        Returns:
            Tuple of (features, bot_probability)
        """
        # Transpose for conv1d: [batch, 4, seq_len]
        x = trajectory.transpose(1, 2)

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global pooling
        pooled = self.pool(x).squeeze(-1)

        # Output features and bot classification
        features = self.output_proj(pooled)
        bot_prob = self.bot_classifier(pooled)

        return features, bot_prob


class ClickPatternNetwork(nn.Module):
    """Network for analyzing click patterns.

    Detects rage clicks, dead clicks, and other click anomalies.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 16,
    ) -> None:
        """Initialize click pattern network.

        Args:
            input_dim: Input feature dimension per click
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super().__init__()

        # Temporal convolution for click sequences
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Attention for important click patterns
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Anomaly classifiers
        self.rage_detector = nn.Linear(hidden_dim, 1)
        self.dead_detector = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        click_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Analyze click patterns.

        Args:
            click_features: Click features [batch, seq_len, input_dim]

        Returns:
            Tuple of (features, rage_score, dead_score)
        """
        # Transpose for conv1d
        x = click_features.transpose(1, 2)

        # Temporal convolution
        conv_out = self.temporal_conv(x)

        # Transpose back
        conv_out = conv_out.transpose(1, 2)

        # Attention-weighted aggregation
        attn_weights = torch.softmax(self.attention(conv_out), dim=1)
        context = (conv_out * attn_weights).sum(dim=1)

        # Output features and anomaly scores
        features = self.output_proj(context)
        rage_score = torch.sigmoid(self.rage_detector(context))
        dead_score = torch.sigmoid(self.dead_detector(context))

        return features, rage_score, dead_score


class BehaviorClassificationNetwork(nn.Module):
    """Network for classifying user behavior types.

    Combines multiple signal sources to classify overall behavior pattern.
    """

    def __init__(
        self,
        click_dim: int = 16,
        mouse_dim: int = 16,
        scroll_dim: int = 8,
        nav_dim: int = 8,
        num_classes: int = len(UserBehaviorClass),
    ) -> None:
        """Initialize behavior classification network.

        Args:
            click_dim: Click feature dimension
            mouse_dim: Mouse trajectory feature dimension
            scroll_dim: Scroll feature dimension
            nav_dim: Navigation feature dimension
            num_classes: Number of behavior classes
        """
        super().__init__()

        total_dim = click_dim + mouse_dim + scroll_dim + nav_dim

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        click_features: torch.Tensor,
        mouse_features: torch.Tensor,
        scroll_features: torch.Tensor,
        nav_features: torch.Tensor,
    ) -> torch.Tensor:
        """Classify user behavior.

        Args:
            click_features: Click pattern features
            mouse_features: Mouse trajectory features
            scroll_features: Scroll behavior features
            nav_features: Navigation pattern features

        Returns:
            Class logits [batch, num_classes]
        """
        combined = torch.cat(
            [
                click_features,
                mouse_features,
                scroll_features,
                nav_features,
            ],
            dim=-1,
        )

        return self.classifier(combined)


# =============================================================================
# Main UI/UX Anomaly Detector
# =============================================================================


class UIUXAnomalyDetector(BaseDetector):
    """Comprehensive UI/UX anomaly detector.

    Analyzes user interaction patterns to detect:
    - Usability issues (dead clicks, confusing navigation)
    - User frustration (rage clicks, rapid scrolling)
    - Abnormal behavior (bot detection, unusual patterns)
    - Engagement problems (attention loss, abandonment)

    Example:
        >>> detector = UIUXAnomalyDetector(config={
        ...     "rage_click_threshold": 0.5,
        ...     "threshold": 0.6,
        ... })
        >>> detector.fit(normal_session_data)
        >>> result = detector.detect(test_session_interactions)
        >>> print(result["behavior_class"], result["anomaly_score"])
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize UI/UX anomaly detector.

        Args:
            config: Configuration dictionary. See UIUXConfig.
        """
        super().__init__(config)

        # Parse configuration
        self._uiux_config = UIUXConfig(
            rage_click_threshold=self.config.get("rage_click_threshold", 0.5),
            rage_click_count=self.config.get("rage_click_count", 3),
            dead_click_timeout=self.config.get("dead_click_timeout", 3.0),
            scroll_velocity_threshold=self.config.get("scroll_velocity_threshold", 3000.0),
            navigation_loop_threshold=self.config.get("navigation_loop_threshold", 3),
            session_timeout=self.config.get("session_timeout", 1800.0),
            mouse_velocity_threshold=self.config.get("mouse_velocity_threshold", 5000.0),
            timing_z_score_threshold=self.config.get("timing_z_score_threshold", 3.0),
            bot_detection_threshold=self.config.get("bot_detection_threshold", 0.7),
            min_session_length=self.config.get("min_session_length", 5),
            fitts_law_a=self.config.get("fitts_law_a", 0.1),
            fitts_law_b=self.config.get("fitts_law_b", 0.15),
            threshold=self.threshold,
        )

        # Initialize neural components
        self.device = torch.device(self.config.get("device", "cpu"))
        self._init_networks()

        # Reference statistics
        self._reference_timing_mean: float = 0.0
        self._reference_timing_std: float = 1.0
        self._reference_click_density: np.ndarray | None = None
        self._reference_scroll_velocity: float = 500.0
        self._reference_session_duration: float = 300.0
        self._page_transition_probs: dict[str, dict[str, float]] = {}
        self._reference_features_mean: np.ndarray | None = None
        self._reference_features_std: np.ndarray | None = None

    def _init_networks(self) -> None:
        """Initialize neural network components."""
        self._sequence_encoder = InteractionSequenceEncoder(
            input_dim=16,
            hidden_dim=64,
            output_dim=32,
        ).to(self.device)

        self._mouse_network = MouseTrajectoryNetwork(
            hidden_dim=32,
            output_dim=16,
        ).to(self.device)

        self._click_network = ClickPatternNetwork(
            input_dim=8,
            hidden_dim=32,
            output_dim=16,
        ).to(self.device)

        self._behavior_network = BehaviorClassificationNetwork(
            click_dim=16,
            mouse_dim=16,
            scroll_dim=8,
            nav_dim=8,
        ).to(self.device)

        # Set to eval mode
        self._sequence_encoder.eval()
        self._mouse_network.eval()
        self._click_network.eval()
        self._behavior_network.eval()

    def fit(
        self,
        interactions: list[UserInteraction] | list[list[UserInteraction]],  # type: ignore[override]
    ) -> UIUXAnomalyDetector:
        """Fit detector on reference/training data.

        Args:
            interactions: Single session or list of sessions of interactions

        Returns:
            Self for method chaining.

        Raises:
            DetectorException: If data is empty or invalid.
        """
        # Handle single session vs multiple sessions
        if len(interactions) == 0:
            raise DetectorException("Cannot fit UIUXAnomalyDetector with empty data.")

        sessions: list[list[UserInteraction]]
        if isinstance(interactions[0], UserInteraction):
            single_session: list[UserInteraction] = [
                item for item in interactions if isinstance(item, UserInteraction)
            ]
            sessions = [single_session]
        else:
            sessions = [list(s) for s in interactions if isinstance(s, list)]

        # Collect statistics from all sessions
        all_timings = []
        all_click_positions = []
        all_scroll_velocities = []
        all_session_durations = []
        page_transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for session in sessions:
            if len(session) < self._uiux_config.min_session_length:
                continue

            # Extract timing intervals
            for i in range(1, len(session)):
                timing = session[i].timestamp - session[i - 1].timestamp
                if 0 < timing < 60:  # Reasonable range
                    all_timings.append(timing)

            # Extract click positions
            for interaction in session:
                if interaction.interaction_type == InteractionType.CLICK:
                    if interaction.x is not None and interaction.y is not None:
                        all_click_positions.append([interaction.x, interaction.y])

            # Extract scroll velocities
            for i in range(1, len(session)):
                if session[i].interaction_type == InteractionType.SCROLL:
                    dt = session[i].timestamp - session[i - 1].timestamp
                    if dt > 0:
                        velocity = abs(session[i].scroll_delta) / dt
                        all_scroll_velocities.append(velocity)

            # Session duration
            if len(session) >= 2:
                duration = session[-1].timestamp - session[0].timestamp
                all_session_durations.append(duration)

            # Page transitions
            prev_page = None
            for interaction in session:
                if interaction.page_url:
                    if prev_page:
                        page_transitions[prev_page][interaction.page_url] += 1
                    prev_page = interaction.page_url

        # Store reference statistics
        if all_timings:
            self._reference_timing_mean = float(np.mean(all_timings))
            self._reference_timing_std = float(np.std(all_timings)) + 1e-8

        if all_click_positions:
            positions = np.array(all_click_positions)
            # Create density map (simplified: just store positions for now)
            self._reference_click_density = positions

        if all_scroll_velocities:
            self._reference_scroll_velocity = float(np.median(all_scroll_velocities))

        if all_session_durations:
            self._reference_session_duration = float(np.mean(all_session_durations))

        # Compute transition probabilities
        for from_page, to_pages in page_transitions.items():
            total = sum(to_pages.values())
            self._page_transition_probs[from_page] = {
                to_page: count / total for to_page, count in to_pages.items()
            }

        self._is_fitted = True
        logger.info(
            f"UIUXAnomalyDetector fitted on {len(sessions)} sessions. "
            f"Reference timing: {self._reference_timing_mean:.3f}s ± {self._reference_timing_std:.3f}s"
        )

        return self

    def detect(
        self,
        interactions: list[UserInteraction] | np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect UI/UX anomalies in user interactions.

        Args:
            interactions: List of user interactions for a session,
                or pre-computed feature array for ML fusion

        Returns:
            Dictionary containing detection results.

        Raises:
            DetectorException: If detector not fitted.
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        # Handle array input (for ML fusion compatibility)
        if isinstance(interactions, (np.ndarray, torch.Tensor)):
            return self._detect_from_features(interactions)

        # Full interaction analysis
        return self._detect_from_interactions(interactions)

    def _detect_from_interactions(
        self,
        interactions: list[UserInteraction],
    ) -> dict[str, Any]:
        """Perform full analysis on user interactions.

        Args:
            interactions: List of user interactions

        Returns:
            Complete detection result dictionary
        """
        if len(interactions) < self._uiux_config.min_session_length:
            # Return default for short sessions
            return self._create_default_result()

        # Analyze different aspects
        click_analysis = self._analyze_clicks(interactions)
        scroll_analysis = self._analyze_scrolls(interactions)
        navigation_analysis = self._analyze_navigation(interactions)
        session_analysis = self._analyze_session(interactions)

        mouse_score = self._compute_mouse_trajectory_score(interactions)
        timing_score = self._compute_timing_anomaly_score(interactions)
        bot_probability = self._estimate_bot_probability(interactions)

        # Collect anomaly categories
        anomaly_categories = self._collect_anomaly_categories(
            click_analysis,
            scroll_analysis,
            navigation_analysis,
            session_analysis,
            mouse_score,
            bot_probability,
        )

        # Compute combined anomaly score
        combined_score = self._compute_combined_score(
            click_analysis,
            scroll_analysis,
            navigation_analysis,
            session_analysis,
            mouse_score,
            timing_score,
            bot_probability,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            anomaly_categories,
            click_analysis,
            scroll_analysis,
            navigation_analysis,
        )

        # Auto-calibration
        effective_threshold = self.threshold
        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(np.array([combined_score]))

        is_anomaly = combined_score > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": combined_score,
            "anomaly_categories": [c.value for c in anomaly_categories],
            "click_analysis": click_analysis,
            "scroll_analysis": scroll_analysis,
            "navigation_analysis": navigation_analysis,
            "session_analysis": session_analysis,
            "mouse_trajectory_score": mouse_score,
            "timing_anomaly_score": timing_score,
            "bot_probability": bot_probability,
            "behavior_class": session_analysis.behavior_class.value,
            "engagement_score": session_analysis.engagement_score,
            "frustration_indicators": [f.value for f in session_analysis.frustration_indicators],
            "recommendations": recommendations,
            "detector_type": "uiux_anomaly",
            "threshold": effective_threshold,
        }

    def _detect_from_features(
        self,
        features: np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect from pre-computed features (for ML fusion).

        Args:
            features: Feature array

        Returns:
            Simplified detection result
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Simple z-score based anomaly detection for features
        if self._reference_features_mean is not None and self._reference_features_std is not None:
            z_scores = (features - self._reference_features_mean) / (
                self._reference_features_std + 1e-8
            )
            anomaly_score = float(np.mean(np.abs(z_scores)) / 3.0)
        else:
            anomaly_score = 0.5

        anomaly_score = np.clip(anomaly_score, 0.0, 1.0)

        return {
            "is_anomaly": anomaly_score > self.threshold,
            "anomaly_score": float(anomaly_score),
            "detector_type": "uiux_anomaly",
        }

    def extract_features(
        self,
        interactions: list[UserInteraction] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Extract features for ML fusion.

        Args:
            interactions: User interactions or pre-computed features

        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        # Handle array input
        if isinstance(interactions, (np.ndarray, torch.Tensor)):
            if isinstance(interactions, np.ndarray):
                return torch.tensor(interactions, dtype=torch.float32)
            return interactions

        # Extract features from interactions
        features = self._extract_interaction_features(interactions)
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _extract_interaction_features(
        self,
        interactions: list[UserInteraction],
    ) -> np.ndarray:
        """Extract numerical features from interactions.

        Args:
            interactions: List of user interactions

        Returns:
            Feature array
        """
        if len(interactions) == 0:
            return np.zeros(64)

        # Click features
        clicks = [i for i in interactions if i.interaction_type == InteractionType.CLICK]
        click_count = len(clicks)

        click_positions = (
            np.array([[c.x or 0, c.y or 0] for c in clicks]) if clicks else np.zeros((1, 2))
        )

        click_position_std = np.std(click_positions, axis=0) if len(clicks) > 1 else np.zeros(2)

        # Timing features
        timings = []
        for i in range(1, len(interactions)):
            dt = interactions[i].timestamp - interactions[i - 1].timestamp
            if 0 < dt < 60:
                timings.append(dt)

        timing_mean = np.mean(timings) if timings else 0
        timing_std = np.std(timings) if len(timings) > 1 else 0
        timing_min = np.min(timings) if timings else 0
        timing_max = np.max(timings) if timings else 0

        # Scroll features
        scrolls = [i for i in interactions if i.interaction_type == InteractionType.SCROLL]
        scroll_count = len(scrolls)
        scroll_total = sum(abs(s.scroll_delta) for s in scrolls)

        # Navigation features
        pages = [i.page_url for i in interactions if i.page_url]
        unique_pages = len(set(pages))
        page_revisits = len(pages) - unique_pages if pages else 0

        # Interaction type distribution
        type_counts = np.zeros(len(InteractionType))
        for interaction in interactions:
            type_counts[list(InteractionType).index(interaction.interaction_type)] += 1
        type_distribution = type_counts / (len(interactions) + 1e-8)

        # Session features
        session_duration = (
            interactions[-1].timestamp - interactions[0].timestamp if len(interactions) > 1 else 0
        )

        # Combine all features
        features = np.concatenate(
            [
                [click_count],
                click_position_std,
                [timing_mean, timing_std, timing_min, timing_max],
                [scroll_count, scroll_total],
                [unique_pages, page_revisits],
                type_distribution,
                [session_duration],
                [len(interactions)],
                np.zeros(64 - 16 - len(InteractionType) - 2),  # Padding to fixed size
            ]
        )

        return features[:64]  # Ensure fixed size

    def _analyze_clicks(self, interactions: list[UserInteraction]) -> ClickAnalysis:
        """Analyze click patterns.

        Args:
            interactions: List of user interactions

        Returns:
            ClickAnalysis results
        """
        cfg = self._uiux_config

        # Filter click events
        clicks = [
            i
            for i in interactions
            if i.interaction_type in [InteractionType.CLICK, InteractionType.DOUBLE_CLICK]
        ]

        if len(clicks) == 0:
            return ClickAnalysis(
                total_clicks=0,
                rage_clicks=0,
                dead_clicks=0,
                double_click_rate=0.0,
                click_accuracy=1.0,
                click_density_map=np.zeros((10, 10)),
                click_timing_stats={"mean": 0, "std": 0, "min": 0, "max": 0},
            )

        total_clicks = len(clicks)

        # Detect rage clicks (multiple clicks in quick succession)
        rage_clicks = 0
        i = 0
        while i < len(clicks):
            consecutive = 1
            j = i + 1
            while j < len(clicks):
                if clicks[j].timestamp - clicks[j - 1].timestamp < cfg.rage_click_threshold:
                    consecutive += 1
                    j += 1
                else:
                    break
            if consecutive >= cfg.rage_click_count:
                rage_clicks += 1
            i = j

        # Detect dead clicks (clicks with no element response)
        dead_clicks = sum(1 for c in clicks if c.element_id is None and c.element_type is None)

        # Double click rate
        double_clicks = sum(1 for c in clicks if c.interaction_type == InteractionType.DOUBLE_CLICK)
        double_click_rate = double_clicks / total_clicks if total_clicks > 0 else 0

        # Click accuracy (based on element targeting)
        targeted_clicks = sum(
            1 for c in clicks if c.element_id is not None or c.element_type is not None
        )
        click_accuracy = targeted_clicks / total_clicks if total_clicks > 0 else 1.0

        # Click density map
        positions = np.array(
            [[c.x or 0, c.y or 0] for c in clicks if c.x is not None and c.y is not None]
        )

        if len(positions) > 0:
            # Normalize to 10x10 grid
            x_bins = np.linspace(0, max(c.viewport_width for c in clicks), 11)
            y_bins = np.linspace(0, max(c.viewport_height for c in clicks), 11)
            click_density_map, _, _ = np.histogram2d(
                positions[:, 0], positions[:, 1], bins=[x_bins, y_bins]
            )
        else:
            click_density_map = np.zeros((10, 10))

        # Click timing statistics
        timings = []
        for i in range(1, len(clicks)):
            dt = clicks[i].timestamp - clicks[i - 1].timestamp
            if 0 < dt < 60:
                timings.append(dt)

        click_timing_stats = {
            "mean": float(np.mean(timings)) if timings else 0.0,
            "std": float(np.std(timings)) if len(timings) > 1 else 0.0,
            "min": float(np.min(timings)) if timings else 0.0,
            "max": float(np.max(timings)) if timings else 0.0,
        }

        return ClickAnalysis(
            total_clicks=total_clicks,
            rage_clicks=rage_clicks,
            dead_clicks=dead_clicks,
            double_click_rate=double_click_rate,
            click_accuracy=click_accuracy,
            click_density_map=click_density_map,
            click_timing_stats=click_timing_stats,
        )

    def _analyze_scrolls(self, interactions: list[UserInteraction]) -> ScrollAnalysis:
        """Analyze scroll behavior.

        Args:
            interactions: List of user interactions

        Returns:
            ScrollAnalysis results
        """
        cfg = self._uiux_config

        # Filter scroll events
        scrolls = [i for i in interactions if i.interaction_type == InteractionType.SCROLL]

        if len(scrolls) == 0:
            return ScrollAnalysis(
                total_scrolls=0,
                rapid_scrolls=0,
                scroll_reversals=0,
                average_velocity=0.0,
                scroll_depth=0.0,
                reading_patterns=0.0,
            )

        total_scrolls = len(scrolls)

        # Calculate velocities
        velocities = []
        prev_time = scrolls[0].timestamp
        for scroll in scrolls[1:]:
            dt = scroll.timestamp - prev_time
            if dt > 0:
                velocity = abs(scroll.scroll_delta) / dt
                velocities.append(velocity)
            prev_time = scroll.timestamp

        # Count rapid scrolls
        rapid_scrolls = sum(1 for v in velocities if v > cfg.scroll_velocity_threshold)

        # Count direction reversals
        scroll_reversals = 0
        prev_direction = 0
        for scroll in scrolls:
            direction = 1 if scroll.scroll_delta > 0 else -1 if scroll.scroll_delta < 0 else 0
            if direction != 0 and direction != prev_direction and prev_direction != 0:
                scroll_reversals += 1
            if direction != 0:
                prev_direction = direction

        average_velocity = float(np.mean(velocities)) if velocities else 0.0

        # Scroll depth (cumulative scroll distance)
        scroll_depth = sum(abs(s.scroll_delta) for s in scrolls)

        # Reading patterns (slow, steady scrolling)
        slow_scrolls = sum(1 for v in velocities if v < cfg.scroll_velocity_threshold / 3)
        reading_patterns = slow_scrolls / len(velocities) if velocities else 0.0

        return ScrollAnalysis(
            total_scrolls=total_scrolls,
            rapid_scrolls=rapid_scrolls,
            scroll_reversals=scroll_reversals,
            average_velocity=average_velocity,
            scroll_depth=scroll_depth,
            reading_patterns=reading_patterns,
        )

    def _analyze_navigation(self, interactions: list[UserInteraction]) -> NavigationAnalysis:
        """Analyze navigation patterns.

        Args:
            interactions: List of user interactions

        Returns:
            NavigationAnalysis results
        """
        cfg = self._uiux_config

        # Extract page views
        pages = [i.page_url for i in interactions if i.page_url]

        if len(pages) == 0:
            return NavigationAnalysis(
                pages_visited=0,
                navigation_loops=0,
                backtrack_rate=0.0,
                path_efficiency=1.0,
                abandonment_risk=0.0,
                confusion_score=0.0,
            )

        unique_pages = set(pages)
        pages_visited = len(unique_pages)

        # Detect navigation loops
        page_visit_counts: dict[str, int] = defaultdict(int)
        navigation_loops = 0
        for page in pages:
            page_visit_counts[page] += 1
            if page_visit_counts[page] == cfg.navigation_loop_threshold:
                navigation_loops += 1

        # Backtrack rate (going back to previously visited pages)
        backtracks = 0
        visited = set()
        for page in pages:
            if page in visited:
                backtracks += 1
            visited.add(page)
        backtrack_rate = backtracks / len(pages) if pages else 0.0

        # Path efficiency (unique pages / total page views)
        path_efficiency = pages_visited / len(pages) if pages else 1.0

        # Abandonment risk (based on session ending patterns)
        # Higher if session is short or ends abruptly
        session_duration = (
            interactions[-1].timestamp - interactions[0].timestamp if len(interactions) > 1 else 0
        )
        abandonment_risk = 1.0 - min(1.0, session_duration / self._reference_session_duration)

        # Confusion score (based on loops and backtracks)
        confusion_score = min(1.0, (navigation_loops * 0.3 + backtrack_rate * 0.5))

        return NavigationAnalysis(
            pages_visited=pages_visited,
            navigation_loops=navigation_loops,
            backtrack_rate=backtrack_rate,
            path_efficiency=path_efficiency,
            abandonment_risk=abandonment_risk,
            confusion_score=confusion_score,
        )

    def _analyze_session(self, interactions: list[UserInteraction]) -> SessionAnalysis:
        """Analyze complete session.

        Args:
            interactions: List of user interactions

        Returns:
            SessionAnalysis results
        """
        if len(interactions) < 2:
            return SessionAnalysis(
                session_duration=0.0,
                total_interactions=len(interactions),
                engagement_score=0.5,
                frustration_indicators=[],
                behavior_class=UserBehaviorClass.NORMAL,
                attention_score=0.5,
                task_completion=0.0,
            )

        session_duration = interactions[-1].timestamp - interactions[0].timestamp
        total_interactions = len(interactions)

        # Engagement score (interactions per minute)
        interactions_per_minute = total_interactions / (session_duration / 60 + 1e-8)
        engagement_score = min(1.0, interactions_per_minute / 10)

        # Detect frustration indicators
        frustration_indicators = []

        # Check for rage clicks
        click_analysis = self._analyze_clicks(interactions)
        if click_analysis.rage_clicks > 0:
            frustration_indicators.append(AnomalyCategory.RAGE_CLICK)

        # Check for rapid scrolling
        scroll_analysis = self._analyze_scrolls(interactions)
        if scroll_analysis.rapid_scrolls > 2:
            frustration_indicators.append(AnomalyCategory.RAPID_SCROLL)

        # Check for navigation loops
        nav_analysis = self._analyze_navigation(interactions)
        if nav_analysis.navigation_loops > 0:
            frustration_indicators.append(AnomalyCategory.NAVIGATION_LOOP)

        # Classify behavior
        behavior_class = self._classify_behavior(
            click_analysis, scroll_analysis, nav_analysis, session_duration
        )

        # Attention score (inverse of long gaps)
        gaps = []
        for i in range(1, len(interactions)):
            gap = interactions[i].timestamp - interactions[i - 1].timestamp
            if gap > 0:
                gaps.append(gap)

        if gaps:
            long_gaps = sum(1 for g in gaps if g > 30)  # > 30 seconds
            attention_score = 1.0 - (long_gaps / len(gaps))
        else:
            attention_score = 0.5

        # Task completion estimate (based on form submissions, etc.)
        completions = sum(
            1 for i in interactions if i.interaction_type == InteractionType.FORM_SUBMIT
        )
        task_completion = min(1.0, completions / max(1, nav_analysis.pages_visited * 0.3))

        return SessionAnalysis(
            session_duration=session_duration,
            total_interactions=total_interactions,
            engagement_score=engagement_score,
            frustration_indicators=frustration_indicators,
            behavior_class=behavior_class,
            attention_score=attention_score,
            task_completion=task_completion,
        )

    def _classify_behavior(
        self,
        click_analysis: ClickAnalysis,
        scroll_analysis: ScrollAnalysis,
        nav_analysis: NavigationAnalysis,
        session_duration: float,
    ) -> UserBehaviorClass:
        """Classify user behavior type.

        Args:
            click_analysis: Click pattern analysis
            scroll_analysis: Scroll behavior analysis
            nav_analysis: Navigation analysis
            session_duration: Session duration

        Returns:
            Classified UserBehaviorClass
        """
        # Check for frustration indicators
        if click_analysis.rage_clicks > 2 or scroll_analysis.rapid_scrolls > 3:
            return UserBehaviorClass.FRUSTRATED

        # Check for confusion
        if nav_analysis.confusion_score > 0.6:
            return UserBehaviorClass.CONFUSED

        # Check for exploration
        if nav_analysis.pages_visited > 5 and nav_analysis.path_efficiency < 0.5:
            return UserBehaviorClass.EXPLORING

        # Check for distraction
        if session_duration > 300 and click_analysis.total_clicks < 5:
            return UserBehaviorClass.DISTRACTED

        # Check for expert behavior (efficient navigation)
        if nav_analysis.path_efficiency > 0.8 and click_analysis.click_accuracy > 0.9:
            return UserBehaviorClass.EXPERT

        # Check for novice (slow, careful)
        if scroll_analysis.reading_patterns > 0.6:
            return UserBehaviorClass.NOVICE

        # Check for task-focused
        if nav_analysis.path_efficiency > 0.6 and click_analysis.total_clicks > 10:
            return UserBehaviorClass.TASK_FOCUSED

        return UserBehaviorClass.NORMAL

    def _compute_mouse_trajectory_score(
        self,
        interactions: list[UserInteraction],
    ) -> float:
        """Compute mouse trajectory anomaly score.

        Args:
            interactions: List of user interactions

        Returns:
            Mouse trajectory anomaly score [0, 1]
        """
        cfg = self._uiux_config

        # Extract mouse movements
        moves = [
            i
            for i in interactions
            if i.interaction_type == InteractionType.MOUSE_MOVE
            and i.x is not None
            and i.y is not None
        ]

        if len(moves) < 3:
            return 0.0

        # Compute velocities
        velocities = []
        for i in range(1, len(moves)):
            dt = moves[i].timestamp - moves[i - 1].timestamp
            if dt > 0:
                dx = (moves[i].x or 0) - (moves[i - 1].x or 0)
                dy = (moves[i].y or 0) - (moves[i - 1].y or 0)
                velocity = math.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)

        if not velocities:
            return 0.0

        # Check for erratic movement
        max_velocity = max(velocities)
        velocity_std = np.std(velocities)

        # Score based on velocity characteristics
        erratic_score = min(1.0, max_velocity / cfg.mouse_velocity_threshold)
        variance_score = min(1.0, velocity_std / (np.mean(velocities) + 1e-8))

        return float((erratic_score + variance_score) / 2)  # type: ignore[operator]

    def _compute_timing_anomaly_score(
        self,
        interactions: list[UserInteraction],
    ) -> float:
        """Compute timing pattern anomaly score.

        Args:
            interactions: List of user interactions

        Returns:
            Timing anomaly score [0, 1]
        """
        cfg = self._uiux_config

        # Extract inter-action timings
        timings = []
        for i in range(1, len(interactions)):
            dt = interactions[i].timestamp - interactions[i - 1].timestamp
            if 0 < dt < 60:
                timings.append(dt)

        if len(timings) < 3:
            return 0.0

        # Compute z-scores against reference
        z_scores = (np.array(timings) - self._reference_timing_mean) / self._reference_timing_std

        # Check for anomalous timings
        anomalous_count = np.sum(np.abs(z_scores) > cfg.timing_z_score_threshold)
        anomaly_rate = anomalous_count / len(timings)

        return float(np.clip(anomaly_rate * 2, 0.0, 1.0))

    def _estimate_bot_probability(
        self,
        interactions: list[UserInteraction],
    ) -> float:
        """Estimate probability of bot behavior.

        Args:
            interactions: List of user interactions

        Returns:
            Bot probability [0, 1]
        """
        if len(interactions) < 5:
            return 0.0

        bot_indicators = []

        # Check for perfectly regular timing (bots often have precise intervals)
        timings = []
        for i in range(1, len(interactions)):
            dt = interactions[i].timestamp - interactions[i - 1].timestamp
            if dt > 0:
                timings.append(dt)

        if timings:
            timing_cv = np.std(timings) / (np.mean(timings) + 1e-8)
            # Very low CV suggests robotic behavior
            if timing_cv < 0.1:
                bot_indicators.append(0.8)
            elif timing_cv < 0.2:
                bot_indicators.append(0.4)

        # Check for linear mouse movements (straight lines)
        moves = [
            (i.x, i.y)
            for i in interactions
            if i.interaction_type == InteractionType.MOUSE_MOVE
            and i.x is not None
            and i.y is not None
        ]

        if len(moves) >= 5:
            # Check linearity of movement segments
            linear_segments = 0
            for i in range(len(moves) - 2):
                # Vector from point i to i+1
                v1 = (moves[i + 1][0] - moves[i][0], moves[i + 1][1] - moves[i][1])
                # Vector from point i+1 to i+2
                v2 = (moves[i + 2][0] - moves[i + 1][0], moves[i + 2][1] - moves[i + 1][1])

                # Check if vectors are parallel (cross product near zero)
                cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                magnitude = (
                    math.sqrt(v1[0] ** 2 + v1[1] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2) + 1e-8
                )

                if cross / magnitude < 0.1:
                    linear_segments += 1

            linearity_ratio = linear_segments / (len(moves) - 2)
            if linearity_ratio > 0.8:
                bot_indicators.append(0.7)

        # Check for lack of scroll behavior (bots often don't scroll naturally)
        scrolls = [i for i in interactions if i.interaction_type == InteractionType.SCROLL]
        if len(scrolls) == 0 and len(interactions) > 20:
            bot_indicators.append(0.5)

        # Check for superhuman speed
        clicks = [i for i in interactions if i.interaction_type == InteractionType.CLICK]
        if len(clicks) >= 2:
            click_timings = [
                clicks[i].timestamp - clicks[i - 1].timestamp for i in range(1, len(clicks))
            ]
            min_click_time = min(click_timings) if click_timings else float("inf")
            if min_click_time < 0.05:  # 50ms is superhuman
                bot_indicators.append(0.9)

        if bot_indicators:
            return float(np.mean(bot_indicators))
        return 0.0

    def _collect_anomaly_categories(
        self,
        click_analysis: ClickAnalysis,
        scroll_analysis: ScrollAnalysis,
        navigation_analysis: NavigationAnalysis,
        session_analysis: SessionAnalysis,
        mouse_score: float,
        bot_probability: float,
    ) -> list[AnomalyCategory]:
        """Collect all detected anomaly categories.

        Args:
            click_analysis: Click analysis results
            scroll_analysis: Scroll analysis results
            navigation_analysis: Navigation analysis results
            session_analysis: Session analysis results
            mouse_score: Mouse trajectory score
            bot_probability: Bot probability

        Returns:
            List of detected anomaly categories
        """
        categories = []

        if click_analysis.rage_clicks > 0:
            categories.append(AnomalyCategory.RAGE_CLICK)

        if click_analysis.dead_clicks > click_analysis.total_clicks * 0.3:
            categories.append(AnomalyCategory.DEAD_CLICK)

        if scroll_analysis.rapid_scrolls > 2:
            categories.append(AnomalyCategory.RAPID_SCROLL)

        if navigation_analysis.navigation_loops > 0:
            categories.append(AnomalyCategory.NAVIGATION_LOOP)

        if navigation_analysis.abandonment_risk > 0.7:
            categories.append(AnomalyCategory.SESSION_ABANDONMENT)

        if mouse_score > 0.6:
            categories.append(AnomalyCategory.ERRATIC_MOVEMENT)

        if bot_probability > self._uiux_config.bot_detection_threshold:
            categories.append(AnomalyCategory.BOT_BEHAVIOR)

        if session_analysis.frustration_indicators:
            categories.append(AnomalyCategory.FRUSTRATION_SIGNAL)

        if navigation_analysis.confusion_score > 0.5:
            categories.append(AnomalyCategory.LAYOUT_CONFUSION)

        if session_analysis.attention_score < 0.3:
            categories.append(AnomalyCategory.ATTENTION_LOSS)

        return categories

    def _compute_combined_score(
        self,
        click_analysis: ClickAnalysis,
        scroll_analysis: ScrollAnalysis,
        navigation_analysis: NavigationAnalysis,
        session_analysis: SessionAnalysis,
        mouse_score: float,
        timing_score: float,
        bot_probability: float,
    ) -> float:
        """Compute combined anomaly score using golden ratio weighting.

        Args:
            click_analysis: Click analysis results
            scroll_analysis: Scroll analysis results
            navigation_analysis: Navigation analysis results
            session_analysis: Session analysis results
            mouse_score: Mouse trajectory score
            timing_score: Timing anomaly score
            bot_probability: Bot probability

        Returns:
            Combined anomaly score [0, 1]
        """
        # Component scores
        click_score = (
            click_analysis.rage_clicks * 0.4
            + (1 - click_analysis.click_accuracy) * 0.3
            + min(1.0, click_analysis.dead_clicks / max(1, click_analysis.total_clicks) * 2) * 0.3
        )

        scroll_score = (
            min(1.0, scroll_analysis.rapid_scrolls / 5) * 0.5
            + min(1.0, scroll_analysis.scroll_reversals / 10) * 0.5
        )

        nav_score = (
            navigation_analysis.confusion_score * 0.5
            + navigation_analysis.abandonment_risk * 0.3
            + (1 - navigation_analysis.path_efficiency) * 0.2
        )

        session_score = (
            (1 - session_analysis.engagement_score) * 0.3
            + (1 - session_analysis.attention_score) * 0.3
            + len(session_analysis.frustration_indicators) / 5 * 0.4
        )

        # Golden ratio weighting
        phi_sum = PHI + 1.0 + (1.0 / PHI) + 0.5 + 0.3 + 0.3 + 0.2

        combined = (
            PHI / phi_sum * click_score
            + 1.0 / phi_sum * scroll_score
            + (1.0 / PHI) / phi_sum * nav_score
            + 0.5 / phi_sum * session_score
            + 0.3 / phi_sum * mouse_score
            + 0.3 / phi_sum * timing_score
            + 0.2 / phi_sum * bot_probability
        )

        return float(np.clip(combined, 0.0, 1.0))

    def _generate_recommendations(
        self,
        categories: list[AnomalyCategory],
        click_analysis: ClickAnalysis,
        scroll_analysis: ScrollAnalysis,
        navigation_analysis: NavigationAnalysis,
    ) -> list[str]:
        """Generate UX improvement recommendations.

        Args:
            categories: Detected anomaly categories
            click_analysis: Click analysis results
            scroll_analysis: Scroll analysis results
            navigation_analysis: Navigation analysis results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if AnomalyCategory.RAGE_CLICK in categories:
            recommendations.append(
                "Users are rage-clicking, indicating frustration. "
                "Review button responsiveness and add visual feedback."
            )

        if AnomalyCategory.DEAD_CLICK in categories:
            recommendations.append(
                "Many clicks are not hitting interactive elements. "
                "Consider increasing clickable area sizes or reviewing layout."
            )

        if AnomalyCategory.RAPID_SCROLL in categories:
            recommendations.append(
                "Users are scrolling very quickly, possibly searching for content. "
                "Consider adding jump links or a table of contents."
            )

        if AnomalyCategory.NAVIGATION_LOOP in categories:
            recommendations.append(
                "Users are revisiting pages repeatedly, indicating confusion. "
                "Review information architecture and navigation labels."
            )

        if AnomalyCategory.LAYOUT_CONFUSION in categories:
            recommendations.append(
                "Navigation patterns suggest layout confusion. "
                "Consider user testing to identify unclear UI elements."
            )

        if AnomalyCategory.ATTENTION_LOSS in categories:
            recommendations.append(
                "Session shows signs of attention loss. "
                "Review content engagement and consider breaking up long content."
            )

        if click_analysis.click_accuracy < 0.7:
            recommendations.append(
                f"Click accuracy is only {click_analysis.click_accuracy:.0%}. "
                "Consider increasing target sizes per Fitts's Law."
            )

        if navigation_analysis.path_efficiency < 0.5:
            recommendations.append(
                "Navigation path efficiency is low. "
                "Review user flows and consider adding shortcuts."
            )

        if not recommendations:
            recommendations.append("No significant usability issues detected in this session.")

        return recommendations

    def _create_default_result(self) -> dict[str, Any]:
        """Create default result for short sessions.

        Returns:
            Default detection result dictionary
        """
        default_click = ClickAnalysis(
            total_clicks=0,
            rage_clicks=0,
            dead_clicks=0,
            double_click_rate=0.0,
            click_accuracy=1.0,
            click_density_map=np.zeros((10, 10)),
            click_timing_stats={"mean": 0, "std": 0, "min": 0, "max": 0},
        )
        default_scroll = ScrollAnalysis(
            total_scrolls=0,
            rapid_scrolls=0,
            scroll_reversals=0,
            average_velocity=0.0,
            scroll_depth=0.0,
            reading_patterns=0.0,
        )
        default_nav = NavigationAnalysis(
            pages_visited=0,
            navigation_loops=0,
            backtrack_rate=0.0,
            path_efficiency=1.0,
            abandonment_risk=0.0,
            confusion_score=0.0,
        )
        default_session = SessionAnalysis(
            session_duration=0.0,
            total_interactions=0,
            engagement_score=0.5,
            frustration_indicators=[],
            behavior_class=UserBehaviorClass.NORMAL,
            attention_score=0.5,
            task_completion=0.0,
        )

        return {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "anomaly_categories": [],
            "click_analysis": default_click,
            "scroll_analysis": default_scroll,
            "navigation_analysis": default_nav,
            "session_analysis": default_session,
            "mouse_trajectory_score": 0.0,
            "timing_anomaly_score": 0.0,
            "bot_probability": 0.0,
            "behavior_class": UserBehaviorClass.NORMAL.value,
            "engagement_score": 0.5,
            "frustration_indicators": [],
            "recommendations": ["Session too short for meaningful analysis."],
            "detector_type": "uiux_anomaly",
            "threshold": self.threshold,
        }


# =============================================================================
# Utility Functions
# =============================================================================


def compute_fitts_law_time(
    distance: float,
    target_width: float,
    a: float = 0.1,
    b: float = 0.15,
) -> float:
    """Compute expected movement time using Fitts's Law.

    Fitts's Law: MT = a + b * log2(D/W + 1)

    Args:
        distance: Distance to target
        target_width: Width of target
        a: Empirical constant (intercept)
        b: Empirical constant (slope)

    Returns:
        Expected movement time in seconds
    """
    if target_width <= 0:
        return float("inf")
    id_ = math.log2(distance / target_width + 1)  # Index of difficulty
    return a + b * id_


def detect_rage_clicks(
    clicks: list[UserInteraction],
    time_threshold: float = 0.5,
    count_threshold: int = 3,
) -> list[tuple[int, int]]:
    """Detect rage click sequences.

    Args:
        clicks: List of click interactions
        time_threshold: Maximum time between clicks for rage detection
        count_threshold: Minimum consecutive clicks for rage detection

    Returns:
        List of (start_index, end_index) tuples for rage click sequences
    """
    rage_sequences = []
    i = 0

    while i < len(clicks):
        sequence_start = i
        j = i + 1

        while j < len(clicks):
            if clicks[j].timestamp - clicks[j - 1].timestamp < time_threshold:
                j += 1
            else:
                break

        sequence_length = j - sequence_start
        if sequence_length >= count_threshold:
            rage_sequences.append((sequence_start, j - 1))

        i = j

    return rage_sequences


def compute_click_heatmap(
    clicks: list[UserInteraction],
    width: int = 1920,
    height: int = 1080,
    grid_size: int = 20,
) -> np.ndarray:
    """Compute click density heatmap.

    Args:
        clicks: List of click interactions
        width: Viewport width
        height: Viewport height
        grid_size: Number of grid cells in each dimension

    Returns:
        2D heatmap array [grid_size, grid_size]
    """
    heatmap = np.zeros((grid_size, grid_size))

    for click in clicks:
        if click.x is not None and click.y is not None:
            x_idx = int(click.x / width * (grid_size - 1))
            y_idx = int(click.y / height * (grid_size - 1))
            x_idx = max(0, min(grid_size - 1, x_idx))
            y_idx = max(0, min(grid_size - 1, y_idx))
            heatmap[y_idx, x_idx] += 1

    return heatmap


def analyze_navigation_flow(
    interactions: list[UserInteraction],
) -> dict[str, Any]:
    """Analyze navigation flow patterns.

    Args:
        interactions: List of user interactions

    Returns:
        Dictionary with navigation flow analysis
    """
    pages = [i.page_url for i in interactions if i.page_url]

    if len(pages) < 2:
        return {
            "flow": [],
            "unique_pages": set(),
            "transition_matrix": {},
        }

    # Build transition matrix
    transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(len(pages) - 1):
        transitions[pages[i]][pages[i + 1]] += 1

    # Identify flow
    flow = []
    visited = set()
    for page in pages:
        if page not in visited:
            flow.append(page)
            visited.add(page)

    return {
        "flow": flow,
        "unique_pages": visited,
        "transition_matrix": dict(transitions),
    }
