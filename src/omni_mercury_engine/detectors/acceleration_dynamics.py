"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Physics Acceleration Dynamics Module for Mercury Agent.

Advanced anomaly detection using physics-based motion analysis:
- Velocity tracking: v = Δx/Δt
- Acceleration detection: a = (v_f - v_i)/t
- Jerk analysis: j = da/dt (rate of acceleration change)
- Momentum-based scoring: p = mv
- Energy analysis: KE = ½mv², PE
- Phase space trajectory analysis
- Lyapunov exponent estimation for chaos detection

This module applies kinematic formulas to time-series data for detecting:
- Sudden velocity changes (anomalous rate-of-change)
- Abnormal acceleration patterns
- Energy conservation violations
- Chaotic vs stable system behavior
- Momentum transfer anomalies

The physics metaphor maps naturally to many domains:
- Network traffic: "velocity" = data rate, "acceleration" = rate change
- Financial: "velocity" = price change rate, "momentum" = trend strength
- System metrics: "velocity" = metric change, "acceleration" = trend acceleration
- User behavior: "velocity" = action rate, "momentum" = engagement intensity

Research foundations:
- Classical mechanics (Newton's laws)
- Dynamical systems theory
- Chaos theory and Lyapunov stability
- Phase space analysis
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
else:
    try:
        import torch
        from torch import nn

        TORCH_AVAILABLE = True
    except ImportError:
        torch = None  # type: ignore[assignment, unused-ignore]
        nn = None  # type: ignore[assignment, unused-ignore]
        TORCH_AVAILABLE = False

from scipy.ndimage import uniform_filter1d

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.utils.constants import MathematicalConstants

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Golden ratio for harmonic weighting (from 3R mechanism)
PHI = MathematicalConstants.GOLDEN_RATIO.value

# Lyapunov stability convergence rate (from fusion.py)
LYAPUNOV_CONVERGENCE_RATE = 0.25


class MotionState(Enum):
    """Classification of motion states."""

    STATIONARY = "stationary"
    UNIFORM_MOTION = "uniform_motion"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    OSCILLATING = "oscillating"
    CHAOTIC = "chaotic"
    ANOMALOUS = "anomalous"


class EnergyState(Enum):
    """Energy conservation states."""

    CONSERVED = "conserved"
    GAINING = "gaining"
    LOSING = "losing"
    UNSTABLE = "unstable"


# =============================================================================
# Configuration and Result Dataclasses
# =============================================================================


@dataclass
class AccelerationDynamicsConfig:
    """
    Configuration for acceleration dynamics analysis.

    Attributes:
        time_step: Time step between samples (default 1.0)
        velocity_window: Window size for velocity calculation
        acceleration_window: Window size for acceleration smoothing
        jerk_sensitivity: Sensitivity threshold for jerk detection
        momentum_mass: Effective "mass" for momentum calculations
        energy_conservation_tolerance: Tolerance for energy conservation checks
        lyapunov_embedding_dim: Embedding dimension for Lyapunov estimation
        lyapunov_delay: Time delay for phase space reconstruction
        phase_space_neighbors: k-nearest neighbors for phase space analysis
        chaos_threshold: Threshold for chaos detection
        threshold: Anomaly detection threshold
    """

    time_step: float = 1.0
    velocity_window: int = 3
    acceleration_window: int = 5
    jerk_sensitivity: float = 2.0
    momentum_mass: float = 1.0
    energy_conservation_tolerance: float = 0.1
    lyapunov_embedding_dim: int = 3
    lyapunov_delay: int = 1
    phase_space_neighbors: int = 5
    chaos_threshold: float = 0.1
    threshold: float = 0.5
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class KinematicFeatures:
    """
    Extracted kinematic features from time-series.

    Attributes:
        position: Original signal (position analog)
        velocity: First derivative (rate of change)
        acceleration: Second derivative
        jerk: Third derivative (rate of acceleration change)
        kinetic_energy: ½mv² analog
        potential_energy: Estimated potential energy
        total_energy: Kinetic + Potential
        momentum: mv analog
        impulse: Change in momentum
        mean_velocity: Average velocity
        mean_acceleration: Average acceleration
        max_jerk: Maximum jerk magnitude
        motion_state: Classified motion state
        energy_state: Energy conservation state
    """

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    momentum: np.ndarray
    impulse: np.ndarray
    mean_velocity: float
    mean_acceleration: float
    max_jerk: float
    motion_state: MotionState
    energy_state: EnergyState


@dataclass
class PhaseSpaceFeatures:
    """
    Phase space analysis features.

    Attributes:
        trajectory: Phase space trajectory [time, embedding_dim]
        lyapunov_exponent: Largest Lyapunov exponent (chaos indicator)
        correlation_dimension: Fractal dimension estimate
        recurrence_rate: Rate of trajectory recurrence
        determinism: Determinism measure from recurrence
        entropy: Phase space entropy
        is_chaotic: Boolean chaos detection
        attractor_type: Type of attractor detected
    """

    trajectory: np.ndarray
    lyapunov_exponent: float
    correlation_dimension: float
    recurrence_rate: float
    determinism: float
    entropy: float
    is_chaotic: bool
    attractor_type: str


@dataclass
class AccelerationAnomalyResult:
    """
    Complete anomaly detection result.

    Attributes:
        anomaly_score: Overall anomaly score [0, 1]
        is_anomaly: Boolean anomaly flag
        kinematic_features: Extracted kinematic features
        phase_space_features: Phase space analysis results
        velocity_anomaly_score: Velocity-based anomaly component
        acceleration_anomaly_score: Acceleration-based component
        jerk_anomaly_score: Jerk-based component
        energy_anomaly_score: Energy conservation anomaly
        chaos_anomaly_score: Chaos/instability anomaly
        anomaly_timestamps: Indices of detected anomalies
        anomaly_descriptions: Human-readable anomaly descriptions
    """

    anomaly_score: float
    is_anomaly: bool
    kinematic_features: KinematicFeatures
    phase_space_features: PhaseSpaceFeatures
    velocity_anomaly_score: float
    acceleration_anomaly_score: float
    jerk_anomaly_score: float
    energy_anomaly_score: float
    chaos_anomaly_score: float
    anomaly_timestamps: list[int]
    anomaly_descriptions: list[str]


# =============================================================================
# Neural Network Components
# =============================================================================

if TYPE_CHECKING or TORCH_AVAILABLE:

    class MotionEncoder(nn.Module):
        """
        Neural network encoder for motion feature extraction.

        Learns representations from kinematic features that capture normal motion patterns and
        detect deviations.
        """

        def __init__(
            self,
            input_dim: int = 4,  # position, velocity, acceleration, jerk
            hidden_dim: int = 64,
            output_dim: int = 32,
            num_layers: int = 2,
        ) -> None:
            """
            Initialize motion encoder.

            Args:
                input_dim: Number of kinematic input features
                hidden_dim: Hidden layer dimension
                output_dim: Output embedding dimension
                num_layers: Number of LSTM layers
            """
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim

            # Bidirectional LSTM for temporal patterns
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1 if num_layers > 1 else 0.0,
            )

            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through motion encoder.

            Args:
                x: Kinematic features [batch, time, input_dim]

            Returns:
                Motion embedding [batch, output_dim]
            """
            # LSTM encoding
            lstm_out, _ = self.lstm(x)  # [batch, time, hidden*2]

            # Attention-weighted aggregation
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)

            # Project to output
            return self.output_proj(context)

    class PhaseSpaceNetwork(nn.Module):
        """
        Neural network for phase space trajectory analysis.

        Processes phase space embeddings to detect chaotic behavior and trajectory anomalies.
        """

        def __init__(
            self,
            embedding_dim: int = 3,
            hidden_dim: int = 32,
            output_dim: int = 16,
        ) -> None:
            """
            Initialize phase space network.

            Args:
                embedding_dim: Phase space embedding dimension
                hidden_dim: Hidden layer dimension
                output_dim: Output feature dimension
            """
            super().__init__()

            # Trajectory encoder
            self.trajectory_encoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Temporal convolution for local patterns
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(),
            )

            # Global pooling and output
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.output_proj = nn.Linear(hidden_dim // 2, output_dim)

            # Chaos classifier
            self.chaos_classifier = nn.Sequential(
                nn.Linear(hidden_dim // 2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            trajectory: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass through phase space network.

            Args:
                trajectory: Phase space trajectory [batch, time, embedding_dim]

            Returns:
                Tuple of (features, chaos_score)
            """
            # Encode trajectory points
            encoded = self.trajectory_encoder(trajectory)  # [batch, time, hidden]

            # Temporal convolution
            encoded_t = encoded.transpose(1, 2)  # [batch, hidden, time]
            conv_out = self.temporal_conv(encoded_t)  # [batch, hidden/2, time]

            # Global pooling
            pooled = self.global_pool(conv_out).squeeze(-1)  # [batch, hidden/2]

            # Output features and chaos score
            features = self.output_proj(pooled)
            chaos_score = self.chaos_classifier(pooled)

            return features, chaos_score

    class EnergyConservationNetwork(nn.Module):
        """
        Network for detecting energy conservation violations.

        Learns to predict energy at each timestep and flags deviations that indicate anomalous
        energy injection or dissipation.
        """

        def __init__(
            self,
            input_dim: int = 3,  # KE, PE, momentum
            hidden_dim: int = 32,
        ) -> None:
            """
            Initialize energy conservation network.

            Args:
                input_dim: Number of energy-related inputs
                hidden_dim: Hidden layer dimension
            """
            super().__init__()

            # Energy predictor (predicts next energy from current state)
            self.energy_predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

            # Violation detector
            self.violation_detector = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),  # +1 for prediction error
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            energy_features: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Detect energy conservation violations.

            Args:
                energy_features: Energy features [batch, time, input_dim]

            Returns:
                Tuple of (predicted_energy, violation_score)
            """
            batch_size, time_steps, _ = energy_features.shape

            # Predict energy at each step
            predicted = self.energy_predictor(energy_features)  # [batch, time, 1]

            # Compute prediction error (actual total energy vs predicted)
            # For simplicity, use first feature as total energy proxy
            actual_energy = energy_features[:, :, 0:1]
            prediction_error = (actual_energy - predicted).abs()

            # Detect violations
            violation_input = torch.cat([energy_features, prediction_error], dim=-1)
            violation_scores = self.violation_detector(violation_input)

            return predicted.squeeze(-1), violation_scores.squeeze(-1)

else:

    class MotionEncoder:
        """Stub: MotionEncoder requires PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("MotionEncoder requires PyTorch. Install with: pip install torch")

    class PhaseSpaceNetwork:
        """Stub: PhaseSpaceNetwork requires PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PhaseSpaceNetwork requires PyTorch. Install with: pip install torch")

    class EnergyConservationNetwork:
        """Stub: EnergyConservationNetwork requires PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "EnergyConservationNetwork requires PyTorch. Install with: pip install torch"
            )


# =============================================================================
# Main Acceleration Dynamics Detector
# =============================================================================


class AccelerationDynamicsDetector(BaseDetector):
    """Physics-based acceleration dynamics anomaly detector.

    Uses kinematic analysis to detect anomalies in time-series data:
    - Tracks position, velocity, acceleration, jerk
    - Monitors energy conservation
    - Analyzes phase space trajectories
    - Estimates Lyapunov exponents for chaos detection

    The physics metaphor provides interpretable anomaly detection:
    - "This point shows anomalous acceleration" = sudden rate change
    - "Energy is not conserved" = unexpected system behavior
    - "Chaotic trajectory detected" = unpredictable dynamics

    Example:
        >>> detector = AccelerationDynamicsDetector(config={
        ...     "time_step": 0.1,
        ...     "threshold": 0.6,
        ... })
        >>> detector.fit(normal_time_series)
        >>> result = detector.detect(test_time_series)
        >>> print(result["motion_state"], result["anomaly_score"])
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize acceleration dynamics detector.

        Args:
            config: Configuration dictionary. See AccelerationDynamicsConfig.
                ``config["seed"]`` (optional int) seeds the per-instance
                ``Generator`` used for phase-space neighbour subsampling.
        """
        super().__init__(config)
        self._rng: np.random.Generator = np.random.default_rng(self.config.get("seed"))

        # Parse configuration
        self._dynamics_config = AccelerationDynamicsConfig(
            time_step=self.config.get("time_step", 1.0),
            velocity_window=self.config.get("velocity_window", 3),
            acceleration_window=self.config.get("acceleration_window", 5),
            jerk_sensitivity=self.config.get("jerk_sensitivity", 2.0),
            momentum_mass=self.config.get("momentum_mass", 1.0),
            energy_conservation_tolerance=self.config.get("energy_conservation_tolerance", 0.1),
            lyapunov_embedding_dim=self.config.get("lyapunov_embedding_dim", 3),
            lyapunov_delay=self.config.get("lyapunov_delay", 1),
            phase_space_neighbors=self.config.get("phase_space_neighbors", 5),
            chaos_threshold=self.config.get("chaos_threshold", 0.1),
            threshold=self.threshold,
        )

        # Initialize neural components
        self.device = torch.device(self.config.get("device", "cpu"))
        self._init_networks()

        # Reference statistics
        self._reference_velocity_mean: float = 0.0
        self._reference_velocity_std: float = 1.0
        self._reference_acceleration_mean: float = 0.0
        self._reference_acceleration_std: float = 1.0
        self._reference_jerk_mean: float = 0.0
        self._reference_jerk_std: float = 1.0
        self._reference_energy_mean: float = 0.0
        self._reference_energy_std: float = 1.0
        self._reference_lyapunov: float = 0.0

    def _init_networks(self) -> None:
        """Initialize neural network components."""
        cfg = self._dynamics_config

        self._motion_encoder = MotionEncoder(
            input_dim=4,
            hidden_dim=64,
            output_dim=32,
        ).to(self.device)

        self._phase_network = PhaseSpaceNetwork(
            embedding_dim=cfg.lyapunov_embedding_dim,
            hidden_dim=32,
            output_dim=16,
        ).to(self.device)

        self._energy_network = EnergyConservationNetwork(
            input_dim=3,
            hidden_dim=32,
        ).to(self.device)

        # Set to eval mode
        self._motion_encoder.eval()
        self._phase_network.eval()
        self._energy_network.eval()

    def fit(self, data: np.ndarray | torch.Tensor) -> AccelerationDynamicsDetector:
        """
        Fit detector on reference/training data.

        Args:
            data: Time-series data [num_samples] or [batch, num_samples]

        Returns:
            Self for method chaining.

        Raises:
            DetectorException: If data is empty or invalid.
        """
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.size == 0:
            raise DetectorException("Cannot fit AccelerationDynamicsDetector with empty data.")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Compute kinematic features for all samples
        all_velocities = []
        all_accelerations = []
        all_jerks = []
        all_energies = []
        all_lyapunov = []

        for sample in data:
            features = self._compute_kinematic_features(sample)
            all_velocities.extend(features.velocity.tolist())
            all_accelerations.extend(features.acceleration.tolist())
            all_jerks.extend(features.jerk.tolist())
            all_energies.extend(features.total_energy.tolist())

            # Compute Lyapunov exponent
            phase_features = self._analyze_phase_space(sample)
            all_lyapunov.append(phase_features.lyapunov_exponent)

        # Store reference statistics
        self._reference_velocity_mean = float(np.mean(all_velocities))
        self._reference_velocity_std = float(np.std(all_velocities)) + 1e-8
        self._reference_acceleration_mean = float(np.mean(all_accelerations))
        self._reference_acceleration_std = float(np.std(all_accelerations)) + 1e-8
        self._reference_jerk_mean = float(np.mean(all_jerks))
        self._reference_jerk_std = float(np.std(all_jerks)) + 1e-8
        self._reference_energy_mean = float(np.mean(all_energies))
        self._reference_energy_std = float(np.std(all_energies)) + 1e-8
        self._reference_lyapunov = float(np.mean(all_lyapunov))

        self._is_fitted = True
        logger.info(
            f"AccelerationDynamicsDetector fitted. "
            f"Reference velocity: {self._reference_velocity_mean:.4f} ± {self._reference_velocity_std:.4f}"
        )

        return self

    def detect(self, data: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """
        Detect anomalies using acceleration dynamics analysis.

        Args:
            data: Time-series data [num_samples] or [batch, num_samples]

        Returns:
            Dictionary containing detection results and analysis.

        Raises:
            DetectorException: If detector not fitted.
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(1, -1)

        all_scores = []
        all_results = []

        for sample in data:
            result = self._analyze_sample(sample)
            all_scores.append(result.anomaly_score)
            all_results.append(result)

        # Aggregate
        mean_score = float(np.mean(all_scores))
        primary_result = (
            all_results[0]
            if len(all_results) == 1
            else max(all_results, key=lambda r: r.anomaly_score)
        )

        # Auto-calibration
        effective_threshold = self.threshold
        calibration_diagnostics = None
        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(np.array(all_scores))
            calibration_diagnostics = self._last_diagnostics

        is_anomaly = mean_score > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": mean_score,
            "scores": np.array(all_scores),
            "motion_state": primary_result.kinematic_features.motion_state.value,
            "energy_state": primary_result.kinematic_features.energy_state.value,
            "velocity_anomaly": primary_result.velocity_anomaly_score,
            "acceleration_anomaly": primary_result.acceleration_anomaly_score,
            "jerk_anomaly": primary_result.jerk_anomaly_score,
            "energy_anomaly": primary_result.energy_anomaly_score,
            "chaos_score": primary_result.chaos_anomaly_score,
            "lyapunov_exponent": primary_result.phase_space_features.lyapunov_exponent,
            "is_chaotic": primary_result.phase_space_features.is_chaotic,
            "attractor_type": primary_result.phase_space_features.attractor_type,
            "anomaly_timestamps": primary_result.anomaly_timestamps,
            "descriptions": primary_result.anomaly_descriptions,
            "mean_velocity": primary_result.kinematic_features.mean_velocity,
            "mean_acceleration": primary_result.kinematic_features.mean_acceleration,
            "max_jerk": primary_result.kinematic_features.max_jerk,
            "detector_type": "acceleration_dynamics",
            "threshold": effective_threshold,
            "calibration_diagnostics": calibration_diagnostics,
        }

    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Extract features for ML fusion.

        Args:
            data: Time-series data

        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(1, -1)

        all_features = []

        for sample in data:
            # Compute kinematic features
            kin_features = self._compute_kinematic_features(sample)

            # Compute phase space features
            phase_features = self._analyze_phase_space(sample)

            # Extract neural features
            with torch.no_grad():
                # Prepare kinematic sequence
                kin_seq = np.stack(
                    [
                        kin_features.position,
                        kin_features.velocity,
                        kin_features.acceleration,
                        kin_features.jerk,
                    ],
                    axis=-1,
                )
                kin_tensor = torch.tensor(
                    kin_seq, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                motion_features = self._motion_encoder(kin_tensor).cpu().numpy().flatten()

                # Prepare phase space sequence
                phase_tensor = torch.tensor(
                    phase_features.trajectory,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                phase_nn_features, _ = self._phase_network(phase_tensor)
                phase_nn_features = phase_nn_features.cpu().numpy().flatten()

            # Combine all features
            combined = np.concatenate(
                [
                    # Kinematic statistics
                    [kin_features.mean_velocity],
                    [kin_features.mean_acceleration],
                    [kin_features.max_jerk],
                    [np.std(kin_features.velocity)],
                    [np.std(kin_features.acceleration)],
                    # Energy features
                    [np.mean(kin_features.kinetic_energy)],
                    [np.std(kin_features.total_energy)],
                    # Phase space features
                    [phase_features.lyapunov_exponent],
                    [phase_features.correlation_dimension],
                    [phase_features.entropy],
                    [phase_features.recurrence_rate],
                    [phase_features.determinism],
                    # Neural features
                    motion_features,
                    phase_nn_features,
                ]
            )

            all_features.append(combined)

        return torch.tensor(np.array(all_features), dtype=torch.float32)

    def _analyze_sample(self, signal: np.ndarray) -> AccelerationAnomalyResult:
        """
        Perform complete analysis on a single sample.

        Args:
            signal: Time-series signal

        Returns:
            AccelerationAnomalyResult with full analysis
        """
        # Compute kinematic features
        kin_features = self._compute_kinematic_features(signal)

        # Analyze phase space
        phase_features = self._analyze_phase_space(signal)

        # Compute anomaly scores
        velocity_score = self._compute_velocity_anomaly(kin_features)
        accel_score = self._compute_acceleration_anomaly(kin_features)
        jerk_score = self._compute_jerk_anomaly(kin_features)
        energy_score = self._compute_energy_anomaly(kin_features)
        chaos_score = self._compute_chaos_anomaly(phase_features)

        # Find anomaly timestamps
        timestamps, descriptions = self._find_anomaly_timestamps(kin_features, phase_features)

        # Combine scores with golden ratio weighting
        phi_sum = PHI + 1.0 + (1.0 / PHI) + 0.5 + 0.3
        weights = {
            "velocity": PHI / phi_sum,
            "acceleration": 1.0 / phi_sum,
            "jerk": (1.0 / PHI) / phi_sum,
            "energy": 0.5 / phi_sum,
            "chaos": 0.3 / phi_sum,
        }

        combined_score = (
            weights["velocity"] * velocity_score
            + weights["acceleration"] * accel_score
            + weights["jerk"] * jerk_score
            + weights["energy"] * energy_score
            + weights["chaos"] * chaos_score
        )
        combined_score = float(np.clip(combined_score, 0.0, 1.0))

        return AccelerationAnomalyResult(
            anomaly_score=combined_score,
            is_anomaly=combined_score > self.threshold,
            kinematic_features=kin_features,
            phase_space_features=phase_features,
            velocity_anomaly_score=velocity_score,
            acceleration_anomaly_score=accel_score,
            jerk_anomaly_score=jerk_score,
            energy_anomaly_score=energy_score,
            chaos_anomaly_score=chaos_score,
            anomaly_timestamps=timestamps,
            anomaly_descriptions=descriptions,
        )

    def _compute_kinematic_features(self, signal: np.ndarray) -> KinematicFeatures:
        """Compute all kinematic features from signal.

        Physics formulas applied:
        - Velocity: v = dx/dt (finite difference)
        - Acceleration: a = dv/dt = d²x/dt²
        - Jerk: j = da/dt = d³x/dt³
        - Kinetic Energy: KE = ½mv²
        - Momentum: p = mv
        - Impulse: J = Δp

        Args:
            signal: Time-series signal (position analog)

        Returns:
            KinematicFeatures with all computed values
        """
        cfg = self._dynamics_config
        dt = cfg.time_step
        m = cfg.momentum_mass

        # Position is the signal itself
        position = signal.copy()

        # Velocity: v = dx/dt (central difference)
        velocity = np.gradient(position, dt)

        # Smooth velocity
        if cfg.velocity_window > 1:
            velocity = uniform_filter1d(velocity, size=cfg.velocity_window, mode="nearest")

        # Acceleration: a = dv/dt
        acceleration = np.gradient(velocity, dt)

        # Smooth acceleration
        if cfg.acceleration_window > 1:
            acceleration = uniform_filter1d(
                acceleration, size=cfg.acceleration_window, mode="nearest"
            )

        # Jerk: j = da/dt
        jerk = np.gradient(acceleration, dt)

        # Kinetic Energy: KE = ½mv²
        kinetic_energy = 0.5 * m * velocity**2

        # Potential Energy: Estimate from position (assume spring-like: PE = ½kx²)
        # Use reference mean as equilibrium
        equilibrium = np.mean(position)
        potential_energy = 0.5 * (position - equilibrium) ** 2

        # Total Energy
        total_energy = kinetic_energy + potential_energy

        # Momentum: p = mv
        momentum = m * velocity

        # Impulse: J = Δp (change in momentum)
        impulse = np.gradient(momentum, dt) * dt

        # Statistics
        mean_velocity = float(np.mean(velocity))
        mean_acceleration = float(np.mean(np.abs(acceleration)))
        max_jerk = float(np.max(np.abs(jerk)))

        # Classify motion state
        motion_state = self._classify_motion_state(velocity, acceleration, jerk)

        # Classify energy state
        energy_state = self._classify_energy_state(total_energy)

        return KinematicFeatures(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            total_energy=total_energy,
            momentum=momentum,
            impulse=impulse,
            mean_velocity=mean_velocity,
            mean_acceleration=mean_acceleration,
            max_jerk=max_jerk,
            motion_state=motion_state,
            energy_state=energy_state,
        )

    def _classify_motion_state(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        jerk: np.ndarray,
    ) -> MotionState:
        """
        Classify the motion state from kinematic features.

        Args:
            velocity: Velocity array
            acceleration: Acceleration array
            jerk: Jerk array

        Returns:
            Classified MotionState
        """
        v_std = np.std(velocity)
        v_mean = np.abs(np.mean(velocity))
        a_mean = np.mean(np.abs(acceleration))
        j_max = np.max(np.abs(jerk))

        # Thresholds (relative to reference if available)
        v_threshold = max(self._reference_velocity_std * 0.1, 0.01)
        a_threshold = max(self._reference_acceleration_std * 0.5, 0.01)
        j_threshold = max(self._reference_jerk_std * 2.0, 0.01)

        # Check for stationary
        if v_std < v_threshold and v_mean < v_threshold:
            return MotionState.STATIONARY

        # Check for uniform motion
        if a_mean < a_threshold:
            return MotionState.UNIFORM_MOTION

        # Check for oscillation (sign changes in velocity)
        sign_changes = np.sum(np.diff(np.sign(velocity)) != 0)
        if sign_changes > len(velocity) * 0.3:
            return MotionState.OSCILLATING

        # Check acceleration direction
        positive_accel = np.mean(acceleration) > a_threshold
        negative_accel = np.mean(acceleration) < -a_threshold

        if positive_accel:
            return MotionState.ACCELERATING
        elif negative_accel:
            return MotionState.DECELERATING

        # Check for chaotic behavior (high jerk variance)
        if j_max > j_threshold * 3:
            return MotionState.CHAOTIC

        return MotionState.ANOMALOUS

    def _classify_energy_state(self, total_energy: np.ndarray) -> EnergyState:
        """
        Classify energy conservation state.

        Args:
            total_energy: Total energy array

        Returns:
            Classified EnergyState
        """
        cfg = self._dynamics_config

        # Compute energy trend
        energy_diff = np.diff(total_energy)
        mean_diff = np.mean(energy_diff)
        std_diff = np.std(energy_diff)

        tolerance = cfg.energy_conservation_tolerance * np.mean(total_energy)

        if abs(mean_diff) < tolerance and std_diff < tolerance:
            return EnergyState.CONSERVED
        elif mean_diff > tolerance:
            return EnergyState.GAINING
        elif mean_diff < -tolerance:
            return EnergyState.LOSING
        else:
            return EnergyState.UNSTABLE

    def _analyze_phase_space(self, signal: np.ndarray) -> PhaseSpaceFeatures:
        """
        Analyze phase space trajectory.

        Constructs phase space embedding using delay coordinates and
        computes dynamical systems metrics.

        Args:
            signal: Time-series signal

        Returns:
            PhaseSpaceFeatures with analysis results
        """
        cfg = self._dynamics_config

        # Construct delay embedding
        trajectory = self._construct_delay_embedding(
            signal,
            cfg.lyapunov_embedding_dim,
            cfg.lyapunov_delay,
        )

        # Estimate largest Lyapunov exponent
        lyapunov = self._estimate_lyapunov_exponent(trajectory)

        # Estimate correlation dimension
        corr_dim = self._estimate_correlation_dimension(trajectory)

        # Compute recurrence quantification
        rr, det = self._compute_recurrence_metrics(trajectory)

        # Compute phase space entropy
        entropy = self._compute_phase_entropy(trajectory)

        # Determine if chaotic
        is_chaotic = lyapunov > cfg.chaos_threshold

        # Classify attractor type
        attractor_type = self._classify_attractor(lyapunov, corr_dim, rr)

        return PhaseSpaceFeatures(
            trajectory=trajectory,
            lyapunov_exponent=lyapunov,
            correlation_dimension=corr_dim,
            recurrence_rate=rr,
            determinism=det,
            entropy=entropy,
            is_chaotic=is_chaotic,
            attractor_type=attractor_type,
        )

    def _construct_delay_embedding(
        self,
        signal: np.ndarray,
        dim: int,
        delay: int,
    ) -> np.ndarray:
        """
        Construct delay coordinate embedding.

        Creates phase space reconstruction: [x(t), x(t-τ), x(t-2τ), ...]

        Args:
            signal: Time series
            dim: Embedding dimension
            delay: Time delay

        Returns:
            Delay embedding array [num_points, dim]
        """
        n = len(signal) - (dim - 1) * delay
        if n < 1:
            # Fallback for short signals
            return signal.reshape(-1, 1)

        embedding = np.zeros((n, dim))
        for i in range(dim):
            start = i * delay
            end = start + n
            embedding[:, i] = signal[start:end]

        return embedding

    def _estimate_lyapunov_exponent(self, trajectory: np.ndarray) -> float:
        """
        Estimate largest Lyapunov exponent.

        Uses the method of Wolf et al. (1985) for estimating the largest
        Lyapunov exponent from time series.

        Args:
            trajectory: Phase space trajectory [n, dim]

        Returns:
            Estimated Lyapunov exponent
        """
        n, dim = trajectory.shape
        if n < 10:
            return 0.0

        cfg = self._dynamics_config
        k = min(cfg.phase_space_neighbors, n - 1)

        # Find nearest neighbors
        divergences = []

        for i in range(0, n - k - 1, max(1, n // 20)):
            # Find k nearest neighbors
            distances = np.linalg.norm(trajectory[i] - trajectory, axis=1)
            distances[i] = np.inf  # Exclude self
            neighbor_idx = np.argpartition(distances, k)[:k]

            # Track divergence over time
            for j in neighbor_idx:
                if i + k < n and j + k < n:
                    initial_dist = max(distances[j], 1e-10)
                    final_dist = max(np.linalg.norm(trajectory[i + k] - trajectory[j + k]), 1e-10)

                    # Log divergence rate
                    divergence = np.log(final_dist / initial_dist) / (k * cfg.time_step)
                    divergences.append(divergence)

        if not divergences:
            return 0.0

        return float(np.mean(divergences))

    def _estimate_correlation_dimension(self, trajectory: np.ndarray) -> float:
        """
        Estimate correlation dimension using Grassberger-Procaccia algorithm.

        Args:
            trajectory: Phase space trajectory

        Returns:
            Estimated correlation dimension
        """
        n = len(trajectory)
        if n < 20:
            return 1.0

        # Compute pairwise distances (subsample for efficiency)
        if n > 50:
            indices = self._rng.choice(n, min(n, 50), replace=False)
            points = trajectory[indices]
        else:
            points = trajectory

        # Compute distances
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                if d > 0:
                    distances.append(d)

        if len(distances) < 10:
            return 1.0

        dist_arr = np.array(distances)  # type: ignore[assignment, unused-ignore]

        # Correlation sum for different radii
        radii = np.logspace(
            float(np.log10(np.min(dist_arr[dist_arr > 0]))),
            float(np.log10(np.max(dist_arr))),
            10,  # type: ignore[operator, unused-ignore]
        )
        correlations = []

        for r in radii:
            c = np.sum(dist_arr < r) / len(dist_arr)
            if c > 0:
                correlations.append((np.log(r), np.log(c)))

        if len(correlations) < 3:
            return 1.0

        # Linear regression to estimate dimension
        log_r = np.array([c[0] for c in correlations])
        log_c = np.array([c[1] for c in correlations])

        # Fit line
        slope, _ = np.polyfit(log_r, log_c, 1)

        return float(max(0.1, min(slope, 10.0)))

    def _compute_recurrence_metrics(
        self,
        trajectory: np.ndarray,
        threshold_percentile: float = 10.0,
    ) -> tuple[float, float]:
        """
        Compute recurrence quantification analysis metrics.

        Args:
            trajectory: Phase space trajectory
            threshold_percentile: Percentile for recurrence threshold

        Returns:
            Tuple of (recurrence_rate, determinism)
        """
        n = len(trajectory)
        if n < 10:
            return 0.0, 0.0

        # Subsample for efficiency
        max_n = 100
        if n > max_n:
            indices = np.linspace(0, n - 1, max_n).astype(int)
            trajectory = trajectory[indices]
            n = max_n

        # Compute distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            dist_matrix[i] = np.linalg.norm(trajectory[i] - trajectory, axis=1)

        # Threshold for recurrence
        threshold = np.percentile(dist_matrix[dist_matrix > 0], threshold_percentile)

        # Recurrence matrix
        recurrence = (dist_matrix < threshold).astype(int)

        # Recurrence rate
        rr = np.sum(recurrence) / (n * n)

        # Determinism (fraction of recurrent points forming diagonal lines)
        diag_lengths = []
        for k in range(-n + 2, n - 1):
            diag = np.diag(recurrence, k)
            # Find consecutive 1s
            in_line = False
            length = 0
            for val in diag:
                if val == 1:
                    length += 1
                    in_line = True
                else:
                    if in_line and length >= 2:
                        diag_lengths.append(length)
                    length = 0
                    in_line = False
            if in_line and length >= 2:
                diag_lengths.append(length)

        if diag_lengths:
            det = sum(diag_lengths) / max(1, np.sum(recurrence))
        else:
            det = 0.0

        return float(rr), float(det)

    def _compute_phase_entropy(self, trajectory: np.ndarray) -> float:
        """
        Compute entropy of phase space distribution.

        Args:
            trajectory: Phase space trajectory

        Returns:
            Phase space entropy
        """
        n, dim = trajectory.shape
        if n < 10:
            return 0.0

        # Bin the phase space
        num_bins = max(3, int(n ** (1 / dim)))

        # Flatten to 1D histogram for simplicity
        hist, _ = np.histogramdd(trajectory, bins=num_bins)
        hist = hist.flatten()

        # Normalize
        hist = hist / hist.sum()

        # Compute entropy
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))

        return float(entropy)

    def _classify_attractor(
        self,
        lyapunov: float,
        corr_dim: float,
        recurrence_rate: float,
    ) -> str:
        """
        Classify the type of attractor.

        Args:
            lyapunov: Lyapunov exponent
            corr_dim: Correlation dimension
            recurrence_rate: Recurrence rate

        Returns:
            Attractor type string
        """
        if lyapunov < -0.1:
            return "fixed_point"
        elif lyapunov < 0.01:
            if recurrence_rate > 0.3:
                return "limit_cycle"
            else:
                return "quasi_periodic"
        elif lyapunov < 0.1:
            if corr_dim < 2.5:
                return "weak_chaos"
            else:
                return "strange_attractor"
        else:
            return "strong_chaos"

    def _compute_velocity_anomaly(self, features: KinematicFeatures) -> float:
        """Compute velocity-based anomaly score.

        Uses formula: a = (v_f - v_i) / t for acceleration from velocity change.

        Args:
            features: Kinematic features

        Returns:
            Velocity anomaly score [0, 1]
        """
        z_scores = (
            features.velocity - self._reference_velocity_mean
        ) / self._reference_velocity_std
        max_z = np.max(np.abs(z_scores))
        return float(np.clip(max_z / 3.0, 0.0, 1.0))

    def _compute_acceleration_anomaly(self, features: KinematicFeatures) -> float:
        """
        Compute acceleration-based anomaly score.

        Args:
            features: Kinematic features

        Returns:
            Acceleration anomaly score [0, 1]
        """
        z_scores = (
            features.acceleration - self._reference_acceleration_mean
        ) / self._reference_acceleration_std
        max_z = np.max(np.abs(z_scores))
        return float(np.clip(max_z / 3.0, 0.0, 1.0))

    def _compute_jerk_anomaly(self, features: KinematicFeatures) -> float:
        """
        Compute jerk-based anomaly score.

        High jerk indicates sudden changes in acceleration, often anomalous.

        Args:
            features: Kinematic features

        Returns:
            Jerk anomaly score [0, 1]
        """
        z_scores = (features.jerk - self._reference_jerk_mean) / self._reference_jerk_std
        max_z = np.max(np.abs(z_scores))

        # Apply sensitivity multiplier
        sensitivity = self._dynamics_config.jerk_sensitivity
        return float(np.clip(max_z * sensitivity / 5.0, 0.0, 1.0))

    def _compute_energy_anomaly(self, features: KinematicFeatures) -> float:
        """
        Compute energy conservation anomaly score.

        Checks for unexpected energy changes that violate conservation.

        Args:
            features: Kinematic features

        Returns:
            Energy anomaly score [0, 1]
        """
        # Energy should be relatively constant in conservative systems
        energy_std = np.std(features.total_energy)
        energy_mean = np.mean(features.total_energy) + 1e-10

        # Coefficient of variation
        cv = energy_std / energy_mean

        # Compare to reference
        expected_cv = self._reference_energy_std / (self._reference_energy_mean + 1e-10)
        anomaly = abs(cv - expected_cv) / (expected_cv + 0.1)

        return float(np.clip(anomaly, 0.0, 1.0))

    def _compute_chaos_anomaly(self, features: PhaseSpaceFeatures) -> float:
        """
        Compute chaos-based anomaly score.

        Higher Lyapunov exponents indicate more chaotic (potentially anomalous) behavior.

        Args:
            features: Phase space features

        Returns:
            Chaos anomaly score [0, 1]
        """
        # Compare Lyapunov to reference
        lyapunov_diff = features.lyapunov_exponent - self._reference_lyapunov

        if lyapunov_diff > 0:
            # More chaotic than reference
            score = np.clip(lyapunov_diff / self._dynamics_config.chaos_threshold, 0.0, 1.0)
        else:
            # Less chaotic (potentially anomalous if system should be chaotic)
            score = np.clip(-lyapunov_diff / self._dynamics_config.chaos_threshold, 0.0, 0.5)

        return float(score)

    def _find_anomaly_timestamps(
        self,
        kin_features: KinematicFeatures,
        phase_features: PhaseSpaceFeatures,
    ) -> tuple[list[int], list[str]]:
        """
        Find specific timestamps of anomalies.

        Args:
            kin_features: Kinematic features
            phase_features: Phase space features

        Returns:
            Tuple of (timestamp_indices, descriptions)
        """
        timestamps = []
        descriptions = []

        # Check for velocity anomalies
        v_z = (
            np.abs(kin_features.velocity - self._reference_velocity_mean)
            / self._reference_velocity_std
        )
        v_anomaly_idx = np.where(v_z > 3.0)[0]
        for idx in v_anomaly_idx[:5]:  # Limit to 5
            timestamps.append(int(idx))
            descriptions.append(f"Anomalous velocity at t={idx}: {kin_features.velocity[idx]:.4f}")

        # Check for acceleration anomalies
        a_z = (
            np.abs(kin_features.acceleration - self._reference_acceleration_mean)
            / self._reference_acceleration_std
        )
        a_anomaly_idx = np.where(a_z > 3.0)[0]
        for idx in a_anomaly_idx[:5]:
            if idx not in timestamps:
                timestamps.append(int(idx))
                descriptions.append(
                    f"Anomalous acceleration at t={idx}: {kin_features.acceleration[idx]:.4f}"
                )

        # Check for jerk spikes
        j_z = np.abs(kin_features.jerk - self._reference_jerk_mean) / self._reference_jerk_std
        j_threshold = 3.0 * self._dynamics_config.jerk_sensitivity
        j_anomaly_idx = np.where(j_z > j_threshold)[0]
        for idx in j_anomaly_idx[:5]:
            if idx not in timestamps:
                timestamps.append(int(idx))
                descriptions.append(f"Jerk spike at t={idx}: {kin_features.jerk[idx]:.4f}")

        # Check for energy jumps
        energy_diff = np.abs(np.diff(kin_features.total_energy))
        energy_threshold = 3 * np.std(energy_diff)
        energy_jump_idx = np.where(energy_diff > energy_threshold)[0]
        for idx in energy_jump_idx[:3]:
            if idx not in timestamps:
                timestamps.append(int(idx))
                descriptions.append(f"Energy discontinuity at t={idx}")

        # Sort by timestamp
        sorted_pairs = sorted(zip(timestamps, descriptions))
        timestamps = [p[0] for p in sorted_pairs]
        descriptions = [p[1] for p in sorted_pairs]

        return timestamps, descriptions


# =============================================================================
# Utility Functions
# =============================================================================


def compute_velocity(
    position: np.ndarray,
    time_step: float = 1.0,
    method: str = "central",
) -> np.ndarray:
    """Compute velocity from position using finite differences.

    Implements: v = (x_f - x_i) / t

    Args:
        position: Position array
        time_step: Time step between samples
        method: Differentiation method ("forward", "backward", "central")

    Returns:
        Velocity array
    """
    if method == "forward":
        velocity = np.zeros_like(position)
        velocity[:-1] = (position[1:] - position[:-1]) / time_step
        velocity[-1] = velocity[-2]
    elif method == "backward":
        velocity = np.zeros_like(position)
        velocity[1:] = (position[1:] - position[:-1]) / time_step
        velocity[0] = velocity[1]
    else:  # central
        velocity = np.gradient(position, time_step)

    return velocity


def compute_acceleration(
    velocity: np.ndarray,
    time_step: float = 1.0,
) -> np.ndarray:
    """Compute acceleration from velocity.

    Implements: a = (v_f - v_i) / t

    Args:
        velocity: Velocity array
        time_step: Time step

    Returns:
        Acceleration array
    """
    return np.gradient(velocity, time_step)


def compute_kinetic_energy(
    velocity: np.ndarray,
    mass: float = 1.0,
) -> np.ndarray:
    """Compute kinetic energy.

    Implements: KE = ½mv²

    Args:
        velocity: Velocity array
        mass: Mass value

    Returns:
        Kinetic energy array
    """
    return 0.5 * mass * velocity**2


def compute_momentum(
    velocity: np.ndarray,
    mass: float = 1.0,
) -> np.ndarray:
    """Compute momentum.

    Implements: p = mv

    Args:
        velocity: Velocity array
        mass: Mass value

    Returns:
        Momentum array
    """
    return mass * velocity


def compute_impulse(
    momentum: np.ndarray,
    time_step: float = 1.0,
) -> np.ndarray:
    """Compute impulse (change in momentum).

    Implements: J = Δp = FΔt

    Args:
        momentum: Momentum array
        time_step: Time step

    Returns:
        Impulse array
    """
    return np.gradient(momentum, time_step) * time_step


def estimate_initial_acceleration(
    final_acceleration: float,
    average_acceleration: float,
) -> float:
    """Estimate initial acceleration from final and average.

    Implements: A_i = 2A - A_f

    Args:
        final_acceleration: Final acceleration value
        average_acceleration: Average acceleration

    Returns:
        Estimated initial acceleration
    """
    return 2 * average_acceleration - final_acceleration


def compute_average_acceleration(
    initial_velocity: float,
    final_velocity: float,
    time: float,
) -> float:
    """Compute average acceleration.

    Implements: a = (v_f - v_i) / t

    Args:
        initial_velocity: Initial velocity
        final_velocity: Final velocity
        time: Time interval

    Returns:
        Average acceleration
    """
    if time == 0:
        return 0.0
    return (final_velocity - initial_velocity) / time
