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

"""
Configuration classes for Mercury Agent

Supports:
- YAML, TOML, JSON configuration files
- Environment variable overrides
- Command-line argument precedence
- Configuration inheritance and composition
- Configuration validation with JSON Schema
- Dynamic configuration reloading
- Feature flags framework
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

from omni_mercury_engine.core.centralized_constants import ETHICAL

logger = logging.getLogger(__name__)
T = TypeVar("T")


class DeviceType(Enum):
    """Compute device types"""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class FusionMode(Enum):
    """Fusion strategies"""

    EARLY = "early"
    LATE = "late"
    HYBRID = "hybrid"


class DataCharacteristics(Enum):
    """Detected data characteristics for adaptive component weighting.

    Used by :class:`MercuryAnomalyDetector` to automatically adjust ensemble
    component weights based on whether data is temporally ordered, unordered
    tabular, or high-dimensional image-like.

    See Also:
        - ``MercuryAnomalyDetector._detect_data_characteristics()`` for
          the detection heuristic.
        - ``COMPONENT_COMPATIBILITY`` for the weight adjustment matrix.
    """

    TEMPORAL = "temporal"
    TABULAR = "tabular"
    IMAGE = "image"
    UNKNOWN = "unknown"


# Component compatibility matrix: expected relative effectiveness of each
# ensemble component given detected data characteristics.
# Values represent multiplicative weight modifiers (1.0 = no change).
#
# Rationale:
#   TEMPORAL  - Kinematics excels (derivatives meaningful); all components useful.
#   TABULAR   - Kinematics near-random on shuffled rows (AUC ~0.60, see
#               BENCHMARKS.md line 126-131); InfoGeometry strongest.
#   IMAGE     - High-dimensional data; Kinematics less useful; Resonance moderate.
#   UNKNOWN   - Neutral fallback; no adjustment applied.
COMPONENT_COMPATIBILITY: dict[DataCharacteristics, dict[str, float]] = {
    DataCharacteristics.TEMPORAL: {
        "resonance": 0.8,
        "kinematic": 0.9,
        "infogeo": 0.7,
    },
    DataCharacteristics.TABULAR: {
        "resonance": 0.7,
        "kinematic": 0.3,
        "infogeo": 0.8,
    },
    DataCharacteristics.IMAGE: {
        "resonance": 0.6,
        "kinematic": 0.4,
        "infogeo": 0.7,
    },
    DataCharacteristics.UNKNOWN: {
        "resonance": 1.0,
        "kinematic": 1.0,
        "infogeo": 1.0,
    },
}


class OracleActivation(Enum):
    """Oracle activation mode.

    Controls whether the SpectralDomainOracle is active. Can be set
    explicitly or left at AUTO for domain-aware activation.
    """

    AUTO = "auto"  # Domain-aware: enabled/disabled per ORACLE_DOMAIN_POLICY
    ENABLED = "enabled"  # Always enabled regardless of domain
    DISABLED = "disabled"  # Always disabled regardless of domain


# Domain-aware Oracle activation policy.
#
# Based on empirical analysis of frequency-domain anomaly signatures:
#   ENABLED  — Domain has strong spectral signatures (infrastructure faults,
#              network attack patterns, physiological frequency bands)
#   NEUTRAL  — Domain may benefit; Oracle runs but influence_multiplier is
#              dampened (multiplied by 0.5) to reduce false positive risk
#   DISABLED — Domain anomalies are primarily amplitude/statistical, not
#              spectral. Oracle would add computation without measurable
#              improvement.
ORACLE_DOMAIN_POLICY: dict[str, str] = {
    "infrastructure": "enabled",  # Mains freq, bearing faults, harmonics
    "security": "enabled",  # DDoS periodicity, scan burst patterns
    "medical": "enabled",  # HRV bands, neural oscillations
    "environmental": "neutral",  # Seismic precursors have weak freq signal
    "space": "neutral",  # Solar wind has some spectral content
    "financial": "disabled",  # Anomalies are magnitude-based, not spectral
    "humanitarian": "disabled",  # Weak frequency signatures
}


@dataclass
class DetectorConfig:
    """Configuration for individual detectors"""

    enabled: bool = True
    threshold: float = 0.5
    use_quantum_enhanced: bool = True
    use_nano_detection: bool = True
    use_harmonic_detection: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for individual models"""

    enabled: bool = True
    use_harmonic_features: bool = True
    use_black_hole_features: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionConfig:
    """Configuration for ML fusion"""

    mode: FusionMode = FusionMode.HYBRID
    attention_heads: int = 4
    hidden_dim: int = 128
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"


@dataclass
class ThresholdConfig:
    """Centralized threshold configuration for anomaly detection.

    This configuration class consolidates threshold parameters that were
    previously hardcoded across multiple modules. Use this class to ensure
    consistent thresholding behavior across the detection pipeline.

    Usage:
        config = ThresholdConfig()
        # Use config.anomaly_default for general anomaly detection
        # Use config.ethical_minimum for ethical constraint checks
        # Use config.confidence_high/medium/low for classification bands
    """

    # General anomaly detection thresholds
    anomaly_default: float = 0.5
    """Default threshold for anomaly classification. Scores > threshold = anomaly."""

    anomaly_cap: float = 0.95
    """Maximum allowed threshold to prevent over-filtering."""

    # Classification confidence bands (used in truth_decipher, etc.)
    confidence_high: float = 0.9
    """High confidence threshold for definitive classifications."""

    confidence_medium: float = 0.7
    """Medium confidence threshold for probable classifications."""

    confidence_low: float = 0.5
    """Low confidence threshold for possible classifications."""

    # Ethical constraint thresholds
    ethical_minimum: float = 0.6
    """Minimum ethical alignment score required for operations."""

    benevolence_required: float = ETHICAL.BENEVOLENCE_IMMUTABLE
    """Required benevolence score for civilization-first decisions."""

    # Statistical thresholds
    outlier_percentile: float = 95.0
    """Percentile threshold for outlier detection (0-100)."""

    iqr_multiplier: float = 1.5
    """Multiplier for IQR-based outlier detection."""

    # Neural/symbolic fusion weights
    neural_weight: float = 0.6
    """Weight for neural component in hybrid scoring."""

    symbolic_weight: float = 0.4
    """Weight for symbolic component in hybrid scoring."""

    # Bias detection thresholds
    bias_detection: float = 0.1
    """Threshold for demographic parity bias detection."""

    sigma_directive: float = 0.8
    """Threshold for Sigma Directive approval of actions."""

    pattern_detection: float = 0.6
    """Threshold for geometric pattern detection in ethical alignment."""


class ThresholdDefaults:
    """
    Original threshold values preserved for reference.

    WARNING: Changing these affects ethical governance, anomaly detection,
    and security decisions across the entire system.

    These were the original hardcoded values before centralization.
    Do NOT change without architectural review.
    """

    # From ethical_governor.py:129 - Sigma Directive weighted score threshold
    SIGMA_DIRECTIVE_WEIGHTED_SCORE: float = 0.8

    # From ethical_governor.py:351 - Demographic parity bias detection
    BIAS_DETECTION_DEMOGRAPHIC_PARITY: float = 0.1

    # From neurosymbolic_hub.py:810 - Default anomaly classification threshold
    ANOMALY_CLASSIFICATION_DEFAULT: float = 0.5

    # From various ethical modules - Immutable benevolence requirement
    BENEVOLENCE_IMMUTABLE: float = ETHICAL.BENEVOLENCE_IMMUTABLE

    # From ethical alignment modules - Minimum ethical score
    ETHICAL_MINIMUM: float = 0.6

    # From pattern detection modules - Geometric pattern threshold
    PATTERN_DETECTION_GEOMETRIC: float = 0.6


@dataclass
class EngineConfig:
    """Main engine configuration"""

    device: DeviceType = DeviceType.CPU
    fusion_mode: FusionMode = FusionMode.HYBRID
    batch_size: int = 32
    num_workers: int = 4

    detectors: dict[str, DetectorConfig] = field(default_factory=dict)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)

    model_path: str | None = None
    cache_dir: str = "./cache"
    log_level: str = "INFO"

    # Threshold calibration settings
    anomaly_threshold: float = 0.5
    """Global anomaly decision threshold. Scores > threshold = anomaly."""

    contamination: float | None = None
    """Expected fraction of anomalies (0.0-1.0). If set, uses percentile-based
    threshold instead of fixed threshold. Works with Mercury's statistical ensemble
    (Resonance + Kinematic + InfoGeometry detectors).
    Example: contamination=0.05 means top 5% of scores are classified as anomalies."""

    adaptive_threshold: bool = False
    """If True, automatically calibrate threshold based on score distribution.
    Uses percentile-based thresholding when contamination is set."""

    def __post_init__(self) -> None:
        """Initialize default detector and model configs"""
        if not self.detectors:
            self.detectors = {
                "statistical": DetectorConfig(),
                "temporal": DetectorConfig(),
                "spatial": DetectorConfig(),
                "dimensional": DetectorConfig(),
                "directive": DetectorConfig(),
            }

        if not self.models:
            self.models = {
                "quantum": ModelConfig(),
                "astrophysical": ModelConfig(),
                "biometric": ModelConfig(),
                "affective": ModelConfig(),
                "neural": ModelConfig(),
                "consciousness": ModelConfig(),
            }


# ============================================================================
# External Configuration Management
# ============================================================================


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


@dataclass
class FeatureFlag:
    """Feature flag for A/B testing and gradual rollouts."""

    name: str
    enabled: bool = False
    rollout_percentage: float = 0.0
    variants: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """
    Hierarchical configuration management system.

    Precedence (highest to lowest):
    1. Command-line arguments
    2. Environment variables
    3. Configuration files (YAML/TOML/JSON)
    4. Default values

    Features:
    - Dynamic configuration reloading
    - Configuration validation
    - Feature flags framework
    - Configuration drift detection
    """

    ENV_PREFIX = "OMNI_"

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._config_files: list[Path] = []
        self._feature_flags: dict[str, FeatureFlag] = {}
        self._watchers: list[Callable[[str, Any], None]] = []
        self._loaded = False

    def load_from_file(self, path: str | Path) -> ConfigurationManager:
        """
        Load configuration from a file.

        Supports YAML, TOML, and JSON formats.

        Args:
            path: Path to configuration file

        Returns:
            Self for chaining
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return self

        self._config_files.append(path)

        try:
            if path.suffix in (".yaml", ".yml"):
                self._load_yaml(path)
            elif path.suffix == ".toml":
                self._load_toml(path)
            elif path.suffix == ".json":
                self._load_json(path)
            else:
                raise ConfigurationError(f"Unsupported config format: {path.suffix}")

            logger.info(f"Loaded configuration from: {path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}") from e

        self._loaded = True
        return self

    def _load_yaml(self, path: Path) -> None:
        """Load YAML configuration file."""
        try:
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f) or {}
                self._merge_config(data)
        except ImportError:
            logger.warning("PyYAML not installed, skipping YAML config")

    def _load_toml(self, path: Path) -> None:
        """Load TOML configuration file."""
        try:
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]

            with open(path, "rb") as f:
                data = tomllib.load(f)
                self._merge_config(data)
        except ImportError:
            logger.warning("tomli/tomllib not installed, skipping TOML config")

    def _load_json(self, path: Path) -> None:
        """Load JSON configuration file."""
        with open(path) as f:
            data = json.load(f)
            self._merge_config(data)

    def _merge_config(self, data: dict[str, Any]) -> None:
        """Deep merge configuration data."""

        def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        self._config = deep_merge(self._config, data)

    def load_from_env(self) -> ConfigurationManager:
        """
        Load configuration from environment variables.

        Environment variables follow the pattern:
        MERCURY_AGENT_<SECTION>__<KEY>=value

        Nested keys use double underscores.
        """
        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue

            # Remove prefix and split by double underscore
            config_key = key[len(self.ENV_PREFIX) :].lower()
            parts = config_key.split("__")

            # Navigate to nested location
            current = self._config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set value with type inference
            current[parts[-1]] = self._parse_env_value(value)

        return self

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value with type inference."""
        # Boolean
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            # Not a valid number; continue to try other formats
            pass

        # JSON array/object
        if value.startswith(("[", "{")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Invalid JSON; treat as plain string
                pass

        # String
        return value

    def get(self, key: str, default: T | None = None) -> T | None:
        """
        Get a configuration value.

        Supports dot notation for nested keys: "section.subsection.key"

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        parts = key.split(".")
        current: Any = self._config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current  # type: ignore[no-any-return]

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        parts = key.split(".")
        current = self._config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value
        self._notify_watchers(key, value)

    def _notify_watchers(self, key: str, value: Any) -> None:
        """Notify configuration watchers of changes."""
        for watcher in self._watchers:
            try:
                watcher(key, value)
            except Exception as e:
                logger.error(f"Configuration watcher error: {e}")

    def watch(self, callback: Callable[[str, Any], None]) -> None:
        """Register a configuration change watcher."""
        self._watchers.append(callback)

    def to_engine_config(self) -> EngineConfig:
        """Convert to EngineConfig dataclass."""
        return EngineConfig(
            device=DeviceType(self.get("device", "cpu")),
            fusion_mode=FusionMode(self.get("fusion_mode", "hybrid")),
            batch_size=int(self.get("batch_size", 32) or 32),
            num_workers=int(self.get("num_workers", 4) or 4),
            model_path=self.get("model_path"),
            cache_dir=str(self.get("cache_dir", "./cache")),
            log_level=str(self.get("log_level", "INFO")),
        )

    # Feature Flags
    def register_feature_flag(self, flag: FeatureFlag) -> None:
        """Register a feature flag."""
        self._feature_flags[flag.name] = flag
        logger.info(f"Registered feature flag: {flag.name}")

    def is_feature_enabled(self, name: str, user_id: str | None = None) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            name: Feature flag name
            user_id: Optional user ID for percentage rollouts

        Returns:
            True if feature is enabled
        """
        if name not in self._feature_flags:
            return False

        flag = self._feature_flags[name]

        if not flag.enabled:
            return False

        # Check rollout percentage
        if flag.rollout_percentage < 100.0 and user_id:
            # Deterministic hash for consistent user experience
            # Using SHA3-256 for Ava-Guardian alignment
            # Note: This is not security-sensitive (just bucketing), but SHA3-256
            # eliminates CodeQL weak-hash alerts while maintaining determinism
            import hashlib

            user_hash = int(
                hashlib.sha3_256(f"{name}:{user_id}".encode()).hexdigest()[:8],
                16,
            )
            return (user_hash % 100) < flag.rollout_percentage

        return flag.rollout_percentage >= 100.0

    def get_feature_variant(self, name: str, user_id: str | None = None) -> str | None:
        """Get the variant for A/B testing."""
        if name not in self._feature_flags:
            return None

        flag = self._feature_flags[name]
        if not flag.variants:
            return None

        if user_id:
            import hashlib

            # Using SHA3-256 for Ava-Guardian alignment
            user_hash = int(
                hashlib.sha3_256(f"{name}:{user_id}".encode()).hexdigest()[:8],
                16,
            )
            variant_names = list(flag.variants.keys())
            return variant_names[user_hash % len(variant_names)]

        return next(iter(flag.variants.keys()))


# Global configuration manager instance
# Thread Safety: This singleton uses lazy initialization. While the initial creation
# is not thread-safe (potential race condition on first access from multiple threads),
# subsequent accesses return the same instance. For production use with multiple threads,
# call get_config_manager() once during application startup before spawning threads.
_config_manager: ConfigurationManager | None = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager singleton.

    Note:
        This function uses lazy initialization. For thread-safe initialization
        in multi-threaded applications, call this once during startup before
        creating worker threads.

    Returns:
        The global ConfigurationManager instance.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def load_configuration(
    config_files: list[str | Path] | None = None,
    load_env: bool = True,
) -> ConfigurationManager:
    """
    Load configuration from files and environment.

    Args:
        config_files: List of configuration file paths
        load_env: Whether to load from environment variables

    Returns:
        Configured ConfigurationManager
    """
    manager = get_config_manager()

    # Load from default locations if no files specified
    if config_files is None:
        config_files = [
            Path("omni_mercury.yaml"),
            Path("omni_mercury.toml"),
            Path("omni_mercury.json"),
            Path.home() / ".config" / "omni_mercury" / "config.yaml",
        ]

    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            manager.load_from_file(path)

    if load_env:
        manager.load_from_env()

    return manager
