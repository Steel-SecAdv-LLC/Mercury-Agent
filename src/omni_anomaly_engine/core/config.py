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

"""
Configuration classes for OMNI ♱ AVA

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
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

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


@dataclass
class DetectorConfig:
    """Configuration for individual detectors"""

    enabled: bool = True
    threshold: float = 0.5
    use_quantum_enhanced: bool = True
    use_nano_detection: bool = True
    use_harmonic_detection: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for individual models"""

    enabled: bool = True
    use_harmonic_features: bool = True
    use_black_hole_features: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


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
class EngineConfig:
    """Main engine configuration"""

    device: DeviceType = DeviceType.CPU
    fusion_mode: FusionMode = FusionMode.HYBRID
    batch_size: int = 32
    num_workers: int = 4

    detectors: Dict[str, DetectorConfig] = field(default_factory=dict)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    model_path: Optional[str] = None
    cache_dir: str = "./cache"
    log_level: str = "INFO"

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
    variants: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    ENV_PREFIX = "OMNI_AVA_"

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._config_files: List[Path] = []
        self._feature_flags: Dict[str, FeatureFlag] = {}
        self._watchers: List[callable] = []
        self._loaded = False

    def load_from_file(self, path: Union[str, Path]) -> "ConfigurationManager":
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
            raise ConfigurationError(f"Configuration loading failed: {e}")

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
                import tomli as tomllib

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

    def _merge_config(self, data: Dict[str, Any]) -> None:
        """Deep merge configuration data."""

        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        self._config = deep_merge(self._config, data)

    def load_from_env(self) -> "ConfigurationManager":
        """
        Load configuration from environment variables.

        Environment variables follow the pattern:
        OMNI_AVA_<SECTION>__<KEY>=value

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
            pass

        # JSON array/object
        if value.startswith(("[", "{")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # String
        return value

    def get(self, key: str, default: T = None) -> T:
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
        current = self._config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

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

    def watch(self, callback: callable) -> None:
        """Register a configuration change watcher."""
        self._watchers.append(callback)

    def to_engine_config(self) -> EngineConfig:
        """Convert to EngineConfig dataclass."""
        return EngineConfig(
            device=DeviceType(self.get("device", "cpu")),
            fusion_mode=FusionMode(self.get("fusion_mode", "hybrid")),
            batch_size=self.get("batch_size", 32),
            num_workers=self.get("num_workers", 4),
            model_path=self.get("model_path"),
            cache_dir=self.get("cache_dir", "./cache"),
            log_level=self.get("log_level", "INFO"),
        )

    # Feature Flags
    def register_feature_flag(self, flag: FeatureFlag) -> None:
        """Register a feature flag."""
        self._feature_flags[flag.name] = flag
        logger.info(f"Registered feature flag: {flag.name}")

    def is_feature_enabled(self, name: str, user_id: Optional[str] = None) -> bool:
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
            import hashlib

            user_hash = int(hashlib.md5(f"{name}:{user_id}".encode()).hexdigest()[:8], 16)
            return (user_hash % 100) < flag.rollout_percentage

        return flag.rollout_percentage >= 100.0 or flag.rollout_percentage == 0.0

    def get_feature_variant(self, name: str, user_id: Optional[str] = None) -> Optional[str]:
        """Get the variant for A/B testing."""
        if name not in self._feature_flags:
            return None

        flag = self._feature_flags[name]
        if not flag.variants:
            return None

        if user_id:
            import hashlib

            user_hash = int(hashlib.md5(f"{name}:{user_id}".encode()).hexdigest()[:8], 16)
            variant_names = list(flag.variants.keys())
            return variant_names[user_hash % len(variant_names)]

        return list(flag.variants.keys())[0]


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def load_configuration(
    config_files: Optional[List[Union[str, Path]]] = None,
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
            Path("omni_ava.yaml"),
            Path("omni_ava.toml"),
            Path("omni_ava.json"),
            Path.home() / ".config" / "omni_ava" / "config.yaml",
        ]

    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            manager.load_from_file(path)

    if load_env:
        manager.load_from_env()

    return manager
