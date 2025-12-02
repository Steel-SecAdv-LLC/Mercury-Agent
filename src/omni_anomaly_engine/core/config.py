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
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


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
