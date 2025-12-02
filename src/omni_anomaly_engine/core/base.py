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
Abstract base classes for detectors, models, and encoders
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn


class BaseDetector(ABC):
    """Abstract base class for all anomaly detectors"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get("threshold", 0.5)
        self._is_fitted = False

    @abstractmethod
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> "BaseDetector":
        """Fit the detector to normal data"""
        pass

    @abstractmethod
    def detect(self, data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Detect anomalies in data"""
        pass

    @abstractmethod
    def extract_features(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract features for ML fusion"""
        pass

    def is_fitted(self) -> bool:
        """Check if detector has been fitted"""
        return self._is_fitted


class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def predict(self, data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Make predictions on data"""
        pass

    @abstractmethod
    def extract_features(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract features for ML fusion"""
        pass


class BaseEncoder(nn.Module):
    """Abstract base class for feature encoders"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features to fixed-size embedding"""
        pass
