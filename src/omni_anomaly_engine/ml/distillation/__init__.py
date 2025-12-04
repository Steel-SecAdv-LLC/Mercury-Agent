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
Knowledge Distillation for Anomaly Detection

Advanced distillation methods for efficient anomaly detection:
- Dual-Student Knowledge Distillation (DSKD)
- Multi-Scale Feature Distillation
- Cosine Similarity Knowledge Distillation (CSKD)

These methods enable training lightweight models while
maintaining detection accuracy.
"""

from omni_anomaly_engine.ml.distillation.dual_student import (
    DualStudentDistillation,
    DualStudentConfig,
)

__all__ = [
    "DualStudentDistillation",
    "DualStudentConfig",
]
