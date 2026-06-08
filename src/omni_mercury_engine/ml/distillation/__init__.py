# Copyright (C) 2025 Steel Security Advisors LLC
"""Knowledge Distillation for Anomaly Detection.

Advanced distillation methods for efficient anomaly detection:
- Dual-Student Knowledge Distillation (DSKD)
- Multi-Scale Feature Distillation
- Cosine Similarity Knowledge Distillation (CSKD)

These methods enable training lightweight models while
maintaining detection accuracy.
"""

from __future__ import annotations

from omni_mercury_engine.ml.distillation.dual_student import (
    DualStudentConfig,
    DualStudentDistillation,
)

__all__ = [
    "DualStudentConfig",
    "DualStudentDistillation",
]
