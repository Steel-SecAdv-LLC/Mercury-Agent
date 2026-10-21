# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bio-Threat Detection Module (Medical Interdiction).

Flags sensor snapshots whose Gaussian energy against a fitted baseline exceeds
a false-alarm-calibrated threshold, with MASINT agent reports for typing.

Part of Medical Interdiction and Intervention framework.
"""

from __future__ import annotations

from omni_mercury_engine.medical.pandemic.bio_threats.pathogen_detector import (
    BioThreatResult,
    PathogenDetector,
)

__all__ = ["BioThreatResult", "PathogenDetector"]
