# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Repository-local ``assets`` package.

Houses deterministic, reproducible *simulated*-data generators that
the real-data validation suites (``tests/cyber``, ``tests/emergent``,
``tests/medical``) use as drop-in stand-ins for licensed/proprietary
upstream datasets (CICIDS, MIMIC-III, SETI@home Allen array, etc.).

The synthesis pipelines are first-party: they encode the documented
statistical structure of each domain (e.g. MIMIC-III septic shock
trajectories, SETI narrow-band candidate signals) so the detectors
under test exercise their full code paths against data whose ground
truth is *known by construction*.
"""

from __future__ import annotations

from . import loaders

__all__ = ["loaders"]
