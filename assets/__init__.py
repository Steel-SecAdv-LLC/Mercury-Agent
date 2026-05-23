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

------------------------------------------------------------------------

Repository-local ``assets`` package.

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
