# Copyright (C) 2025 Steel Security Advisors LLC
"""Cross-domain anomaly fusion subpackage for Mercury Agent.

This package hosts multi-modal anomaly detectors that combine ML, rule,
and adapter signals into a single decision-support output.  Single-mode
algorithmic detectors (statistical, spectral, dimensional) live in
:mod:`omni_mercury_engine.detectors`.  Single-domain telemetry detectors
live in :mod:`omni_mercury_engine.detectors.<domain>` (for example
:mod:`omni_mercury_engine.detectors.drone`,
:mod:`omni_mercury_engine.detectors.marine`,
:mod:`omni_mercury_engine.detectors.energy`).

Use ``anomaly/`` only for cross-domain fusion that consumes two or more
``detectors/<domain>/`` outputs.  No such fusion detector ships in this
release; the package is retained as the documented home for future
work so the architectural intent is not lost.
"""

from __future__ import annotations

__all__: list[str] = []
