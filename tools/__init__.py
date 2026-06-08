# Copyright (C) 2025 Steel Security Advisors LLC
"""Mercury Agent developer/operator tooling.

Modules placed under :mod:`tools` are *operator tools*, not part of the
public :mod:`omni_mercury_engine` package surface.  Every module in this
package is importable using only dependencies that ship with the
``mercury-agent`` core install: currently :mod:`numpy` and
:mod:`yaml` (PyYAML), both declared as core dependencies in
``pyproject.toml``.  Adding a module that pulls in any other
third-party library requires either declaring the new dep in
``pyproject.toml`` *or* introducing a corresponding extras group --
silent reliance on a transitive dep is not acceptable for operator
tooling that must run on minimal installs.
"""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
