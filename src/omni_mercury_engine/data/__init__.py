"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Data Module — deprecation shim for ``omni_mercury_engine.datasets``.

Historical legacy: prior versions exposed every symbol from
``datasets`` (via ``from … import *``).  That wildcard re-export
required a star-import suppression directive and obscured the
actual public surface.  This module now provides the same
backward-compatibility guarantee through PEP 562 ``__getattr__``
forwarding:
attribute lookups against ``omni_mercury_engine.data`` resolve into
``omni_mercury_engine.datasets`` and emit a ``DeprecationWarning``
the first time each name is touched.  The forward is transparent for
runtime callers; for static analysis, the canonical import path is
``omni_mercury_engine.datasets``.

The ``benchmarks`` submodule is still re-exported as an actual
module attribute so ``from omni_mercury_engine.data import
benchmarks`` (and ``from omni_mercury_engine.data.benchmarks import
…``) continue to work without triggering the deprecation warning;
the benchmarks submodule is the one publicly-stable surface this
module is documented to expose.

Deprecated:
    Import from ``omni_mercury_engine.datasets`` instead.  This shim
    will be removed in a future major release.
"""

import warnings
from typing import Any

from omni_mercury_engine.datasets import benchmarks


def __getattr__(name: str) -> Any:
    """Forward every other lookup into ``omni_mercury_engine.datasets``.

    Emits a ``DeprecationWarning`` on each forwarded name so callers
    are nudged toward the canonical import path without breaking
    legacy ``from omni_mercury_engine.data import X`` imports.
    """
    from omni_mercury_engine import datasets as _datasets

    try:
        value = getattr(_datasets, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} "
            f"(also not present in omni_mercury_engine.datasets)"
        ) from None
    warnings.warn(
        f"'omni_mercury_engine.data.{name}' is a deprecated re-export "
        "of 'omni_mercury_engine.datasets'; import from "
        "'omni_mercury_engine.datasets' directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


__all__ = ["benchmarks"]
