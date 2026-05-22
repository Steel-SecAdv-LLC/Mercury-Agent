"""Mercury Agent developer/operator tooling.

Modules placed under :mod:`tools` are *operator tools*, not part of the
public :mod:`omni_mercury_engine` package surface.  Every module in this
package must remain importable using only dependencies already required
by ``omni_mercury_engine``'s core install (currently ``numpy`` and
``pyyaml``); do not introduce optional/heavy third-party dependencies
here without a corresponding extras-group declaration in
``pyproject.toml``.
"""
