# Copyright (C) 2025 Steel Security Advisors LLC
"""Cross-component Lyapunov λ reconciliation tests.

The canonical λ for the Mercury Agent fusion-trajectory stability proof
is defined in three places that must stay in lock-step:

* ``configs/lyapunov_canonical.yaml`` -- machine-readable certificate
  consumed by :mod:`tools.lyapunov_validator` and
  :mod:`scripts.run_ablation`.
* ``src/omni_mercury_engine/core/centralized_constants.py``
  ``LyapunovConstants.LAMBDA_CONVERGENCE`` -- the Python constant
  imported by the fusion/three-R/GOSNN stack.
* ``README.md`` and ``docs/MATH_SPEC.md`` prose claims (covered by
  ``scripts/check_readme_lyapunov.py``).

This test asserts the first two stay equal *exactly*, and additionally
asserts the computed certified rate from the canonical (A, P) matrices
is at least as large as the claimed rate.  If anyone changes the
constant or the config in isolation, CI fails here -- not silently in
production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omni_mercury_engine.core.centralized_constants import LYAPUNOV
from tools.lyapunov_validator import validate_lyapunov_from_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_CFG = _REPO_ROOT / "configs" / "lyapunov_canonical.yaml"


def test_canonical_yaml_lambda_matches_lyapunov_constant() -> None:
    """The yaml's claimed λ must equal the in-code LAMBDA_CONVERGENCE."""
    import yaml  # local import, pyyaml is a core runtime dep

    cfg = yaml.safe_load(_CANONICAL_CFG.read_text())
    assert cfg["lambda"] == pytest.approx(LYAPUNOV.LAMBDA_CONVERGENCE), (
        f"configs/lyapunov_canonical.yaml lambda={cfg['lambda']} drifts "
        f"from LyapunovConstants.LAMBDA_CONVERGENCE="
        f"{LYAPUNOV.LAMBDA_CONVERGENCE}"
    )


def test_certified_rate_covers_in_code_lambda() -> None:
    """Computed rate from (A, P) must be >= in-code LAMBDA_CONVERGENCE."""
    ok, details = validate_lyapunov_from_config(_CANONICAL_CFG)
    assert ok, f"canonical config no longer certifies its claim: {details}"
    assert details["computed_lambda"] >= LYAPUNOV.LAMBDA_CONVERGENCE, (
        f"certified lambda={details['computed_lambda']} < "
        f"LYAPUNOV.LAMBDA_CONVERGENCE={LYAPUNOV.LAMBDA_CONVERGENCE}; "
        "update configs/lyapunov_canonical.yaml's (A, P) to certify the "
        "in-code rate, or lower LAMBDA_CONVERGENCE."
    )
