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
Theorem-tier verifier: an external formal proof checker (Lean 4) as the oracle.

Where the number-theory verifiers confirm *instances*, this confirms a *theorem* -- but only
when a proof object exists.  The proof script is the certificate; the Lean kernel is the oracle.
A correct proof compiles (valid); a hallucinated or wrong proof fails to compile (refuted).
That rejection is the point: it is exactly the apparatus that tells you, honestly, whether any
candidate proof -- from a human, a search, or a model -- is real.

This verifier never fakes a verdict.  If the Lean toolchain is absent it reports
``available=False`` and registers no scalar; it returns ``valid=True`` only when Lean's kernel
accepts the proof.
"""

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup

if TYPE_CHECKING:
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

logger = logging.getLogger(__name__)

# Known theorems provable in core Lean 4 (no mathlib), for demonstrating the path end to end.
KNOWN_THEOREM: str = "theorem two_plus_two : 2 + 2 = 4 := rfl\n"
FALSE_THEOREM: str = "theorem wrong : 2 + 2 = 5 := rfl\n"


@dataclass(frozen=True)
class LeanVerdict:
    """Result of submitting a proof script to the Lean kernel."""

    valid: bool
    available: bool
    reason: str
    checker: str = "lean4"

    def as_metadata(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "available": self.available,
            "reason": self.reason,
            "checker": self.checker,
        }


def lean_available() -> bool:
    """Whether a Lean executable is on PATH."""
    return shutil.which("lean") is not None


def verify_lean_proof(source: str, *, timeout: float = 60.0) -> LeanVerdict:
    """Submit a Lean proof script to the kernel and report its verdict.

    Returns ``available=False`` (and ``valid=False``) when no Lean toolchain is installed --
    an honest "no oracle here", never a fabricated pass.  Otherwise ``valid`` reflects whether
    Lean's kernel accepted the proof (exit code 0 with no error diagnostics).
    """
    lean = shutil.which("lean")
    if lean is None:
        return LeanVerdict(False, False, "lean toolchain not available on PATH")

    with tempfile.TemporaryDirectory() as tmp:
        proof_path = Path(tmp) / "proof.lean"
        proof_path.write_text(source, encoding="utf-8")
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, path under our control
                [lean, str(proof_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return LeanVerdict(False, True, f"lean timed out after {timeout}s")

    diagnostics = (result.stdout + result.stderr).strip()
    if result.returncode == 0 and "error" not in diagnostics.lower():
        return LeanVerdict(True, True, "lean kernel accepted the proof")
    return LeanVerdict(False, True, diagnostics[:500] or f"lean exited {result.returncode}")


def register_verified_theorem(
    gosnn: GlobalOmniScalarNetwork,
    source: str,
    *,
    theorem_name: str,
    component_name: str = "mathematical_mysteries_theorem",
    timeout: float = 60.0,
) -> tuple[float | None, LeanVerdict]:
    """Check ``source`` with Lean and ground a scalar from the verdict.

    Registers 1.0 (kernel-accepted) or 0.0 (kernel-rejected).  When Lean is unavailable, returns
    ``(None, verdict)`` and registers nothing -- no oracle, no scalar.
    """
    verdict = verify_lean_proof(source, timeout=timeout)
    if not verdict.available:
        logger.info("Theorem scalar not registered (%s)", verdict.reason)
        return None, verdict

    scalar_value = 1.0 if verdict.valid else 0.0
    gosnn.register_scalars(
        component_name=component_name,
        scalars={f"omni_mystery_theorem_{theorem_name}_verified": scalar_value},
        group=ScalarGroup.MATHEMATICAL_MYSTERIES,
        metadata={"source": "lean_oracle", "theorem": theorem_name, **verdict.as_metadata()},
    )
    logger.info(
        "Theorem '%s' scalar grounded to %.1f (%s)", theorem_name, scalar_value, verdict.reason
    )
    return scalar_value, verdict


def demonstrate() -> None:
    """End-to-end demonstration of the theorem path (live where Lean is installed)."""
    if not lean_available():
        print("[lean] toolchain not installed here -- reporting UNAVAILABLE (no faked verdict).")
        verdict = verify_lean_proof(KNOWN_THEOREM)
        print(f"[lean] available={verdict.available} valid={verdict.valid} ({verdict.reason})")
        print("[lean] install Lean 4 (`elan`) and re-run to verify the known theorem live.")
        return

    good = verify_lean_proof(KNOWN_THEOREM)
    print(f"[lean] KNOWN  '2 + 2 = 4': valid={good.valid} ({good.reason})")
    bad = verify_lean_proof(FALSE_THEOREM)
    print(f"[lean] FALSE  '2 + 2 = 5': valid={bad.valid} ({bad.reason})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demonstrate()
