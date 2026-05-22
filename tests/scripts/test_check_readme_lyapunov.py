"""Tests for :mod:`scripts.check_readme_lyapunov`.

The script gates documentation drift on the Lyapunov decay rate λ.  We
exercise it against synthetic markdown fixtures so the regression
remains stable even when the real README evolves.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import check_readme_lyapunov as crl


def test_real_readme_and_math_spec_pass() -> None:
    """The shipped docs must agree with the canonical λ.

    This is the live regression: if anyone edits README.md or
    docs/MATH_SPEC.md and changes a Lyapunov-context λ value without
    updating ``LyapunovConstants.LAMBDA_CONVERGENCE`` and
    ``configs/lyapunov_canonical.yaml``, this test fails.
    """
    assert crl.main([]) == 0


def test_matching_doc_passes(tmp_path: Path) -> None:
    doc = tmp_path / "ok.md"
    doc.write_text(
        "# Lyapunov Stability\n"
        "Convergence rate `λ=0.25` is enforced.\n"
        "Implementation uses `lambda_lyapunov=0.25`.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0


def test_mismatching_doc_fails(tmp_path: Path) -> None:
    doc = tmp_path / "bad.md"
    doc.write_text(
        "# Lyapunov Stability\n"
        "Convergence rate `λ=0.50` is enforced.\n"  # wrong
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 1


def test_non_lyapunov_lambda_is_ignored(tmp_path: Path) -> None:
    """Unrelated mentions of λ must NOT trigger the gate."""
    doc = tmp_path / "ok.md"
    doc.write_text(
        "# Uncertainty fusion\n"
        "We use the entropy weight `λ=0.99` for calibration.\n"
        # No Lyapunov-context tokens within 4 lines; the value is
        # outside any stability context and must be ignored.
    )
    # canonical is 0.25; a naive scanner would flag 0.99 here.
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0


def test_missing_file_reports_error(tmp_path: Path) -> None:
    assert (
        crl.main(["--files", str(tmp_path / "nope.md"), "--canonical", "0.25"])
        == 1
    )


def test_find_lambda_claims_dedupes_overlapping_patterns() -> None:
    """Two patterns matching the same span shouldn't double-count."""
    text = "Lyapunov stability: convergence rate λ=0.25 is fixed."
    hits = crl.find_lambda_claims(text)
    # Should be exactly one hit despite both patterns matching.
    assert len(hits) == 1
    assert hits[0][2] == 0.25
