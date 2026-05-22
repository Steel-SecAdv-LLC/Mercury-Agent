"""Tests for :mod:`scripts.check_readme_lyapunov`.

The script gates documentation drift on the Lyapunov decay rate λ.  We
exercise it against both synthetic fixtures (stable across README
edits) AND the shipped README/MATH_SPEC, which forces the gate to keep
matching the prose forms that those documents actually use.
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
    ``configs/lyapunov_canonical.yaml``, this test fails.  The default
    ``--require-hits`` set additionally fails the gate if no claims
    were found at all -- prevents the regex from silently drifting away
    from the prose form used in the shipped docs.
    """
    assert crl.main([]) == 0


def test_real_readme_has_at_least_one_lyapunov_claim() -> None:
    """Positive-coverage guard: the README MUST contain a detectable claim.

    A previous iteration of the gate matched only ``λ=...`` (Greek
    letter) and silently passed on a README that uses the English
    ``lambda = ...`` form -- a vacuous gate.  This test pins the
    invariant directly so any regex regression that re-introduces
    vacuity fails here.
    """
    readme = _REPO_ROOT / "README.md"
    hits = crl.find_lambda_claims(readme.read_text(encoding="utf-8"))
    assert hits, "README.md no longer contains any detectable Lyapunov-λ claim"
    for _, _, val in hits:
        assert val == 0.25


def test_real_math_spec_has_at_least_one_lyapunov_claim() -> None:
    """Positive-coverage guard for the LaTeX ``\\lambda = ...`` form."""
    spec = _REPO_ROOT / "docs" / "MATH_SPEC.md"
    hits = crl.find_lambda_claims(spec.read_text(encoding="utf-8"))
    assert hits, "MATH_SPEC.md no longer contains any detectable Lyapunov-λ claim"
    for _, _, val in hits:
        assert val == 0.25


def test_matching_doc_passes(tmp_path: Path) -> None:
    doc = tmp_path / "ok.md"
    doc.write_text(
        "# Lyapunov Stability\n"
        "Convergence rate `λ=0.25` is enforced.\n"
        "Implementation uses `lambda_lyapunov=0.25`.\n"
    )
    assert crl.main(
        ["--files", str(doc), "--canonical", "0.25", "--require-hits"]
    ) == 0


def test_mismatching_doc_fails(tmp_path: Path) -> None:
    doc = tmp_path / "bad.md"
    doc.write_text(
        "# Lyapunov Stability\n"
        "Convergence rate `λ=0.50` is enforced.\n"  # wrong
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 1


def test_latex_form_is_matched(tmp_path: Path) -> None:
    """LaTeX ``\\lambda = 0.25`` (the form MATH_SPEC.md uses) is enforced."""
    doc = tmp_path / "spec.md"
    doc.write_text(
        "Lyapunov stability theorem: $\\dot V \\leq -\\lambda V$ with "
        "$\\lambda = 0.25$.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0
    doc.write_text(
        "Lyapunov stability theorem: $\\dot V \\leq -\\lambda V$ with "
        "$\\lambda = 0.50$.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 1


def test_english_lambda_word_is_matched(tmp_path: Path) -> None:
    """The English ``lambda = 0.25`` prose form must be enforced.

    The previous regex caught only the Greek letter; a stale
    ``lambda = 0.18`` in README would silently survive.  Pinning the
    behaviour with both a positive and negative case.
    """
    doc = tmp_path / "prose.md"
    doc.write_text(
        "# Lyapunov\n"
        "The fusion-trajectory bound holds with `lambda = 0.25`.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0
    doc.write_text(
        "# Lyapunov\n"
        "The fusion-trajectory bound holds with `lambda = 0.18`.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 1


def test_elevated_from_historical_value_is_excluded(tmp_path: Path) -> None:
    """``λ=0.25 (elevated from 0.18 ...)`` keeps 0.25 and drops 0.18.

    The exclusion is directional: only numbers that appear AFTER
    ``elevated from`` are treated as historical references.  The
    current value cited just before the parenthetical must still be
    enforced.
    """
    doc = tmp_path / "historical.md"
    doc.write_text(
        "Lyapunov stability: convergence rate `λ=0.25` "
        "(elevated from 0.18 for faster convergence).\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0
    # If someone elevates the current λ but forgets to update it, the
    # gate must still catch the new λ.
    doc.write_text(
        "Lyapunov stability: convergence rate `λ=0.30` "
        "(elevated from 0.25 for faster convergence).\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 1


def test_lambda_decay_is_excluded(tmp_path: Path) -> None:
    """``LAMBDA_DECAY = 0.18`` (double-helix adaptation) must NOT trigger.

    The two constants are intentionally distinct; conflating them
    inside the Lyapunov drift gate would produce a false positive on
    a legitimate documentation reference.
    """
    doc = tmp_path / "decay.md"
    doc.write_text(
        "# Lyapunov stability and adaptation\n"
        "The double-helix engine uses `LAMBDA_DECAY = 0.18` for "
        "evolutionary adaptation.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0


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


def test_require_hits_catches_vacuous_gate(tmp_path: Path) -> None:
    """A required file with zero matches must fail under --require-hits.

    Prevents the regex from drifting silently away from the prose form.
    """
    doc = tmp_path / "empty.md"
    doc.write_text(
        "# Lyapunov\n"
        "This document discusses Lyapunov stability but never cites a number.\n"
    )
    # Without --require-hits: passes vacuously.
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0
    # With --require-hits: must fail with a guard message.
    assert crl.main([
        "--files", str(doc),
        "--canonical", "0.25",
        "--require-hits", str(doc),
    ]) == 1


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


def test_lambda_convergence_constant_assignment_is_not_flagged(
    tmp_path: Path,
) -> None:
    """``LAMBDA_CONVERGENCE: float = 0.25`` (code snippet) is not a claim.

    The English-word regex uses a word-boundary anchor so it does not
    match the constant-assignment form (which is the canonical source,
    not a prose claim).
    """
    doc = tmp_path / "code.md"
    doc.write_text(
        "# Lyapunov internals\n"
        "```python\n"
        "LAMBDA_CONVERGENCE: float = 0.18  # would otherwise be a drift\n"
        "```\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0
