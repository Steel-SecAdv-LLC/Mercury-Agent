# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`scripts.check_readme_lyapunov`.

The script is the **import-based** λ drift gate: it imports the
canonical constants from the package and asserts the documented numeric
literals match.  These tests exercise the registry behaviour against
synthetic fixtures *and* the shipped ``README.md`` / ``docs/MATH_SPEC.md``,
so the gate stays honest as the docs evolve.

Five mandated test categories from the doctrine are covered:

1. **clean repo passes** — the live ``README`` and ``MATH_SPEC`` agree
   with the imported canonicals.
2. **tampered README λ fails** — a single bad numeric trips the gate.
3. **tampered MATH_SPEC λ fails** — same, against the LaTeX prose form.
4. **deleting a documented mention trips the min_occurrences floor** —
   a vacuous-green pass is no longer possible.
5. **monkeypatched canonical fails until docs follow** — moving the
   runtime constant without updating the docs fails immediately, and
   updating the docs to match restores green.

Plus the original behavioural pins (Greek/LaTeX/English forms,
``elevated from`` directional exclusion, ``LAMBDA_DECAY`` separation,
constant-assignment-form rejection) so a regex-side regression cannot
re-introduce vacuity.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import check_readme_lyapunov as crl

if TYPE_CHECKING:
    import pytest
    from _pytest.monkeypatch import MonkeyPatch

# ---------------------------------------------------------------------------
# Category 1: clean repo passes.
# ---------------------------------------------------------------------------


def test_real_readme_and_math_spec_pass() -> None:
    """The shipped docs must agree with both imported canonical constants.

    Live regression: if anyone edits ``README.md`` or ``docs/MATH_SPEC.md``
    and changes a Lyapunov-context λ value without updating
    ``LyapunovConstants.LAMBDA_CONVERGENCE`` or
    ``double_helix_engine.LAMBDA_DECAY``, this test fails.
    """
    assert crl.main([]) == 0


def test_real_readme_has_at_least_one_lambda_lyapunov_claim() -> None:
    """Positive-coverage guard: README MUST contain a detectable λ_lyapunov claim.

    A previous iteration of the gate matched only ``λ=...`` (Greek
    letter) and silently passed on a README that uses the English
    ``lambda = ...`` form -- a vacuous gate.  This test pins the
    invariant directly so any regex regression that re-introduces
    vacuity fails here.
    """
    readme = _REPO_ROOT / "README.md"
    hits = crl.find_lambda_claims(
        readme.read_text(encoding="utf-8"),
        check=_check_by_name("lambda_lyapunov"),
    )
    assert hits, "README.md no longer contains any detectable λ_lyapunov claim"
    for _, _, val in hits:
        assert val == 0.25


def test_real_math_spec_has_at_least_one_lambda_lyapunov_claim() -> None:
    """Positive-coverage guard for MATH_SPEC's LaTeX ``\\lambda = ...`` form."""
    spec = _REPO_ROOT / "docs" / "MATH_SPEC.md"
    hits = crl.find_lambda_claims(
        spec.read_text(encoding="utf-8"),
        check=_check_by_name("lambda_lyapunov"),
    )
    assert hits, "MATH_SPEC.md no longer contains any detectable λ_lyapunov claim"
    for _, _, val in hits:
        assert val == 0.25


def test_real_readme_has_at_least_one_lambda_decay_claim() -> None:
    """README MUST contain a detectable λ_decay (LAMBDA_DECAY) mention.

    The ``min_occurrences`` floor on the ``lambda_decay`` check would
    catch silent deletion; this assertion catches the per-check
    regression directly.
    """
    readme = _REPO_ROOT / "README.md"
    hits = crl.find_lambda_claims(
        readme.read_text(encoding="utf-8"),
        check=_check_by_name("lambda_decay"),
    )
    assert hits, "README.md no longer contains any detectable λ_decay claim"
    for _, _, val in hits:
        assert val == 0.18


# ---------------------------------------------------------------------------
# Categories 2 & 3: tampered README / MATH_SPEC fail.
# ---------------------------------------------------------------------------


def test_tampered_readme_lambda_lyapunov_fails(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A README claim that disagrees with LAMBDA_CONVERGENCE fails with a useful message."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Lyapunov stability\n"
        "Convergence rate `λ = 0.25` (elevated from 0.18 for faster convergence).\n"
        "Implementation uses `lambda_lyapunov=0.25`.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` controls the double-helix adaptation rate.\n"
    )
    # Clean baseline first.
    assert crl.main(["--files", str(readme)]) == 0

    # Tamper: bump README's λ to 0.30 while the runtime constant stays at 0.25.
    readme.write_text(
        "# Lyapunov stability\n"
        "Convergence rate `λ = 0.30` (elevated from 0.18 for faster convergence).\n"
        "Implementation uses `lambda_lyapunov=0.30`.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` controls the double-helix adaptation rate.\n"
    )
    rc = crl.main(["--files", str(readme)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "[lambda_lyapunov]" in err
    assert "claimed lambda_lyapunov=0.3" in err
    assert "canonical lambda_lyapunov=0.25" in err


def test_tampered_math_spec_lambda_lyapunov_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A MATH_SPEC LaTeX claim that disagrees with LAMBDA_CONVERGENCE fails."""
    spec = tmp_path / "MATH_SPEC.md"
    spec.write_text(
        "## §2.2 Lyapunov stability theorem\n"
        "We claim $\\dot V \\leq -\\lambda V$ with $\\lambda = 0.25$.\n"
        "The convergence rate $\\lambda = 0.25$ is enforced by the executable certificate.\n"
        "Glossary: LAMBDA_DECAY = 0.18 for the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(spec)]) == 0

    spec.write_text(
        "## §2.2 Lyapunov stability theorem\n"
        "We claim $\\dot V \\leq -\\lambda V$ with $\\lambda = 0.50$.\n"
        "The convergence rate $\\lambda = 0.50$ is enforced by the executable certificate.\n"
        "Glossary: LAMBDA_DECAY = 0.18 for the double-helix adaptation rate.\n"
    )
    rc = crl.main(["--files", str(spec)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "[lambda_lyapunov]" in err
    assert "claimed lambda_lyapunov=0.5" in err


def test_tampered_lambda_decay_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tampering the documented LAMBDA_DECAY value fails the λ_decay check."""
    doc = tmp_path / "README.md"
    doc.write_text(
        "# Lyapunov stability\n"
        "Convergence rate `λ = 0.25` (elevated from 0.18 for faster convergence).\n"
        "Glossary: `LAMBDA_DECAY = 0.99` controls the double-helix adaptation rate.\n"
    )
    rc = crl.main(["--files", str(doc)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "[lambda_decay]" in err
    assert "claimed lambda_decay=0.99" in err
    assert "canonical lambda_decay=0.18" in err


# ---------------------------------------------------------------------------
# Category 4: min_occurrences floor catches silent deletion.
# ---------------------------------------------------------------------------


def test_deleting_lambda_lyapunov_trips_min_occurrences_floor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A file with zero λ_lyapunov mentions trips the vacuous-green guard."""
    doc = tmp_path / "README.md"
    doc.write_text(
        "# Stability discussion\n"
        "This document discusses stability but never cites a numeric Lyapunov claim.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` controls the double-helix adaptation rate.\n"
    )
    rc = crl.main(["--files", str(doc)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "[lambda_lyapunov]" in err
    assert "vacuous-green guard" in err


def test_deleting_lambda_decay_trips_min_occurrences_floor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A file with zero λ_decay mentions trips the vacuous-green guard."""
    doc = tmp_path / "README.md"
    doc.write_text(
        "# Lyapunov stability only\n"
        "Convergence rate `λ = 0.25` is the certified bound.\n"
        "Implementation uses `lambda_lyapunov=0.25`.\n"
        # No LAMBDA_DECAY mention anywhere.
    )
    rc = crl.main(["--files", str(doc)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "[lambda_decay]" in err
    assert "vacuous-green guard" in err


# ---------------------------------------------------------------------------
# Category 5: monkeypatched canonical fails until docs follow.
# ---------------------------------------------------------------------------


def test_monkeypatched_lambda_convergence_fails_until_docs_follow(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Moving the runtime constant must fail until every documented mention matches.

    This is the *whole point* of the import-based gate: the value at
    the call site of ``check.canonical_provider()`` IS the source of
    truth.  We monkeypatch the import target so the next provider call
    returns the new value, then assert the gate fails on the stale
    docs and passes after the docs are updated to match.
    """
    doc = tmp_path / "README.md"
    doc.write_text(
        "# Lyapunov stability\n"
        "Convergence rate `λ = 0.25` (elevated from 0.18 for faster convergence).\n"
        "Implementation uses `lambda_lyapunov=0.25`.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` controls the double-helix adaptation rate.\n"
    )
    # Baseline: passes because runtime λ_lyapunov == 0.25 == docs.
    assert crl.main(["--files", str(doc)]) == 0

    # Override the import-based provider for λ_lyapunov so it returns
    # a new value.  This is exactly the failure mode the gate exists
    # to catch: somebody updates the constant in code and forgets the
    # documentation.
    new_lambda = 0.3
    monkeypatch.setattr(crl, "_import_lambda_convergence", lambda: new_lambda)

    rc = crl.main(["--files", str(doc)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "[lambda_lyapunov]" in err
    assert "canonical lambda_lyapunov=0.3" in err
    # Every existing documented λ_lyapunov mention now reads as drift.
    assert "claimed lambda_lyapunov=0.25" in err

    # Update the docs to match the new constant and the gate must go green.
    doc.write_text(
        "# Lyapunov stability\n"
        "Convergence rate `λ = 0.3` (elevated from 0.18 for faster convergence).\n"
        "Implementation uses `lambda_lyapunov=0.3`.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` controls the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc)]) == 0


def test_monkeypatched_lambda_decay_fails_until_docs_follow(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Same contract for the λ_decay check."""
    doc = tmp_path / "README.md"
    doc.write_text(
        "# Lyapunov stability\n"
        "Convergence rate `λ = 0.25` is the certified bound.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` controls the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc)]) == 0

    monkeypatch.setattr(crl, "_import_lambda_decay", lambda: 0.42)
    assert crl.main(["--files", str(doc)]) == 1

    doc.write_text(
        "# Lyapunov stability\n"
        "Convergence rate `λ = 0.25` is the certified bound.\n"
        "Glossary: `LAMBDA_DECAY = 0.42` controls the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc)]) == 0


# ---------------------------------------------------------------------------
# Behavioural pins (forms, exclusions, edge cases) carried forward from the
# pre-refactor test surface so a regex-side regression cannot re-introduce
# vacuity even if the registry-level invariants above are satisfied.
# ---------------------------------------------------------------------------


def test_matching_doc_passes(tmp_path: Path) -> None:
    doc = tmp_path / "ok.md"
    doc.write_text(
        "# Lyapunov Stability\n"
        "Convergence rate `λ=0.25` is enforced.\n"
        "Implementation uses `lambda_lyapunov=0.25`.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` for the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0


def test_mismatching_doc_fails(tmp_path: Path) -> None:
    doc = tmp_path / "bad.md"
    doc.write_text(
        "# Lyapunov Stability\n"
        "Convergence rate `λ=0.50` is enforced.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` for the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 1


def test_latex_form_is_matched(tmp_path: Path) -> None:
    doc = tmp_path / "spec.md"
    doc.write_text(
        "Lyapunov stability theorem: $\\dot V \\leq -\\lambda V$ with $\\lambda = 0.25$.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` for the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0
    doc.write_text(
        "Lyapunov stability theorem: $\\dot V \\leq -\\lambda V$ with $\\lambda = 0.50$.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` for the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 1


def test_english_lambda_word_is_matched(tmp_path: Path) -> None:
    """The English ``lambda = 0.25`` prose form must be enforced.

    The previous regex caught only the Greek letter; a stale
    ``lambda = 0.18`` in README would silently survive.  Pin both
    positive and negative cases.
    """
    doc = tmp_path / "prose.md"
    doc.write_text(
        "# Lyapunov\n"
        "The fusion-trajectory bound holds with `lambda = 0.25`.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` for the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0
    doc.write_text(
        "# Lyapunov\n"
        "The fusion-trajectory bound holds with `lambda = 0.18`.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` for the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 1


def test_elevated_from_historical_value_is_excluded_from_lambda_lyapunov(
    tmp_path: Path,
) -> None:
    """``λ=0.25 (elevated from 0.18 ...)`` keeps 0.25 for λ_lyapunov.

    The exclusion is directional: numbers AFTER ``elevated from`` are
    historical and belong to ``λ_decay`` (covered by its own
    enforcement); numbers before are the current ``λ_lyapunov`` claim
    and must still be enforced.
    """
    doc = tmp_path / "historical.md"
    doc.write_text(
        "Lyapunov stability: convergence rate `λ=0.25` "
        "(elevated from 0.18 for faster convergence).\n"
        "Glossary: `LAMBDA_DECAY = 0.18` for the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0


def test_lambda_decay_constant_is_NOT_flagged_against_lambda_lyapunov(
    tmp_path: Path,
) -> None:
    """``LAMBDA_DECAY = 0.18`` is the *λ_decay* claim, not a λ_lyapunov drift."""
    doc = tmp_path / "decay.md"
    doc.write_text(
        "# Lyapunov stability and adaptation\n"
        "Convergence rate `λ = 0.25` is the certified bound.\n"
        "The double-helix engine uses `LAMBDA_DECAY = 0.18` for evolutionary adaptation.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0


def test_non_lyapunov_lambda_is_ignored(tmp_path: Path) -> None:
    """Unrelated mentions of λ outside any anchor context must NOT trigger."""
    doc = tmp_path / "ok.md"
    doc.write_text(
        "# Uncertainty fusion\n"
        "We use the entropy weight `λ=0.99` for calibration.\n"
        # No Lyapunov anchor tokens within 4 lines.
        "Glossary line for λ_decay: `LAMBDA_DECAY = 0.18` for the double-helix.\n"
    )
    # Should pass: 0.99 isn't picked up (no λ_lyapunov anchor near it),
    # and LAMBDA_DECAY = 0.18 satisfies the λ_decay min_occurrences floor.
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0


def test_missing_file_reports_error(tmp_path: Path) -> None:
    """A non-existent --files argument must fail."""
    assert crl.main(["--files", str(tmp_path / "nope.md"), "--canonical", "0.25"]) == 1


def test_find_lambda_claims_dedupes_overlapping_patterns() -> None:
    """Two patterns matching the same span shouldn't double-count."""
    text = "Lyapunov stability: convergence rate λ=0.25 is fixed."
    hits = crl.find_lambda_claims(text)
    assert len(hits) == 1
    assert hits[0][2] == 0.25


def test_lambda_convergence_constant_assignment_is_not_flagged(
    tmp_path: Path,
) -> None:
    """``LAMBDA_CONVERGENCE: float = 0.25`` (code snippet) is not a prose claim.

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
        "Glossary: `LAMBDA_DECAY = 0.18` for the double-helix adaptation rate.\n"
    )
    assert crl.main(["--files", str(doc), "--canonical", "0.25"]) == 0


def test_canonical_override_only_affects_first_check(
    tmp_path: Path,
) -> None:
    """``--canonical`` is the legacy single-check escape hatch.

    It overrides the *first* check's canonical (historically
    ``lambda_lyapunov``) so synthetic tests can exercise the gate
    without an editable install of the full package; the rest of the
    registry continues to import normally.  This test pins the
    contract so a future refactor doesn't accidentally make
    ``--canonical`` global.
    """
    doc = tmp_path / "x.md"
    doc.write_text(
        "Lyapunov: `λ = 0.7` is enforced.\n"
        "Glossary: `LAMBDA_DECAY = 0.18` for the double-helix adaptation rate.\n"
    )
    # λ_lyapunov override = 0.7 ⇒ docs match the override; λ_decay still
    # imports 0.18 ⇒ docs match too ⇒ green.
    assert crl.main(["--files", str(doc), "--canonical", "0.7"]) == 0


def _check_by_name(name: str) -> crl.LambdaCheck:
    """Return the :class:`crl.LambdaCheck` with the given ``name``."""
    for check in crl.CHECKS:
        if check.name == name:
            return check
    raise AssertionError(f"no LambdaCheck named {name!r} in registry")
