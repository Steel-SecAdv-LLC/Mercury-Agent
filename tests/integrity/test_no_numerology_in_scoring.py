# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`scripts.check_no_numerology_in_scoring`.

The script is the **golden-ratio drift gate**: it fails the build if φ
numerology re-enters a scoring path (the Workstream A1 invariant — a reported
score may never be multiplied by an unlearned constant).  These tests give the
gate teeth so it cannot rot into a vacuous pass:

1. **clean repo passes** — the live engine tree carries no scoring-path
   numerology and the gate reports zero violations / exit 0.
2. **each forbidden idiom is detected** — the φ-scalar generator, the scalar
   index, the ``self.golden_ratio`` multiplier (both operand orders), the
   ``self.phi`` multiplier, and the bare ``score * phi`` regression each trip
   the gate on synthetic source.
3. **architectural φ is NOT flagged** — ``int(input_dim * phi)`` and
   ``nn.Linear(d, int(d * phi))`` layer sizing pass, because that is what
   ``test_abms_disciplines::test_golden_ratio_architecture`` validates.
4. **prose is NOT flagged** — comments / docstrings / string literals that
   merely mention a forbidden idiom never produce a false positive (the gate
   is token-based).
5. **allow-listed math passes** — a ``self.phi`` multiplier on a
   legitimate-math path (``core/fusion.py`` GA-optimized coefficient) is
   permitted.
6. **vacuous-green guard** — a scan target with too few files fails with exit
   code 2 rather than a silent green.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import check_no_numerology_in_scoring as gate

_FAKE = Path("fake/module.py")


def _scan(source: str, *, phi_allowlisted: bool = False) -> list[str]:
    """Return the list of violated rule names for ``source``."""
    return [v.rule for v in gate.scan_source(source, _FAKE, phi_allowlisted=phi_allowlisted)]


# ---------------------------------------------------------------------------
# 1. Clean repo passes.
# ---------------------------------------------------------------------------
def test_live_engine_tree_is_clean() -> None:
    violations, files_scanned = gate.check_tree(gate._DEFAULT_ROOT)
    assert (
        files_scanned >= gate.MIN_FILES_SCANNED
    ), f"only {files_scanned} files scanned; the scan target may have moved"
    assert violations == [], "scoring-path numerology detected:\n" + "\n".join(
        f"  [{v.rule}] {v.path}:{v.line_no}: {v.snippet}" for v in violations
    )


def test_main_exit_zero_on_live_tree() -> None:
    assert gate.main([]) == 0


# ---------------------------------------------------------------------------
# 2. Each forbidden idiom is detected.
# ---------------------------------------------------------------------------
def test_detects_omni_scalar_generator() -> None:
    src = (
        "def create_omni_medical_scalars():\n" "    phi = 1.618\n" "    return {'a': 1.42 * phi}\n"
    )
    assert "omni_scalar_generator" in _scan(src)


def test_detects_omni_scalar_index() -> None:
    src = "risk = confidence * self.omni_medical_scalars['omni_diagnostic_precision']\n"
    rules = _scan(src)
    assert "omni_scalar_index" in rules


def test_detects_golden_ratio_multiplier_right_operand() -> None:
    src = "anomaly_detected = risk_score > (0.5 * self.golden_ratio)\n"
    assert "golden_ratio_multiplier" in _scan(src)


def test_detects_golden_ratio_multiplier_left_operand() -> None:
    src = "threshold = self.golden_ratio * std_power\n"
    assert "golden_ratio_multiplier" in _scan(src)


def test_detects_self_phi_multiplier_when_not_allowlisted() -> None:
    src = "risk_score = confidence * self.phi\n"
    assert "phi_attr_multiplier" in _scan(src, phi_allowlisted=False)


def test_detects_bare_score_times_phi_regression() -> None:
    src = "phi = 1.618\nrisk_score = confidence\nout = risk_score * phi\n"
    assert "phi_local_score_multiplier" in _scan(src)


def test_detects_bare_phi_times_score_regression() -> None:
    src = "phi = 1.618\nanomaly_score = phi * anomaly_score\n"
    assert "phi_local_score_multiplier" in _scan(src)


# ---------------------------------------------------------------------------
# 3. Architectural φ is NOT flagged.
# ---------------------------------------------------------------------------
def test_architectural_phi_layer_sizing_is_clean() -> None:
    src = (
        "phi = 1.618\n"
        "hidden_1 = int(input_dim * phi)\n"
        "hidden_2 = int(hidden_1 * phi)\n"
        "layer = nn.Linear(embedding_dim, int(embedding_dim * phi))\n"
    )
    assert _scan(src) == []


# ---------------------------------------------------------------------------
# 4. Prose (comments / docstrings / strings) is NOT flagged.
# ---------------------------------------------------------------------------
def test_comment_mentioning_idiom_is_clean() -> None:
    src = (
        "# former 0.5 * self.golden_ratio threshold removed for integrity\n"
        "# create_omni_medical_scalars used 1.42 * phi historically\n"
        "risk_score = confidence\n"
    )
    assert _scan(src) == []


def test_docstring_mentioning_idiom_is_clean() -> None:
    src = (
        "def f():\n"
        '    """We removed risk_score = confidence * self.golden_ratio here."""\n'
        "    return confidence\n"
    )
    assert _scan(src) == []


def test_string_literal_mentioning_idiom_is_clean() -> None:
    src = "label = 'omni_scalars[x] and create_omni_x_scalars are forbidden'\n"
    assert _scan(src) == []


# ---------------------------------------------------------------------------
# 5. Allow-listed legitimate math passes.
# ---------------------------------------------------------------------------
def test_self_phi_multiplier_permitted_when_allowlisted() -> None:
    src = "strand += self.phi * self._term_Phi(state)\n"
    assert _scan(src, phi_allowlisted=True) == []


def test_core_fusion_path_is_allowlisted() -> None:
    p = gate._DEFAULT_ROOT / "core" / "fusion.py"
    assert gate._allowlisted(p, gate._DEFAULT_ROOT) is True


def test_arbitrary_path_is_not_allowlisted() -> None:
    p = gate._DEFAULT_ROOT / "medical" / "abms_disciplines.py"
    assert gate._allowlisted(p, gate._DEFAULT_ROOT) is False


# ---------------------------------------------------------------------------
# 6. Vacuous-green guard.
# ---------------------------------------------------------------------------
def test_vacuous_green_guard_trips_on_tiny_tree(tmp_path: Path) -> None:
    (tmp_path / "only.py").write_text("x = 1\n", encoding="utf-8")
    assert gate.main(["--root", str(tmp_path)]) == 2


def test_main_reports_violation_on_tiny_tree(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(
        "risk_score = confidence * self.golden_ratio\n", encoding="utf-8"
    )
    # --min-files 1 disables the vacuous guard so the violation surfaces as 1.
    assert gate.main(["--root", str(tmp_path), "--min-files", "1"]) == 1


def test_missing_root_returns_usage_error() -> None:
    assert gate.main(["--root", "/nonexistent/path/xyz"]) == 2
