# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""EquationSolver: every reported answer is substitution-verified, or it is not reported."""

from __future__ import annotations

import pytest

import omni_mercury_engine.observatory.equation_solver as solver_module
from omni_mercury_engine.observatory import EquationSolver


@pytest.fixture
def solver() -> EquationSolver:
    """A solver at default tolerances."""
    return EquationSolver()


class TestSymbolicSolving:
    """Exact classroom algebra, verified exactly."""

    def test_quadratic_exact_roots(self, solver: EquationSolver) -> None:
        """x^2 - 5x + 6 = 0 has exactly {2, 3}, verified exact with residual 0."""
        result = solver.solve("x^2 - 5x + 6 = 0")
        assert result["status"] == "grounded"
        assert result["method"] == "symbolic"
        values = sorted(s["value"] for s in result["solutions"])
        assert values == ["2", "3"]
        assert all(s["exact"] for s in result["solutions"])
        assert all(s["residual"] == 0.0 for s in result["solutions"])
        assert result["anomalies"] == []

    def test_implicit_multiplication_and_constants(self, solver: EquationSolver) -> None:
        """Classroom notation: 2x multiplies, e is Euler's number."""
        result = solver.solve("2x + e = 0")
        assert len(result["solutions"]) == 1
        assert result["solutions"][0]["numeric_real"] == pytest.approx(-1.3591409142295225)

    def test_no_real_solutions_is_grounded_with_exclusion_count(
        self, solver: EquationSolver
    ) -> None:
        """x^2 + 1 = 0 in the real domain: zero solutions, two excluded, grounded."""
        result = solver.solve("x^2 + 1 = 0")
        assert result["status"] == "grounded"
        assert result["solutions"] == []
        assert result["excluded_by_domain"] == 2

    def test_complex_domain_shows_both_roots(self, solver: EquationSolver) -> None:
        """The complex filter reveals +/- i, each verified."""
        result = solver.solve("x^2 + 1 = 0", domain="complex")
        imags = sorted(s["numeric_imag"] for s in result["solutions"])
        assert imags == pytest.approx([-1.0, 1.0])
        assert result["excluded_by_domain"] == 0

    def test_polynomial_empty_solve_is_authoritative(
        self, solver: EquationSolver, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty solve() on a polynomial is grounded emptiness, not unavailable."""
        monkeypatch.setattr(
            "omni_mercury_engine.observatory.equation_solver.sympy.solve", lambda expr, var: []
        )
        result = solver.solve("x^2 + 1 = 0", domain="complex")
        assert result["status"] == "grounded"
        assert result["method"] == "symbolic"
        assert result["solutions"] == []
        assert any("no solutions" in note for note in result["notes"])

    def test_parametric_solution_verified_symbolically(self, solver: EquationSolver) -> None:
        """Solving a + b = 1 for a gives the exact parametric answer 1 - b."""
        result = solver.solve("a + b = 1", variable="a")
        assert result["status"] == "grounded"
        assert [s["value"] for s in result["solutions"]] == ["1 - b"]
        assert result["solutions"][0]["exact"] is True
        assert result["solutions"][0]["residual"] == 0.0


class TestNumericFallback:
    """Transcendental equations: deterministic scan + nsolve, still verified."""

    def test_dottie_number(self, solver: EquationSolver) -> None:
        """cos(x) = x has the single real root 0.7390851332151607 (Dottie)."""
        result = solver.solve("cos(x) = x")
        assert result["status"] == "grounded"
        assert len(result["solutions"]) == 1
        assert result["solutions"][0]["numeric_real"] == pytest.approx(
            0.7390851332151607, abs=1e-12
        )
        assert result["solutions"][0]["residual"] < 1e-9

    def test_periodic_note_present(self, solver: EquationSolver) -> None:
        """Trig equations carry the representatives-only honesty note."""
        result = solver.solve("sin(x) = 0")
        assert any("representatives" in note for note in result["notes"])

    def test_rootless_scan_is_unavailable_not_grounded(self, solver: EquationSolver) -> None:
        """exp(x) = 0 finds nothing numerically: UNAVAILABLE, nonexistence NOT claimed."""
        result = solver.solve("exp(x) = 0")
        assert result["status"] == "unavailable"
        assert result["solutions"] == []
        assert any("NOT established" in note for note in result["notes"])


class TestVerificationGate:
    """A candidate that fails substitution is an anomaly, never an answer."""

    def test_spurious_root_rejected(
        self, solver: EquationSolver, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wrong candidate from the root source is rejected by its residual."""
        import sympy

        monkeypatch.setattr(
            "omni_mercury_engine.observatory.equation_solver.sympy.solve",
            lambda expr, var: [sympy.Integer(7)],
        )
        result = solver.solve("x - 1 = 0")
        assert result["solutions"] == []
        assert len(result["anomalies"]) == 1
        assert result["anomalies"][0]["kind"] == "spurious_root_rejected"
        assert result["anomalies"][0]["residual"] == pytest.approx(6.0)

    def test_double_root_flagged_ill_conditioned(self, solver: EquationSolver) -> None:
        """(x-1)^2 = 0: the root is reported AND flagged near_multiple_root."""
        result = solver.solve("(x - 1)^2 = 0")
        assert [s["value"] for s in result["solutions"]] == ["1"]
        kinds = [a["kind"] for a in result["anomalies"]]
        assert kinds == ["near_multiple_root"]

    def test_simple_root_not_flagged(self, solver: EquationSolver) -> None:
        """x - 1 = 0 is well-conditioned: no anomaly."""
        result = solver.solve("x - 1 = 0")
        assert result["anomalies"] == []


class TestInputGuards:
    """Unsafe or ambiguous input is refused loudly, never guessed at."""

    @pytest.mark.parametrize(
        "equation",
        [
            "x.__class__ = 0",
            "__import__(os)",
            "x[0] = 1",
            "x = 'a'",
            "lambda : 0",  # colon outside the charset
        ],
    )
    def test_unsafe_characters_refused(self, solver: EquationSolver, equation: str) -> None:
        """Underscores, quotes, brackets and colons never reach the parser."""
        with pytest.raises(ValueError, match="allowed set"):
            solver.solve(equation)

    def test_letter_adjacent_dot_refused(self, solver: EquationSolver) -> None:
        """A dot next to a letter is attribute syntax, not a decimal point."""
        with pytest.raises(ValueError, match="dot adjacent"):
            solver.solve("x.imag = 0")

    def test_length_cap(self, solver: EquationSolver) -> None:
        """Over-long input is refused."""
        with pytest.raises(ValueError, match="characters"):
            solver.solve("x + " * 200 + "1")

    def test_two_equals_refused(self, solver: EquationSolver) -> None:
        """At most one '=' sign."""
        with pytest.raises(ValueError, match="at most one"):
            solver.solve("x = 1 = 2")

    def test_ambiguous_variable_refused_with_symbol_list(self, solver: EquationSolver) -> None:
        """Several unknowns without a choice name the candidates."""
        with pytest.raises(ValueError, match="a, b"):
            solver.solve("a + b = 1")

    def test_unknown_domain_refused(self, solver: EquationSolver) -> None:
        """Only real/complex are domains."""
        with pytest.raises(ValueError, match="domain"):
            solver.solve("x = 1", domain="rational")

    def test_nonpositive_tolerances_refused(self) -> None:
        """Tolerances must be positive."""
        with pytest.raises(ValueError, match="positive"):
            EquationSolver(residual_tol=0.0)


class TestFailClosed:
    """No oracle means no answers -- never fabrication."""

    def test_without_sympy_reports_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SYMPY_AVAILABLE False: unavailable status, empty results, honest note."""
        monkeypatch.setattr(solver_module, "SYMPY_AVAILABLE", False)
        result = EquationSolver().solve("x^2 = 4")
        assert result["status"] == "unavailable"
        assert result["oracle"] == "unavailable"
        assert result["solutions"] == []
        assert any("sympy is not installed" in note for note in result["notes"])
