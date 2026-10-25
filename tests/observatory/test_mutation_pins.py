# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kill-pins for measured observatory mutation survivors.

Each test targets a named cluster from the full-enumeration mutation run:
documented-contract boundaries (length caps, validation edges, defaults),
verification-tolerance behavior, scan geometry, and exact instrument
arithmetic re-derived independently in the test body.
"""

from __future__ import annotations

import inspect
import math

import pytest

from omni_mercury_engine.observatory import (
    CollatzRecordsInstrument,
    EquationSolver,
    GoldbachCometInstrument,
    ZetaSpectrumInstrument,
)
from omni_mercury_engine.observatory.goldbach_comet import TWIN_PRIME_CONSTANT
from omni_mercury_engine.observatory.zeta_spectrum import gue_spacing_cdf


class TestSolverContractBoundaries:
    """The documented 500-char cap and validation edges, exactly."""

    def test_500_chars_accepted_501_rejected(self) -> None:
        """The cap is inclusive at 500 and refuses at 501."""
        solver = EquationSolver()
        padded_500 = "x = 0" + " " * 495
        assert len(padded_500) == 500
        assert solver.solve(padded_500)["status"] == "grounded"
        with pytest.raises(ValueError, match="characters"):
            solver.solve(padded_500 + " ")

    def test_zero_conditioning_tolerance_refused(self) -> None:
        """conditioning_tol=0 is refused exactly like residual_tol=0."""
        with pytest.raises(ValueError, match="positive"):
            EquationSolver(conditioning_tol=0.0)

    def test_default_residual_tolerance_rejects_near_roots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A candidate off by 1e-4 fails the default 1e-9 gate (not a loose 0.1)."""
        import sympy

        monkeypatch.setattr(
            "omni_mercury_engine.observatory.equation_solver.sympy.solve",
            lambda expr, var: [sympy.Float("1.0001", 20)],
        )
        result = EquationSolver().solve("x - 1 = 0")
        assert result["solutions"] == []
        assert result["anomalies"][0]["kind"] == "spurious_root_rejected"

    def test_conditioning_threshold_is_exclusive_at_the_default(self) -> None:
        """|f'| exactly 1e-6 (and a 0.01 slope) is NOT flagged under the defaults."""
        for equation in ("0.000001x - 0.000001 = 0", "0.01x - 0.01 = 0"):
            result = EquationSolver().solve(equation)
            assert [s["value"] for s in result["solutions"]], equation
            assert result["anomalies"] == [], equation

    def test_unavailable_result_counts_zero_exclusions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fail-closed dict reports excluded_by_domain exactly 0."""
        import omni_mercury_engine.observatory.equation_solver as solver_module

        monkeypatch.setattr(solver_module, "SYMPY_AVAILABLE", False)
        assert EquationSolver().solve("x = 1")["excluded_by_domain"] == 0

    def test_no_solutions_note_absent_when_roots_were_excluded(self) -> None:
        """Complex roots hidden by the real filter must not read as 'no solutions'."""
        result = EquationSolver().solve("x^2 + 1 = 0")
        assert not any("no solutions" in note for note in result["notes"])

    def test_near_real_roots_still_excluded_by_domain(self) -> None:
        """Roots at 1 +/- 0.001i are complex: excluded, never shown as real."""
        result = EquationSolver().solve("x^2 - 2x + 1.000001 = 0")
        assert result["solutions"] == []
        assert result["excluded_by_domain"] == 2


class TestNumericScanGeometry:
    """The sign-change scan's documented behavior at its own edges."""

    @staticmethod
    def _force_numeric(monkeypatch: pytest.MonkeyPatch) -> None:
        def no_symbolic(expr: object, var: object) -> list[object]:
            raise NotImplementedError

        monkeypatch.setattr(
            "omni_mercury_engine.observatory.equation_solver.sympy.solve", no_symbolic
        )

    def test_root_exactly_on_a_grid_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A zero landing on the 0.5-step grid is found once, not skipped or doubled."""
        self._force_numeric(monkeypatch)
        result = EquationSolver().solve("x - 1/2 = 0")
        assert len(result["solutions"]) == 1
        assert result["solutions"][0]["numeric_real"] == pytest.approx(0.5, abs=1e-12)

    def test_root_in_the_last_scan_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A root at 19.75 sits in the final [19.5, 20] interval and is found."""
        self._force_numeric(monkeypatch)
        result = EquationSolver().solve("x - 19.75 = 0")
        assert len(result["solutions"]) == 1
        assert result["solutions"][0]["numeric_real"] == pytest.approx(19.75, abs=1e-9)

    def test_float_root_is_not_exact(self) -> None:
        """A numerically found root is honest about not being exact."""
        result = EquationSolver().solve("cos(x) = x")
        assert result["solutions"][0]["exact"] is False


class TestZetaSpectrumPins:
    """Validation edges, documented defaults, and the KS machinery re-derived."""

    def test_documented_defaults(self) -> None:
        """The docstring's defaults (50 zeros, 0.5 threshold, top 5) are the signature's."""
        params = inspect.signature(ZetaSpectrumInstrument.observe).parameters
        assert params["num_zeros"].default == 50
        assert params["close_pair_threshold"].default == 0.5
        assert params["top_k"].default == 5

    def test_validation_edges_inclusive(self) -> None:
        """num_zeros=2 and top_k=1 are legal; a zero threshold is not."""
        result = ZetaSpectrumInstrument().observe(2, top_k=1)
        assert result["num_zeros"] == 2
        assert len(result["closest_pairs"]) == 1
        with pytest.raises(ValueError, match="close_pair_threshold"):
            ZetaSpectrumInstrument().observe(5, close_pair_threshold=0.0)

    def test_spacing_count_and_pair_indices(self) -> None:
        """10 zeros give exactly 9 spacings; the first pair is zeros #1-#2."""
        result = ZetaSpectrumInstrument().observe(10, close_pair_threshold=100.0, top_k=9)
        assert len(result["close_pair_anomalies"]) == 9
        pair_one = next(p for p in result["closest_pairs"] if p["lower_zero_index"] == 1)
        assert pair_one["upper_zero_index"] == 2

    def test_ks_distance_on_a_synthetic_gue_sample(self) -> None:
        """A sample at the (i+0.5)/n CDF quantiles has KS distance exactly 0.5/n."""
        n = 50

        def inverse_cdf(target: float) -> float:
            lo, hi = 0.0, 6.0
            for _ in range(200):
                mid = (lo + hi) / 2
                if gue_spacing_cdf(mid) < target:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2

        sample = [inverse_cdf((i + 0.5) / n) for i in range(n)]
        distance = ZetaSpectrumInstrument._ks_distance(sample)
        assert distance == pytest.approx(0.5 / n, abs=1e-9)


class TestGoldbachPins:
    """The constant, the edges, and the whole pipeline re-derived independently."""

    def test_twin_prime_constant_against_its_definition(self) -> None:
        """C2 matches the partial Euler product over primes to 1e5 within its tail."""
        limit = 100_000
        flags = GoldbachCometInstrument._sieve(limit)
        product = 1.0
        for p in range(3, limit + 1):
            if flags[p]:
                product *= 1.0 - 1.0 / (p - 1) ** 2
        assert pytest.approx(product, abs=1e-5) == TWIN_PRIME_CONSTANT

    def test_documented_defaults(self) -> None:
        """observe defaults: limit 10000, top_k 5."""
        params = inspect.signature(GoldbachCometInstrument.observe).parameters
        assert params["limit"].default == 10000
        assert params["top_k"].default == 5

    def test_validation_edges_inclusive(self) -> None:
        """limit=100 and top_k=1 are legal."""
        result = GoldbachCometInstrument().observe(100, top_k=1)
        assert result["limit"] == 100
        assert len(result["ratio_outliers"]) == 1

    def test_band_densities_rederived_exactly(self) -> None:
        """Band means at limit=100 equal an independent re-computation to 1e-12."""
        result = GoldbachCometInstrument().observe(100)

        sums = {0: 0.0, 1: 0.0, 2: 0.0}
        counts = {0: 0, 1: 0, 2: 0}
        primes = [p for p in range(2, 101) if all(p % d for d in range(2, p))]
        prime_set = set(primes)
        for n in range(4, 101, 2):
            r = sum(1 for p in primes if p <= n // 2 and (n - p) in prime_set)
            assert r > 0, f"unexpected Goldbach counterexample candidate at {n}"
            sums[n % 3] += r * math.log(n) ** 2 / n
            counts[n % 3] += 1
        for residue in (0, 1, 2):
            expected = sums[residue] / counts[residue]
            assert result["band_density_by_mod3"][residue] == pytest.approx(
                expected, abs=1e-12
            ), residue


class TestCollatzPins:
    """Defaults, edges, the n=2 base record, and the exact budget boundary."""

    def test_documented_defaults(self) -> None:
        """observe defaults: limit 50000, step_budget 1,000,000."""
        params = inspect.signature(CollatzRecordsInstrument.observe).parameters
        assert params["limit"].default == 50000
        assert params["step_budget"].default == 1_000_000

    def test_validation_edges_inclusive(self) -> None:
        """limit=10 and step_budget=1 are legal."""
        result = CollatzRecordsInstrument().observe(10, step_budget=1)
        assert result["limit"] == 10
        assert result["status"] == "grounded"

    def test_base_record_is_n_equals_2(self) -> None:
        """The ladder starts at n=2: one step, peak 2."""
        result = CollatzRecordsInstrument().observe(10)
        first = result["stopping_time_records"][0]
        assert first["n"] == 2
        assert first["stopping_time"] == 1
        assert first["peak"] == 2

    def test_budget_boundary_is_exact(self) -> None:
        """64 needs exactly 6 halvings: a budget of 5 makes it a candidate, 6 does not."""
        at_five = CollatzRecordsInstrument().observe(64, step_budget=5)
        assert 64 in at_five["counterexample_candidates"]
        at_six = CollatzRecordsInstrument().observe(64, step_budget=6)
        assert 64 not in at_six["counterexample_candidates"]


class TestSieveAndRangeBounds:
    """Bound-exactness pins the small-limit tests cannot see."""

    def test_sieve_strikes_perfect_prime_squares(self) -> None:
        """121 = 11^2 and 169 = 13^2 are composite: the k <= sqrt(limit) bound is exact."""
        flags_121 = GoldbachCometInstrument._sieve(121)
        assert flags_121[121] is False
        assert flags_121[113] is True
        flags_169 = GoldbachCometInstrument._sieve(169)
        assert flags_169[169] is False

    def test_collatz_never_checks_beyond_its_limit(self) -> None:
        """With a tiny budget every candidate stays within 2..limit — 65 never leaks in."""
        result = CollatzRecordsInstrument().observe(64, step_budget=5)
        assert all(2 <= n <= 64 for n in result["counterexample_candidates"])
        assert 65 not in result["counterexample_candidates"]
