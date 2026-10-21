# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verified equation solving: every answer is checked before it is reported.

The solver takes classroom notation (``x^2 - 5x + 6 = 0``, implicit
multiplication like ``2x``, ``e`` and ``pi`` as constants), solves to the
best of its ability -- SymPy symbolic solving first, a deterministic
sign-change scan with ``nsolve`` for transcendental equations second -- and
then **verifies every candidate root by substitution**.  The residual
``|f(root)|`` is measured at 30 significant digits; a candidate whose
residual exceeds the tolerance is *rejected* and surfaced under
``anomalies`` as ``spurious_root_rejected``, never returned as an answer.

Observed anomalies within a solve:

* ``spurious_root_rejected`` -- a candidate failed its own substitution
  check (extraneous roots from algebraic manipulation, failed numeric
  convergence).
* ``near_multiple_root`` -- ``|f'(root)|`` is tiny relative to the root's
  scale, so the root's location is ill-conditioned: a perturbation of the
  equation by ``eps`` moves the root by roughly ``eps / |f'(root)|``.

Honest statuses, per the three-state contract
(:mod:`omni_mercury_engine.verifiers.three_state`): ``grounded`` means the
reported set was verified (including a symbolically established "no
solutions in this domain"); ``unavailable`` means no oracle (SymPy not
installed) or a numeric scan that found nothing -- absence of roots in the
scan window is *not* evidence of nonexistence and is never reported as such.
"""

from __future__ import annotations

import math
import re
from typing import Any

# Optional symbolic oracle (same convention as omni_mercury_engine.utils.constants).
try:
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    SYMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatched flag
    SYMPY_AVAILABLE = False

#: Characters an equation string may contain.  Everything needed for
#: classroom algebra and nothing that can reach Python internals: no
#: underscores, quotes, brackets, braces or backslashes, so attribute and
#: subscript access cannot be spelled at all.
_ALLOWED_CHARS = re.compile(r"^[0-9a-zA-Z+\-*/^ ().,=]+$")

#: A dot adjacent to a letter is attribute-access syntax, not a decimal
#: point; refuse it outright.
_LETTER_DOT = re.compile(r"[A-Za-z]\.|\.[A-Za-z]")

_MAX_EQUATION_LENGTH = 500

#: Deterministic scan window and grid for the numeric fallback.
_SCAN_LO = -20.0
_SCAN_HI = 20.0
_SCAN_POINTS = 81


class EquationSolver:
    """Solve single-variable equations with substitution-verified answers.

    Args:
        residual_tol: Largest ``|f(root)|`` accepted as a verified root.
        conditioning_tol: ``|f'(root)| < conditioning_tol * max(1, |root|)``
            flags the root as near-multiple (ill-conditioned).
    """

    def __init__(self, residual_tol: float = 1e-9, conditioning_tol: float = 1e-6) -> None:
        """Store verification tolerances."""
        if residual_tol <= 0 or conditioning_tol <= 0:
            raise ValueError("tolerances must be positive")
        self.residual_tol = residual_tol
        self.conditioning_tol = conditioning_tol

    def solve(
        self,
        equation: str,
        variable: str | None = None,
        *,
        domain: str = "real",
    ) -> dict[str, Any]:
        """Solve ``equation`` for ``variable`` and verify every answer.

        Args:
            equation: Classroom notation, e.g. ``"x^2 - 5x + 6 = 0"`` or
                ``"cos(x) = x"``.  A bare expression means ``= 0``.  ``e``
                and ``pi`` are the constants; ``^`` is exponentiation;
                ``2x`` multiplies.
            variable: The unknown.  Omitted, the single free symbol is used;
                several free symbols without a choice is refused with the
                symbol list in the message.
            domain: ``"real"`` (default; complex roots are counted under
                ``excluded_by_domain``, not shown) or ``"complex"``.

        Returns:
            A dict with ``status`` (``grounded`` / ``unavailable``),
            ``method`` (``symbolic`` / ``numeric_scan`` / ``none``),
            ``solutions`` (each with ``value``, ``numeric_real``,
            ``numeric_imag``, ``residual``, ``exact``), ``excluded_by_domain``,
            ``anomalies``, ``notes`` and ``oracle``.

        Raises:
            ValueError: Unparseable or unsafe input, an unknown domain, or
                an ambiguous variable -- refused loudly, never guessed.
        """
        if domain not in ("real", "complex"):
            raise ValueError(f"domain must be 'real' or 'complex', got {domain!r}")
        if not SYMPY_AVAILABLE:
            return {
                "equation": equation,
                "variable": variable,
                "domain": domain,
                "status": "unavailable",
                "method": "none",
                "oracle": "unavailable",
                "solutions": [],
                "excluded_by_domain": 0,
                "anomalies": [],
                "notes": [
                    "sympy is not installed: no symbolic oracle, so nothing "
                    "was solved (refusing to guess)"
                ],
            }

        expr, var = self._parse(equation, variable)
        anomalies: list[dict[str, Any]] = []
        notes: list[str] = []
        if expr.has(sympy.sin, sympy.cos, sympy.tan):
            notes.append(
                "periodic equation: the solutions shown are representatives; "
                "the full solution set may be infinite"
            )

        candidates, method, symbolic_ok = self._candidates(expr, var, notes)
        solutions, excluded = self._verify(expr, var, candidates, domain, anomalies)

        if symbolic_ok:
            status = "grounded"
            if not solutions and not excluded:
                notes.append("the symbolic oracle establishes there are no solutions")
        elif solutions:
            status = "grounded"
        else:
            status = "unavailable"
            notes.append(
                f"no roots found by the numeric scan over "
                f"[{_SCAN_LO:g}, {_SCAN_HI:g}]; nonexistence is NOT established"
            )

        return {
            "equation": equation,
            "variable": str(var),
            "domain": domain,
            "status": status,
            "method": method,
            "oracle": "sympy",
            "solutions": solutions,
            "excluded_by_domain": excluded,
            "anomalies": anomalies,
            "notes": notes,
        }

    def _parse(self, equation: str, variable: str | None) -> tuple[Any, Any]:
        """Validate and parse ``equation`` into ``(lhs - rhs, symbol)``."""
        if not equation or len(equation) > _MAX_EQUATION_LENGTH:
            raise ValueError(f"equation must be 1..{_MAX_EQUATION_LENGTH} characters")
        if not _ALLOWED_CHARS.match(equation):
            raise ValueError(
                "equation contains characters outside the allowed set "
                "(digits, letters, + - * / ^ ( ) . , = and spaces)"
            )
        if _LETTER_DOT.search(equation):
            raise ValueError("a dot adjacent to a letter is not a decimal point; refused")
        if equation.count("=") > 1:
            raise ValueError("at most one '=' is allowed")

        transformations = standard_transformations + (
            convert_xor,
            implicit_multiplication_application,
        )
        local_dict = {"e": sympy.E}

        def _part(text: str) -> Any:
            try:
                return parse_expr(text, transformations=transformations, local_dict=local_dict)
            except Exception as exc:  # parse_expr raises SyntaxError/TypeError/token errors
                raise ValueError(f"could not parse {text!r}: {exc}") from exc

        if "=" in equation:
            lhs_text, rhs_text = equation.split("=")
            expr = _part(lhs_text) - _part(rhs_text)
        else:
            expr = _part(equation)

        free = sorted(expr.free_symbols, key=str)
        if variable is not None:
            var = sympy.Symbol(variable)
            if var not in expr.free_symbols:
                raise ValueError(f"variable {variable!r} does not occur in the equation")
        elif len(free) == 1:
            var = free[0]
        elif not free:
            raise ValueError("the equation has no unknown to solve for")
        else:
            raise ValueError(
                f"several unknowns ({', '.join(map(str, free))}); pass variable= to choose one"
            )
        return expr, var

    def _candidates(self, expr: Any, var: Any, notes: list[str]) -> tuple[list[Any], str, bool]:
        """Produce candidate roots: symbolic first, numeric scan as fallback.

        Returns ``(candidates, method, symbolic_ok)`` where ``symbolic_ok``
        records whether the symbolic oracle produced an authoritative answer
        (then an empty list genuinely means "no solutions").
        """
        try:
            roots = sympy.solve(expr, var)
            if roots:
                return list(roots), "symbolic", True
            # solve() returning [] is authoritative for algebraic equations;
            # for others it can mean "could not solve", so only trust it
            # when the expression is polynomial in the unknown.
            if expr.is_polynomial(var):
                return [], "symbolic", True
        except NotImplementedError:
            notes.append("the symbolic solver does not cover this form; scanning numerically")
        return self._numeric_scan(expr, var), "numeric_scan", False

    def _numeric_scan(self, expr: Any, var: Any) -> list[Any]:
        """Deterministic sign-change scan + ``nsolve`` over the real window."""
        step = (_SCAN_HI - _SCAN_LO) / (_SCAN_POINTS - 1)
        grid = [_SCAN_LO + i * step for i in range(_SCAN_POINTS)]

        def value_at(point: float) -> float | None:
            try:
                val = complex(expr.subs(var, sympy.Float(point)).evalf(20))
            except (TypeError, ValueError):
                return None
            if abs(val.imag) > 1e-12:
                return None
            return val.real

        values = [value_at(p) for p in grid]
        roots: list[Any] = []
        for i in range(len(grid) - 1):
            left, right = values[i], values[i + 1]
            if left is None or right is None or left * right > 0:
                continue
            midpoint = (grid[i] + grid[i + 1]) / 2
            try:
                root = sympy.nsolve(expr, var, midpoint, prec=30)
            except (ValueError, ZeroDivisionError, TypeError):
                continue
            if all(abs(complex(root) - complex(seen)) > 1e-9 for seen in roots):
                roots.append(root)
        return roots

    def _verify(
        self,
        expr: Any,
        var: Any,
        candidates: list[Any],
        domain: str,
        anomalies: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Substitution-verify candidates; reject failures as anomalies."""
        derivative = sympy.diff(expr, var)
        solutions: list[dict[str, Any]] = []
        excluded = 0
        for root in candidates:
            if not root.is_number:
                # Parametric solution (e.g. 1 - b): verify symbolically.  Its
                # domain depends on the parameters, so the domain filter does
                # not apply; it is included only when exactly verified.
                if sympy.simplify(expr.subs(var, root)) == 0:
                    solutions.append(
                        {
                            "value": str(root),
                            "numeric_real": float("nan"),
                            "numeric_imag": float("nan"),
                            "residual": 0.0,
                            "exact": True,
                        }
                    )
                else:
                    anomalies.append(
                        {
                            "kind": "spurious_root_rejected",
                            "value": str(root),
                            "reason": "symbolic substitution does not simplify to zero",
                        }
                    )
                continue
            try:
                numeric = complex(root.evalf(30))
            except (TypeError, ValueError):
                anomalies.append(
                    {
                        "kind": "spurious_root_rejected",
                        "value": str(root),
                        "reason": "candidate is not numeric; cannot verify by substitution",
                    }
                )
                continue
            try:
                residual = abs(complex(expr.subs(var, root).evalf(30)))
            except (TypeError, ValueError):
                residual = float("inf")
            if residual > self.residual_tol:
                anomalies.append(
                    {
                        "kind": "spurious_root_rejected",
                        "value": str(root),
                        "residual": float(residual),
                        "reason": "substitution residual exceeds tolerance",
                    }
                )
                continue
            if domain == "real" and abs(numeric.imag) > 1e-12:
                excluded += 1
                continue

            exact = False
            if root.is_number and not root.has(sympy.Float):
                exact = sympy.simplify(expr.subs(var, root)) == 0
            try:
                slope = abs(complex(derivative.subs(var, root).evalf(30)))
            except (TypeError, ValueError):
                slope = float("nan")
            scale = max(1.0, abs(numeric))
            if not math.isnan(slope) and slope < self.conditioning_tol * scale:
                anomalies.append(
                    {
                        "kind": "near_multiple_root",
                        "value": str(root),
                        "derivative_magnitude": float(slope),
                        "reason": (
                            "ill-conditioned: a perturbation eps of the equation "
                            "moves this root by roughly eps / |f'(root)|"
                        ),
                    }
                )
            solutions.append(
                {
                    "value": str(root),
                    "numeric_real": float(numeric.real),
                    "numeric_imag": float(numeric.imag),
                    "residual": float(residual),
                    "exact": bool(exact),
                }
            )
        return solutions, excluded
