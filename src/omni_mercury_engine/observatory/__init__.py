# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Mathematical Observatory: solve what is solvable, observe the rest for clues.

Three honest capabilities, none of them theater:

* :mod:`equation_solver` -- solves equations to the best of its ability
  (SymPy symbolic first, scanned ``nsolve`` numerics second) and **verifies
  every solution by substitution** before reporting it.  A root that fails
  its own residual check is rejected and surfaced as an anomaly, never
  returned as an answer.
* The instruments (:mod:`zeta_spectrum`, :mod:`goldbach_comet`,
  :mod:`collatz_records`) point Mercury's core competency -- anomaly
  detection -- at mathematical data itself.  This is the methodology behind
  real discoveries (Birch--Swinnerton-Dyer was conjectured from EDSAC
  numerics; the Montgomery--Odlyzko GUE law came from zero-spacing data):
  compute the objects, measure the structure, flag what deviates.
* Counterexample routing: an observation that *refutes* a universal claim
  is the one way an instance can settle an open conjecture.  Each
  instrument reports ``counterexample_candidates`` shaped for the
  corresponding :class:`~omni_mercury_engine.verifiers.registry.MysteryRegistry`
  submission (e.g. an even number with no Goldbach partition goes to
  ``submit_goldbach``, which can GROUND the refutation).

Everything fails closed: a missing oracle (SymPy / mpmath) is reported as
UNAVAILABLE with zero results, never papered over with fabricated values,
per the three-state contract (:mod:`omni_mercury_engine.verifiers.three_state`).
"""

from __future__ import annotations

from omni_mercury_engine.observatory.collatz_records import CollatzRecordsInstrument
from omni_mercury_engine.observatory.equation_solver import EquationSolver
from omni_mercury_engine.observatory.goldbach_comet import GoldbachCometInstrument
from omni_mercury_engine.observatory.zeta_spectrum import ZetaSpectrumInstrument

__all__ = [
    "CollatzRecordsInstrument",
    "EquationSolver",
    "GoldbachCometInstrument",
    "ZetaSpectrumInstrument",
]
