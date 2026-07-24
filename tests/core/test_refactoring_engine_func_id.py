# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression test: RefactoringEngine must not collide same-named callables.

``analyze_with_harmonics`` cached results under ``f"{module}.{name}"``, which is
identical for distinct callables sharing a module and bare name (e.g. same-named
methods on different classes). With caching enabled the second callable received
the first's cached metrics. The cache identity now also incorporates the
qualified name and code location.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.three_r_mechanism import RefactoringEngine, _stable_func_id


class _Alpha:
    def run(self) -> int:
        return 1


class _Beta:
    def run(self) -> int:
        total = 0
        for i in range(10):
            for j in range(10):
                if (i + j) % 3 == 0:
                    total += i * j
                elif i > j:
                    total -= j
                else:
                    total += 1
        return total


def test_stable_func_id_distinguishes_same_named_methods() -> None:
    assert _stable_func_id(_Alpha.run) != _stable_func_id(_Beta.run)


def test_analyze_with_harmonics_does_not_collide_on_shared_name() -> None:
    engine = RefactoringEngine()  # caching enabled by default

    alpha = engine.analyze_with_harmonics(_Alpha.run)
    beta = engine.analyze_with_harmonics(_Beta.run)

    # _Beta.run is substantially more complex than _Alpha.run; if the two
    # collided on one cache key, beta would echo alpha's metrics verbatim.
    assert alpha != beta
    assert np.isfinite(alpha.get("complexity", {}).get("cyclomatic_complexity", 0.0))
