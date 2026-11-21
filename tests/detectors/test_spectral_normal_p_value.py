# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The spectral detector's normal p-value is ``scipy.stats.norm``'s, exactly.

``_two_sided_normal_p`` evaluates ``scipy.special.ndtr`` directly instead of
going through the ``norm`` distribution object, because the band scorer calls
it for every band of every window and the object's argument handling was
most of the cost (about 25 microseconds a call against 0.07). The detector's
published numbers must not move with that, so equality is held here bit for
bit rather than to a tolerance.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from scipy.stats import norm

from omni_mercury_engine.detectors.spectral_domain_frequency import _two_sided_normal_p


def _reference(z: float) -> float:
    """The expression the detector used before: through the distribution object."""
    return float(2.0 * (1.0 - norm.cdf(abs(z))))


def test_p_value_equals_the_distribution_object_bit_for_bit() -> None:
    rng = np.random.default_rng(0)
    statistics = np.concatenate(
        [
            rng.normal(0.0, 3.0, 20_000),
            rng.uniform(-40.0, 40.0, 5_000),
            10.0 ** rng.uniform(-300.0, 3.0, 5_000),
            np.array([0.0, -0.0, 5e-324, 1e-320, 8.3, -8.3, 37.0, 38.5, 1e308, np.inf, -np.inf]),
        ]
    )
    mismatched = [float(z) for z in statistics if _two_sided_normal_p(float(z)) != _reference(z)]
    assert mismatched == []


def test_p_value_endpoints() -> None:
    assert _two_sided_normal_p(0.0) == 1.0
    assert _two_sided_normal_p(math.inf) == 0.0
    assert _two_sided_normal_p(-math.inf) == 0.0
    assert math.isnan(_two_sided_normal_p(math.nan))
    assert type(_two_sided_normal_p(1.3)) is float


def test_p_value_is_symmetric_and_decreasing() -> None:
    grid = np.linspace(0.0, 8.0, 161)
    values = [_two_sided_normal_p(float(z)) for z in grid]
    assert values == [_two_sided_normal_p(-float(z)) for z in grid]
    assert all(later <= earlier for earlier, later in pairwise(values))
    assert abs(_two_sided_normal_p(1.959963984540054) - 0.05) < 1e-12
