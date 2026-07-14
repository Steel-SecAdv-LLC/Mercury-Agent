# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Trained-gate contract for ``MultiHeadAttentionFusion``.

The torch attention path must be unreachable until genuine trained weights
are loaded (EthicalGate convention): historically the module ran its random
initialisation under ``no_grad`` and presented a fixed random projection as
learned fusion.  Untrained inference must be the deterministic phi-weighted
reference average — identical with and without torch installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.core.global_omni_scalar_network import (
    PHI,
    MultiHeadAttentionFusion,
)


def _reference_average(states: list[np.ndarray]) -> np.ndarray:
    padded = []
    for state in states:
        p = np.zeros(37)
        p[: min(len(state), 37)] = state[:37]
        padded.append(p)
    stacked = np.stack(padded)
    phi_weights = np.tile(np.array([PHI, 1.0, 1.0 / PHI]), len(padded) // 3 + 1)[: len(padded)]
    phi_weights = phi_weights / phi_weights.sum()
    return np.average(stacked, axis=0, weights=phi_weights)


def _states() -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    return [rng.normal(size=37) for _ in range(4)]


def test_untrained_fuse_is_the_phi_reference_even_with_torch() -> None:
    fusion = MultiHeadAttentionFusion()
    assert fusion._trained is False
    states = _states()
    result = fusion.fuse(states)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, _reference_average(states))


def test_untrained_fuse_is_deterministic_across_instances() -> None:
    states = _states()
    a = MultiHeadAttentionFusion().fuse(states)
    b = MultiHeadAttentionFusion().fuse(states)
    np.testing.assert_array_equal(a, b)


def test_load_trained_weights_activates_learned_path() -> None:
    donor = MultiHeadAttentionFusion()
    payload = {
        "attention": donor.attention.state_dict(),
        "projection": donor.projection.state_dict(),
        "output_projection": donor.output_projection.state_dict(),
    }
    fusion = MultiHeadAttentionFusion()
    fusion.load_trained_weights(payload)
    assert fusion._trained is True
    states = _states()
    result = fusion.fuse(states)
    assert isinstance(result, np.ndarray)
    # The learned path is a different computation from the reference average.
    assert not np.allclose(result, _reference_average(states))


def test_load_trained_weights_missing_module_fails_loud() -> None:
    fusion = MultiHeadAttentionFusion()
    with pytest.raises(KeyError):
        fusion.load_trained_weights({"attention": fusion.attention.state_dict()})
