# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic construction of the fusion feature path's random components.

Checkpoints persist fitted state (ROADMAP row 16), which makes *loaded*
engines exact; these tests pin the complementary contract for
*checkpoint-free* engines: components whose parameters are never fit — the
temporal LSTM projector, the dimensional autoencoder's init, the default
multiverse population — are architecture constants built under forked
fixed-seed RNG, not draws from the process-global stream. Two engines
constructed (and fitted on the same data) in different RNG realities must
therefore extract bit-identical fusion features.

Measured at the commit before these fixes, three groups were
instance-dependent under scrambled ambient seeds: ``dimensional``
(max|Δ| ≈ 3.01), ``multiverse`` (≈ 0.98) and ``temporal`` (≈ 0.27).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector
from omni_mercury_engine.engine import OmniMercuryEngine
from omni_mercury_engine.models.multiverse import MultiverseOmniEngine
from omni_mercury_engine.utils.rng import DeterministicRNG


class TestTemporalProjectorConstant:
    def test_lstm_weights_are_instance_independent(self) -> None:
        """Every instance carries bit-identical (untrained) projector weights."""
        torch.manual_seed(0)
        first = TemporalAnomalyDetector()
        torch.manual_seed(999)
        second = TemporalAnomalyDetector()
        assert first.lstm is not None and second.lstm is not None
        state_a, state_b = first.lstm.state_dict(), second.lstm.state_dict()
        assert sorted(state_a) == sorted(state_b)
        for key, tensor in state_a.items():
            assert torch.equal(tensor, state_b[key]), f"lstm.{key} differs across instances"

    def test_construction_leaves_global_rng_untouched(self) -> None:
        """fork_rng isolation: building a detector consumes no ambient draws."""
        torch.manual_seed(42)
        expected = torch.randn(8)
        torch.manual_seed(42)
        TemporalAnomalyDetector()
        actual = torch.randn(8)
        assert torch.equal(expected, actual)


class TestDimensionalFitDeterminism:
    def test_fit_is_a_function_of_its_data(self) -> None:
        """Same data -> bit-identical autoencoder, whatever the ambient seeds."""
        data = np.random.default_rng(5).normal(size=(64, 9))

        torch.manual_seed(0)
        np.random.seed(0)
        first = DimensionalAnalyzer()
        first.fit(data)
        torch.manual_seed(999)
        np.random.seed(999)
        second = DimensionalAnalyzer()
        second.fit(data)

        assert first.autoencoder is not None and second.autoencoder is not None
        state_a = first.autoencoder.state_dict()
        state_b = second.autoencoder.state_dict()
        for key, tensor in state_a.items():
            assert torch.equal(tensor, state_b[key]), f"autoencoder.{key} differs across fits"
        queries = np.random.default_rng(6).normal(size=(16, 9))
        np.testing.assert_array_equal(
            np.asarray(first.detect(queries)["scores"]),
            np.asarray(second.detect(queries)["scores"]),
        )


class TestMultiverseDefaultPopulation:
    def test_default_universes_are_instance_independent(self) -> None:
        """Default-constructed engines host bit-identical initial universes."""
        np.random.seed(0)
        first = MultiverseOmniEngine(num_universes=6, state_dim=12)
        np.random.seed(999)
        second = MultiverseOmniEngine(num_universes=6, state_dim=12)

        assert list(first.universes) == list(second.universes)
        for universe_id, universe in first.universes.items():
            np.testing.assert_array_equal(
                universe.state_vector, second.universes[universe_id].state_vector
            )

    def test_universe_ids_hash_only_the_index(self) -> None:
        """IDs are stable across constructions — no wall-clock component."""
        ids_a = list(MultiverseOmniEngine(num_universes=4, state_dim=8).universes)
        ids_b = list(MultiverseOmniEngine(num_universes=4, state_dim=8).universes)
        assert ids_a == ids_b

    def test_injected_rng_still_controls_initialization(self) -> None:
        """An explicit generator overrides the fixed-seed default population."""
        default = MultiverseOmniEngine(num_universes=4, state_dim=8)
        injected = MultiverseOmniEngine(num_universes=4, state_dim=8, rng=DeterministicRNG(123))
        repeat = MultiverseOmniEngine(num_universes=4, state_dim=8, rng=DeterministicRNG(123))

        default_states = np.stack([u.state_vector for u in default.universes.values()])
        injected_states = np.stack([u.state_vector for u in injected.universes.values()])
        repeat_states = np.stack([u.state_vector for u in repeat.universes.values()])
        assert not np.array_equal(default_states, injected_states)
        np.testing.assert_array_equal(injected_states, repeat_states)


class TestFusionFeaturesInstanceIndependent:
    def test_features_are_instance_independent(self) -> None:
        """Every fusion feature group is a pure function of the input.

        Two engines constructed under scrambled RNG realities (a stand-in for
        two different processes) must extract identical features for every
        group — the property that, combined with fitted-state persistence,
        makes checkpoints reproduce exactly (ROADMAP row 16).
        """
        x = np.random.default_rng(7).normal(size=(32, 12)).astype(np.float32)

        torch.manual_seed(0)
        np.random.seed(0)
        first = OmniMercuryEngine(mode="fusion", device="cpu")
        torch.manual_seed(999)
        np.random.seed(999)
        second = OmniMercuryEngine(mode="fusion", device="cpu")

        features_a = first._extract_fusion_features(x, fit_detectors=True)
        features_b = second._extract_fusion_features(x, fit_detectors=True)
        assert set(features_a) == set(features_b)
        for group in sorted(features_a):
            delta = float((features_a[group] - features_b[group]).abs().max())
            assert delta == 0.0, f"feature group {group!r} is instance-dependent (max Δ {delta})"
