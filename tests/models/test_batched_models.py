# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The batched domain models reproduce the per-row models they replaced.

Each ``_reference_*`` function is the original implementation, one Python
iteration per row, kept here as the oracle. The batched models must return
the same features on every shape and dtype the engine feeds them, including
the degenerate ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.models.astrophysical import AstrophysicalAnomalyModel
from omni_mercury_engine.models.biometric import BiometricAnomalyModel
from omni_mercury_engine.models.consciousness import ConsciousnessPreservationModel
from omni_mercury_engine.models.multiverse import MultiverseOmniEngine, Universe
from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine
from omni_mercury_engine.models.quantum import QuantumAnomalyModel

Array = np.ndarray[Any, Any]


def _data(
    shape: tuple[int, ...], dtype: Any = np.float32, seed: int = 0, scale: float = 1.0
) -> Array:
    return (np.random.default_rng(seed).normal(size=shape) * scale).astype(dtype)


# ---------------------------------------------------------------- quantum


def _reference_entanglement(state: Array) -> float:
    density_matrix = np.outer(state, np.conj(state))
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues + 1e-10)))


class TestQuantum:
    @pytest.mark.parametrize("width", [1, 3, 8, 33, 64])
    @pytest.mark.parametrize("scale", [0.0, 1e-9, 1e-3, 1.0, 1e3])
    def test_entropy_matches_eigendecomposition(self, width: int, scale: float) -> None:
        model = QuantumAnomalyModel()
        states = model._create_quantum_state(_data((30, width), np.float64, width, scale))
        got = model._pure_state_entropy(states)
        want = np.array([_reference_entanglement(s) for s in states])
        # The original used complex64 LAPACK for float32 input; float64 is exact.
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_measure_entanglement_still_answers_for_one_state(self) -> None:
        model = QuantumAnomalyModel()
        for state in model._create_quantum_state(_data((12, 6), np.float64, 3)):
            assert model._measure_entanglement(state) == pytest.approx(
                _reference_entanglement(state), abs=1e-12
            )

    def test_feature_column_12_is_the_schmidt_entropy(self) -> None:
        """Column 12 carries the bipartite Schmidt entropy, checked against SVD by hand."""
        model = QuantumAnomalyModel()
        data = _data((50, 12))
        states = model._create_quantum_state(data)
        features = model.extract_features(data)
        side = 4  # ceil(sqrt(12))
        want = []
        for state in states:
            padded = np.zeros(side * side, dtype=np.complex128)
            padded[:12] = state
            probs = np.linalg.svd(padded.reshape(side, side), compute_uv=False) ** 2
            probs = probs / probs.sum()
            probs = probs[probs > 1e-12]
            want.append(float(-np.sum(probs * np.log2(probs))))
        np.testing.assert_allclose(features[:, 12], want, rtol=0, atol=1e-5)

    def test_quantum_feature_columns_are_not_constant(self) -> None:
        """Regression: columns 12-15 were constants (full pure-state entropy is
        identically 0 on a normalised state; three columns repeated the
        entanglement_strength config value), so 4 of the model's 16 feature
        columns carried nothing into fusion."""
        features = QuantumAnomalyModel().extract_features(_data((80, 12)))
        for column in range(12, 16):
            assert float(np.std(features[:, column])) > 1e-6, column
        # and they are measured, bounded quantities
        assert np.all(features[:, 12] >= 0.0)  # Schmidt entropy, bits
        assert np.all((features[:, 13] > 0.0) & (features[:, 13] <= 1.0 + 1e-6))
        assert np.all((features[:, 14] >= 0.0) & (features[:, 14] <= 1.0 + 1e-6))
        assert np.all((features[:, 15] >= 0.0) & (features[:, 15] <= 1.0 + 1e-6))

    def test_predict_coherence_is_measured_not_constant(self) -> None:
        """Regression: predict()'s coherence was identically 0 (1 - |0 - 1|)."""
        result = QuantumAnomalyModel().fit(_data((80, 12))).predict(_data((80, 12)))
        coherence = np.asarray(result["coherence"])
        assert float(np.std(coherence)) > 1e-6
        assert np.all((coherence >= 0.0) & (coherence <= 1.0 + 1e-6))

    def test_wide_input_is_not_cubic(self) -> None:
        """Width 512 is 134M density-matrix cells per row; this must stay instant."""
        features = QuantumAnomalyModel().extract_features(_data((64, 512)))
        assert features.shape == (64, 16) and bool(np.isfinite(features).all())


# -------------------------------------------------------------- multiverse


def _reference_multiverse(engine: MultiverseOmniEngine, data: Array) -> Array:
    """Original per-row ``extract_features`` (mutates ``engine.best_universe``)."""
    if data.ndim == 1:
        data = data.reshape(1, -1)
    features: list[Array] = []
    for i in range(data.shape[0]):
        working = [
            Universe(
                universe_id=u.universe_id,
                state_vector=u.state_vector.copy(),
                probability_amplitude=u.probability_amplitude,
                fitness=u.fitness,
                state=u.state,
                timeline=u.timeline,
                parent_universe=u.parent_universe,
                metadata=dict(u.metadata),
            )
            for u in engine.universes.values()
        ]

        def fitness_fn(state: Array, i: int = i) -> float:
            data_dim = data[i].shape[0]
            if state.shape[0] > data_dim:
                state_truncated = state[:data_dim]
            elif state.shape[0] < data_dim:
                state_truncated = np.pad(state, (0, data_dim - state.shape[0]))
            else:
                state_truncated = state
            return float(-np.linalg.norm(state_truncated - data[i]))

        best: Universe | None = None
        for universe in working:
            fitness = fitness_fn(universe.state_vector)
            universe.fitness = fitness
            if best is None or fitness > best.fitness:
                best = universe
        total_fitness = sum(u.fitness for u in working)
        if total_fitness > 1e-10:
            for universe in working:
                universe.probability_amplitude = universe.fitness / total_fitness
        ranked = sorted(working, key=lambda u: u.fitness, reverse=True)
        top = ranked[: max(3, engine.num_universes // 3)]
        total_amplitude = sum(u.probability_amplitude for u in top)
        if total_amplitude < 1e-10:
            weights = np.ones(len(top)) / len(top)
        else:
            weights = np.array([u.probability_amplitude / total_amplitude for u in top])
        converged_state = np.zeros(engine.state_dim)
        for j, universe in enumerate(top):
            converged_state += weights[j] * universe.state_vector
        if engine.best_universe is None or (
            best is not None and best.fitness > engine.best_universe.fitness
        ):
            engine.best_universe = best
        features.append(
            np.concatenate([converged_state[:10], [fitness_fn(converged_state)], [total_amplitude]])
        )
    return np.array(features).astype(np.float32)


class TestMultiverse:
    @pytest.mark.parametrize("width", [1, 4, 10, 50, 80])
    @pytest.mark.parametrize("size", [1, 7, 130])
    def test_matches_the_per_row_model(self, width: int, size: int) -> None:
        new = MultiverseOmniEngine(num_universes=10, state_dim=50)
        ref = MultiverseOmniEngine(num_universes=10, state_dim=50)
        data = _data((size, width), seed=width + size)
        np.testing.assert_allclose(
            new.extract_features(data), _reference_multiverse(ref, data), rtol=1e-6, atol=1e-6
        )
        assert new.best_universe is not None and ref.best_universe is not None
        assert new.best_universe.universe_id == ref.best_universe.universe_id
        assert new.best_universe.fitness == pytest.approx(ref.best_universe.fitness, rel=1e-12)

    @pytest.mark.parametrize("population", [2, 3, 5, 7, 12, 31])
    def test_closest_count_follows_the_population_size(self, population: int) -> None:
        new = MultiverseOmniEngine(num_universes=population, state_dim=16)
        ref = MultiverseOmniEngine(num_universes=population, state_dim=16)
        data = _data((25, 9), seed=population)
        np.testing.assert_allclose(
            new.extract_features(data), _reference_multiverse(ref, data), rtol=1e-6, atol=1e-6
        )

    def test_equidistant_universes_keep_their_order(self) -> None:
        """Ties in distance resolve by universe order, as the stable sort did."""
        new = MultiverseOmniEngine(num_universes=8, state_dim=10)
        ref = MultiverseOmniEngine(num_universes=8, state_dim=10)
        for engine in (new, ref):
            universes = list(engine.universes.values())
            universes[5].state_vector = universes[1].state_vector.copy()
            universes[6].state_vector = universes[1].state_vector.copy()
            universes[6].probability_amplitude = 0.4  # distinguishes who ranks first
        data = np.vstack(
            [new.universes[next(iter(new.universes))].state_vector[:10], _data((6, 10))]
        )
        data[1] = list(new.universes.values())[1].state_vector[:10]  # a row exactly on the tie
        np.testing.assert_allclose(
            new.extract_features(data), _reference_multiverse(ref, data), rtol=1e-6, atol=1e-6
        )

    def test_best_universe_only_improves_across_calls(self) -> None:
        new = MultiverseOmniEngine(num_universes=6, state_dim=12)
        ref = MultiverseOmniEngine(num_universes=6, state_dim=12)
        for seed, scale in ((1, 5.0), (2, 0.05), (3, 1.0)):
            data = _data((20, 12), seed=seed, scale=scale)
            new.extract_features(data)
            _reference_multiverse(ref, data)
            assert new.best_universe is not None and ref.best_universe is not None
            assert new.best_universe.fitness == pytest.approx(ref.best_universe.fitness, rel=1e-12)

    def test_does_not_mutate_the_population(self) -> None:
        engine = MultiverseOmniEngine(num_universes=5, state_dim=10)
        before = [(u.state_vector.copy(), u.fitness) for u in engine.universes.values()]
        engine.extract_features(_data((9, 10)))
        for universe, (vector, fitness) in zip(engine.universes.values(), before, strict=True):
            np.testing.assert_array_equal(universe.state_vector, vector)
            assert universe.fitness == fitness

    def test_empty_batch(self) -> None:
        engine = MultiverseOmniEngine(num_universes=5, state_dim=10)
        assert engine.extract_features(np.zeros((0, 10))).shape == (0, 12)


# ----------------------------------------------------------- neurosymbolic


def _reference_neural_inference(features: Array) -> float:
    arr = np.asarray(features, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return 0.5
    centre = float(np.median(arr))
    mad = float(np.median(np.abs(arr - centre)))
    scale = mad if mad > 1e-9 else float(np.std(arr) + 1e-9)
    dispersion = float(np.mean(np.abs(arr - float(np.mean(arr)))) / scale)
    return float(1.0 / (1.0 + np.exp(-(dispersion - 1.0))))


def _reference_neurosymbolic(engine: NeurosymbolicEngine, data: Array) -> Array:
    if data.ndim == 1:
        data = data.reshape(1, -1)
    rows = []
    for i in range(data.shape[0]):
        rows.append(
            np.concatenate(
                [
                    (
                        data[i][:10]
                        if data.shape[1] >= 10
                        else np.pad(data[i], (0, 10 - data.shape[1]))
                    ),
                    [_reference_neural_inference(data[i])],
                    [engine.omni_scalars["omni_logic"]],
                ]
            )
        )
    return np.array(rows).astype(np.float32)


class TestNeurosymbolic:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
    @pytest.mark.parametrize("width", [1, 2, 9, 10, 11, 64])
    def test_matches_the_per_row_model(self, dtype: Any, width: int) -> None:
        engine = NeurosymbolicEngine(input_dim=64)
        data = _data((40, width), dtype, seed=width, scale=7.0)
        np.testing.assert_allclose(
            engine.extract_features(data),
            _reference_neurosymbolic(engine, data),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_degenerate_rows(self) -> None:
        engine = NeurosymbolicEngine(input_dim=64)
        data = np.array(
            [
                [3.0] * 12,  # constant: MAD is zero, falls back to the std
                [0.0] * 12,
                [1.0] + [0.0] * 11,  # MAD zero, std nonzero
                [np.nan] + [1.0] * 11,  # non-finite: neutral 0.5
                [np.inf] + [1.0] * 11,
                [1e12, -1e12] + [0.0] * 10,
                list(np.linspace(-1e-5, 1e-5, 12)),  # MAD far below 1e-3 but above 1e-9
                list(np.linspace(-1e-7, 1e-7, 12)),
                list(np.linspace(-1e-10, 1e-10, 12)),  # MAD below the 1e-9 floor
            ]
        )
        np.testing.assert_allclose(
            engine.extract_features(data),
            _reference_neurosymbolic(engine, data),
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        )

    def test_neural_inference_still_scores_one_flattened_array(self) -> None:
        engine = NeurosymbolicEngine(input_dim=64)
        for array in (
            _data((1, 20)),
            _data((4, 5)),
            _data((20,)),
            np.array([]),
            np.array([np.nan]),
        ):
            assert engine.neural_inference(array) == pytest.approx(
                _reference_neural_inference(array), rel=1e-9
            )


# ------------------------------------------------------------ consciousness


def _reference_encode(data: Array) -> Array:
    batch_size, _dim = data.shape
    pattern_states = np.zeros((batch_size, 16), dtype=np.complex64)
    for i in range(batch_size):
        pattern = data[i]
        norm = np.linalg.norm(pattern)
        normalized = pattern / norm if norm > 0 else pattern
        fft_result = np.fft.fft(normalized)
        pattern_states[i, : min(16, len(fft_result))] = fft_result[: min(16, len(fft_result))]
    return pattern_states


def _reference_coherence(pattern_states: Array) -> Array:
    coherence = np.zeros(pattern_states.shape[0], dtype=np.float32)
    for i in range(pattern_states.shape[0]):
        amplitudes = np.abs(pattern_states[i])
        total_amplitude = np.sum(amplitudes)
        if total_amplitude > 0:
            probabilities = amplitudes / total_amplitude
            coherence[i] = 1.0 - np.sum(probabilities**2)
    return coherence


def _reference_consciousness_entanglement(pattern_states: Array) -> Array:
    entanglement = np.zeros(pattern_states.shape[0], dtype=np.float32)
    for i in range(pattern_states.shape[0]):
        state = pattern_states[i]
        eigenvalues = np.linalg.eigvalsh(np.outer(state, np.conj(state)).real)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) > 0:
            probabilities = eigenvalues / np.sum(eigenvalues)
            entanglement[i] = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entanglement


class TestConsciousness:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
    @pytest.mark.parametrize("width", [1, 5, 16, 40])
    def test_matches_the_per_row_model(self, dtype: Any, width: int) -> None:
        model = ConsciousnessPreservationModel()
        data = _data((60, width), dtype, seed=width, scale=9.0)
        data[3] = 0  # an all-zero row exercises the zero-norm branches
        states = _reference_encode(data)
        np.testing.assert_allclose(model._encode_pattern_states(data), states, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            model._measure_pattern_coherence(states),
            _reference_coherence(states),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            model._compute_entanglement(states),
            _reference_consciousness_entanglement(states),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_large_batch_is_blocked_not_truncated(self) -> None:
        from omni_mercury_engine.models import consciousness

        model = ConsciousnessPreservationModel()
        states = _reference_encode(_data((50, 8)))
        whole = model._compute_entanglement(states)
        original = consciousness._ENTANGLEMENT_BLOCK_ROWS
        consciousness._ENTANGLEMENT_BLOCK_ROWS = 7
        try:
            blocked = model._compute_entanglement(states)
        finally:
            consciousness._ENTANGLEMENT_BLOCK_ROWS = original
        np.testing.assert_array_equal(whole, blocked)


# ------------------------------------------------------------ astrophysical


def _reference_astrophysical(model: AstrophysicalAnomalyModel, data: Array) -> Array:
    rs = model.schwarzschild_radius
    features = []
    for point in data:
        distance = float(np.linalg.norm(point))
        grav = (
            1e6
            if distance < 1e-6
            else float(model.gravitational_constant * model.mass_equivalent / (distance**2))
        )
        dilation = 0.0 if distance <= rs else float(np.sqrt(1 - rs / distance))
        if distance > 10 * rs:
            horizon = 0.0
        elif distance > 2 * rs:
            horizon = 0.33
        elif distance > rs:
            horizon = 0.66
        else:
            horizon = 1.0
        info_density = np.std(point) / (np.mean(np.abs(point)) + 1e-8)
        accretion = np.sum(np.abs(point)) / len(point)
        row = np.array(
            [
                distance,
                grav,
                dilation,
                horizon,
                1.0 / (8 * np.pi * model.mass_equivalent),
                info_density,
                accretion,
                rs,
            ]
        )
        features.append(np.concatenate([row, np.zeros(16)]))
    return np.array(features).astype(np.float32)


class TestAstrophysical:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
    @pytest.mark.parametrize("width", [1, 3, 12, 64])
    def test_matches_the_per_row_model(self, dtype: Any, width: int) -> None:
        model = AstrophysicalAnomalyModel()
        # Scales that put rows inside, between and far outside the horizon radii.
        data = np.concatenate(
            [_data((20, width), dtype, width, s) for s in (0.001, 0.4, 1.0, 20.0)]
        )
        data[0] = 0
        np.testing.assert_allclose(
            model.extract_features(data),
            _reference_astrophysical(model, data),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_heavier_object_widens_the_horizon(self) -> None:
        model = AstrophysicalAnomalyModel({"mass_equivalent": 5.0, "gravitational_constant": 2.0})
        data = _data((30, 6), seed=4, scale=3.0)
        np.testing.assert_allclose(
            model.extract_features(data),
            _reference_astrophysical(model, data),
            rtol=1e-5,
            atol=1e-6,
        )


# ---------------------------------------------------------------- biometric


def _reference_harmonic(model: BiometricAnomalyModel, data: Array) -> Array:
    harmonic = model.harmonic_decomposer.decompose(data)
    fourier = model.fourier_analyzer.analyze(data)
    power_spectrum, phase = fourier["power_spectrum"], fourier["phase"]
    rows = []
    for i in range(data.shape[0]):

        def lead(block: Array, i: int = i) -> Array:
            return (
                block[i, :32]
                if block.shape[1] >= 32
                else np.pad(block[i], (0, 32 - block.shape[1]))
            )

        sample = np.concatenate(
            [
                lead(harmonic),
                lead(power_spectrum),
                lead(phase),
                np.array(
                    [
                        np.mean(harmonic[i]),
                        np.std(harmonic[i]),
                        np.max(harmonic[i]),
                        np.min(harmonic[i]),
                    ]
                ),
                np.array(
                    [
                        np.mean(power_spectrum[i]),
                        np.std(power_spectrum[i]),
                        np.max(power_spectrum[i]),
                        np.min(power_spectrum[i]),
                    ]
                ),
                np.zeros(24),
            ]
        )
        rows.append(sample[: model.target_embedding_size])
    return np.array(rows, dtype=np.float32)


class TestBiometricHarmonic:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint8])
    @pytest.mark.parametrize("width", [1, 8, 32, 33, 200])
    def test_matches_the_per_row_model(self, dtype: Any, width: int) -> None:
        model = BiometricAnomalyModel()
        data = (np.abs(_data((25, width), np.float64, width, 100.0))).astype(dtype)
        np.testing.assert_allclose(
            model._extract_harmonic_features(data),
            _reference_harmonic(model, data),
            rtol=1e-5,
            atol=1e-4,
        )

    def test_three_dimensional_input_is_flattened_per_sample(self) -> None:
        model = BiometricAnomalyModel()
        data = _data((4, 6, 5))
        got = model._extract_harmonic_features(data)
        np.testing.assert_allclose(
            got, _reference_harmonic(model, data.reshape(4, -1)), rtol=1e-5, atol=1e-4
        )

    def test_disabled_harmonic_features_are_zero(self) -> None:
        model = BiometricAnomalyModel({"use_harmonic_features": False})
        assert not model._extract_harmonic_features(_data((3, 8))).any()


class TestReasonMany:
    """Rules see only the context, so one evaluation serves every score."""

    def test_matches_reasoning_each_score_alone(self) -> None:
        from omni_mercury_engine.models.neurosymbolic import SymbolicReasoningLayer, SymbolicRule

        layer = SymbolicReasoningLayer(explainability_threshold=0.5)
        for name, premise, confidence in (
            ("deviant", "deviation_score > 2.0", 0.8),
            ("quiet", "not alarm", 0.9),
            ("weak", "never_set", 0.3),
        ):
            layer.add_rule(
                SymbolicRule(
                    premise=premise,
                    conclusion=name,
                    confidence=confidence,
                    name=name,
                    explanation_template="{premise}",
                )
            )
        context = {"deviation_score": 3.1}
        scores = [0.0, 0.2, 0.5, 0.93, 1.0]

        batched = layer.reason_many(scores, context)
        for score, got in zip(scores, batched, strict=True):
            assert got == layer.reason(score, context)
        assert batched[0].rules_fired == ["deviant", "quiet"]
        batched[0].rules_fired.append("mutated")
        assert batched[1].rules_fired == ["deviant", "quiet"]  # lists are not shared

    def test_empty_scores(self) -> None:
        from omni_mercury_engine.models.neurosymbolic import SymbolicReasoningLayer

        assert SymbolicReasoningLayer().reason_many([], {}) == []

    def test_predict_scores_match_per_row_reasoning(self) -> None:
        engine = NeurosymbolicEngine(input_dim=64)
        data = _data((30, 12), seed=21)
        result = engine.predict(data, context={"deviation_score": 3.0, "frequency": 20})
        neural = result["neural_confidence"]
        want = [
            engine.symbolic_layer.reason(float(score), {"deviation_score": 3.0, "frequency": 20})
            for score in neural
        ]
        np.testing.assert_array_equal(
            result["anomaly_scores"], np.array([r.confidence for r in want], dtype=np.float32)
        )
        assert result["symbolic_conclusions"]["rules_fired"] == [r.rules_fired for r in want]
