"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""
from __future__ import annotations

"""Test suite for extended engines"""

import numpy as np

from omni_anomaly_engine.core.extended_anomaly_engine import (
    EngineConfig,
    EvolutionEngine,
    IntegrationEngine,
    OmniAva,
    SecurityEngine,
)


class TestEvolutionEngine:
    def test_initialization(self):
        engine = EvolutionEngine(state_dim=10, population_size=20)

        assert engine.state_dim == 10
        assert engine.population_size == 20
        assert engine.population.shape == (20, 10)

    def test_evolve_generation(self):
        engine = EvolutionEngine(state_dim=5, population_size=10)

        def fitness_fn(x):
            return -np.sum(x**2)

        result = engine.evolve_generation(fitness_fn)

        assert "generation" in result
        assert "best_fitness" in result
        assert "mean_fitness" in result


class TestSecurityEngine:
    def test_initialization(self):
        engine = SecurityEngine()
        assert len(engine.threat_patterns) > 0

    def test_detect_sql_injection(self):
        engine = SecurityEngine()

        malicious_input = "admin' OR '1'='1"
        result = engine.detect_threats(malicious_input)

        assert "is_threat" in result
        assert "threats" in result

    def test_rate_limiting(self):
        engine = SecurityEngine()
        engine.rate_limit_max = 5

        identifier = "test_user"

        for i in range(5):
            assert engine.check_rate_limit(identifier) is True

        assert engine.check_rate_limit(identifier) is False


class TestIntegrationEngine:
    def test_initialization(self):
        engine = IntegrationEngine()
        assert isinstance(engine.integrations, dict)

    def test_register_integration(self):
        engine = IntegrationEngine()

        engine.register_integration(
            integration_id="test_api",
            endpoint_url="https://api.example.com",
            auth_type="api_key",
            rate_limit=1000,
        )

        assert "test_api" in engine.integrations


class TestOmniAva:
    def test_initialization(self):
        config = EngineConfig(enable_3r_mechanism=True, enable_security=True)
        engine = OmniAva(config)

        assert engine.config is not None
        assert engine.three_r is not None
        assert engine.security_engine is not None

    def test_initialization_with_disabled_features(self):
        config = EngineConfig(
            enable_3r_mechanism=False, enable_evolution=False, enable_security=False
        )
        engine = OmniAva(config)

        assert engine.three_r is None
        assert engine.evolution_engine is None
        assert engine.security_engine is None

    def test_detect_anomaly(self):
        config = EngineConfig(enable_3r_mechanism=False)
        engine = OmniAva(config)
        data = np.random.randn(50)

        result = engine.detect_anomaly(data, use_3r_enhancement=False)

        assert "is_anomaly" in result
        assert "anomaly_score" in result

    def test_detect_anomaly_with_3r(self):
        config = EngineConfig(enable_3r_mechanism=True)
        engine = OmniAva(config)
        data = np.random.randn(50)

        result = engine.detect_anomaly(data, use_3r_enhancement=True)

        assert "is_anomaly" in result
        assert "3r_applied" in result
        assert result["3r_applied"] is True

    def test_detect_anomaly_with_resonance_analysis(self):
        config = EngineConfig(enable_3r_mechanism=True)
        engine = OmniAva(config)
        data = np.random.randn(100)

        result = engine.detect_anomaly(data, use_3r_enhancement=True)

        assert "resonance_analysis" in result

    def test_detect_obvious_anomaly(self):
        config = EngineConfig(enable_3r_mechanism=False)
        engine = OmniAva(config)

        data = np.ones(100)
        data[50] = 100.0

        result = engine.detect_anomaly(data, use_3r_enhancement=False)

        assert result["anomaly_score"] > 0.5

    def test_detect_anomaly_empty_data(self):
        config = EngineConfig(enable_3r_mechanism=False)
        engine = OmniAva(config)
        data = np.array([])

        result = engine.detect_anomaly(data, use_3r_enhancement=False)

        assert result["anomaly_score"] == 0.0

    def test_detect_anomaly_constant_data(self):
        config = EngineConfig(enable_3r_mechanism=False)
        engine = OmniAva(config)
        data = np.ones(50)

        result = engine.detect_anomaly(data, use_3r_enhancement=False)

        assert result["anomaly_score"] == 0.0

    def test_validate_input_security(self):
        engine = OmniAva()

        result = engine.validate_input_security("normal input")

        assert "is_threat" in result or "secure" in result

    def test_validate_input_security_disabled(self):
        config = EngineConfig(enable_security=False)
        engine = OmniAva(config)

        result = engine.validate_input_security("any input")

        assert "secure" in result

    def test_evolve_detector(self):
        config = EngineConfig(enable_evolution=True)
        engine = OmniAva(config)

        def fitness_fn(x):
            return -np.sum(x**2)

        result = engine.evolve_detector(fitness_fn, num_generations=5)

        assert "num_generations" in result
        assert "final_best_fitness" in result
        assert "best_solution" in result

    def test_evolve_detector_disabled(self):
        config = EngineConfig(enable_evolution=False)
        engine = OmniAva(config)

        def fitness_fn(x):
            return -np.sum(x**2)

        result = engine.evolve_detector(fitness_fn, num_generations=5)

        assert "error" in result
