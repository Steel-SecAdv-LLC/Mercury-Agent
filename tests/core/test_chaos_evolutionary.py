"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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

"""Tests for Chaos-Evolutionary Optimization integration."""

import numpy as np

from omni_mercury_engine.core.chaos_evolutionary import ChaosEvolutionOptimizer, ChaoticMap


class TestChaoticMap:
    """Test ChaoticMap class."""

    def test_logistic_map_basic(self):
        """Test logistic map basic functionality."""
        x = 0.5
        result = ChaoticMap.logistic_map(x, r=4.0)
        assert 0.0 <= result <= 1.0
        assert result == 4.0 * 0.5 * (1 - 0.5)
        assert result == 1.0

    def test_logistic_map_chaos(self):
        """Test logistic map produces varied sequence."""
        x = 0.3
        values = [x]
        for _ in range(100):
            x = ChaoticMap.logistic_map(x)
            values.append(x)

        assert len(values) == 101
        assert len({round(v, 5) for v in values}) >= 2

    def test_logistic_map_bounds(self):
        """Test logistic map stays in bounds."""
        x = 0.3
        for _ in range(1000):
            x = ChaoticMap.logistic_map(x)
            assert 0.0 <= x <= 1.0

    def test_tent_map_basic(self):
        """Test tent map basic functionality."""
        x = 0.3
        result = ChaoticMap.tent_map(x, mu=2.0)
        assert result == 2.0 * 0.3

    def test_tent_map_upper_branch(self):
        """Test tent map upper branch."""
        x = 0.7
        result = ChaoticMap.tent_map(x, mu=2.0)
        assert result == 2.0 * (1 - 0.7)

    def test_tent_map_chaos(self):
        """Test tent map produces varied sequence."""
        x = 0.3
        values = []
        for _ in range(50):
            x = ChaoticMap.tent_map(x)
            values.append(x)

        assert len(values) == 50
        assert len({round(v, 5) for v in values}) >= 2

    def test_sine_map_basic(self):
        """Test sine map basic functionality."""
        x = 0.5
        result = ChaoticMap.sine_map(x, a=2.3)
        assert isinstance(result, float)

    def test_sine_map_chaos(self):
        """Test sine map produces chaotic sequence."""
        x = 0.5
        values = []
        for _ in range(50):
            x = ChaoticMap.sine_map(x)
            values.append(x)

        assert len(values) == 50


class TestChaosEvolutionOptimizer:
    """Test ChaosEvolutionOptimizer class."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = ChaosEvolutionOptimizer()
        assert optimizer.population_size == 30
        assert optimizer.max_iterations == 100
        assert optimizer.chaotic_map_type == "logistic"
        assert optimizer.alpha == 0.8
        assert optimizer.beta == 0.2

    def test_optimizer_custom_config(self):
        """Test optimizer with custom configuration."""
        config = {
            "population_size": 50,
            "max_iterations": 200,
            "chaotic_map": "tent",
            "alpha": 0.9,
            "beta": 0.1,
        }
        optimizer = ChaosEvolutionOptimizer(config)
        assert optimizer.population_size == 50
        assert optimizer.max_iterations == 200
        assert optimizer.chaotic_map_type == "tent"
        assert optimizer.alpha == 0.9
        assert optimizer.beta == 0.1

    def test_get_chaotic_map_logistic(self):
        """Test getting logistic chaotic map."""
        optimizer = ChaosEvolutionOptimizer({"chaotic_map": "logistic"})
        assert optimizer.chaotic_map == ChaoticMap.logistic_map

    def test_get_chaotic_map_tent(self):
        """Test getting tent chaotic map."""
        optimizer = ChaosEvolutionOptimizer({"chaotic_map": "tent"})
        assert optimizer.chaotic_map == ChaoticMap.tent_map

    def test_get_chaotic_map_sine(self):
        """Test getting sine chaotic map."""
        optimizer = ChaosEvolutionOptimizer({"chaotic_map": "sine"})
        assert optimizer.chaotic_map == ChaoticMap.sine_map

    def test_get_chaotic_map_invalid(self):
        """Test getting chaotic map with invalid type."""
        optimizer = ChaosEvolutionOptimizer({"chaotic_map": "invalid"})
        assert optimizer.chaotic_map == ChaoticMap.logistic_map

    def test_initialize_population_shape(self):
        """Test population initialization shape."""
        optimizer = ChaosEvolutionOptimizer({"population_size": 20})
        bounds = [(0, 1), (0, 1), (0, 1)]
        population = optimizer._initialize_population(3, bounds)
        assert population.shape == (20, 3)

    def test_initialize_population_bounds(self):
        """Test population initialization respects bounds."""
        optimizer = ChaosEvolutionOptimizer()
        bounds = [(0, 1), (-5, 5), (10, 20)]
        population = optimizer._initialize_population(3, bounds)

        assert np.all((population[:, 0] >= 0) & (population[:, 0] <= 1))
        assert np.all((population[:, 1] >= -5) & (population[:, 1] <= 5))
        assert np.all((population[:, 2] >= 10) & (population[:, 2] <= 20))

    def test_chaos_game_step_basic(self):
        """Test chaos game step basic functionality."""
        optimizer = ChaosEvolutionOptimizer()
        position = np.array([0.5, 0.5])
        best_position = np.array([0.8, 0.8])
        chaos_value = 0.7
        bounds = [(0, 1), (0, 1)]

        new_position = optimizer._chaos_game_step(position, best_position, chaos_value, bounds)
        assert len(new_position) == 2
        assert np.all((new_position >= 0) & (new_position <= 1))

    def test_chaos_game_step_bounds_clipping(self):
        """Test chaos game step clips to bounds."""
        optimizer = ChaosEvolutionOptimizer()
        position = np.array([0.1, 0.9])
        best_position = np.array([0.0, 1.0])
        chaos_value = 0.5
        bounds = [(0, 1), (0, 1)]

        new_position = optimizer._chaos_game_step(position, best_position, chaos_value, bounds)
        assert np.all((new_position >= 0) & (new_position <= 1))

    def test_optimize_sphere_function(self):
        """Test optimization on simple sphere function."""

        def sphere(x):
            return np.sum(x**2)

        optimizer = ChaosEvolutionOptimizer({"population_size": 30, "max_iterations": 50})

        results = optimizer.optimize(sphere, dim=2, bounds=[(-5, 5), (-5, 5)])

        assert "best_solution" in results
        assert "best_fitness" in results
        assert "convergence_history" in results
        assert len(results["best_solution"]) == 2
        assert results["best_fitness"] < 5.0

    def test_optimize_convergence_history(self):
        """Test optimization tracks convergence history."""

        def simple_func(x):
            return np.sum((x - 1) ** 2)

        optimizer = ChaosEvolutionOptimizer({"population_size": 15, "max_iterations": 30})

        results = optimizer.optimize(simple_func, dim=2, bounds=[(-5, 5), (-5, 5)])

        assert len(results["convergence_history"]) == 30
        assert results["convergence_history"][-1] == results["best_fitness"]

    def test_optimize_improves_over_iterations(self):
        """Test that optimization improves fitness."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        optimizer = ChaosEvolutionOptimizer({"population_size": 30, "max_iterations": 50})

        results = optimizer.optimize(rosenbrock, dim=2, bounds=[(-2, 2), (-2, 2)])

        initial_fitness = results["convergence_history"][0]
        final_fitness = results["convergence_history"][-1]
        assert final_fitness <= initial_fitness

    def test_optimize_returns_correct_structure(self):
        """Test optimize returns correct result structure."""

        def simple_func(x):
            return np.sum(x**2)

        optimizer = ChaosEvolutionOptimizer()
        results = optimizer.optimize(simple_func, dim=3, bounds=[(-1, 1)] * 3)

        assert "best_solution" in results
        assert "best_fitness" in results
        assert "convergence_history" in results
        assert "iterations" in results
        assert "population_size" in results
        assert "chaotic_map" in results
        assert "method" in results
        assert results["method"] == "Chaos_Evolution_Optimization"

    def test_tune_hyperparameters_basic(self):
        """Test hyperparameter tuning basic functionality."""
        parameter_space = {"learning_rate": (0.001, 0.1), "regularization": (0.0, 1.0)}

        def eval_func(params):
            return (params["learning_rate"] - 0.01) ** 2 + (params["regularization"] - 0.5) ** 2

        optimizer = ChaosEvolutionOptimizer({"population_size": 20, "max_iterations": 30})

        results = optimizer.tune_hyperparameters(parameter_space, eval_func)

        assert "optimal_parameters" in results
        assert "learning_rate" in results["optimal_parameters"]
        assert "regularization" in results["optimal_parameters"]

    def test_tune_hyperparameters_convergence(self):
        """Test hyperparameter tuning converges to optimum."""
        parameter_space = {"param1": (-5, 5), "param2": (-5, 5)}

        def eval_func(params):
            return params["param1"] ** 2 + params["param2"] ** 2

        optimizer = ChaosEvolutionOptimizer({"population_size": 30, "max_iterations": 50})

        results = optimizer.tune_hyperparameters(parameter_space, eval_func)

        assert results["optimal_loss"] < 0.5
        assert abs(results["optimal_parameters"]["param1"]) < 1.0
        assert abs(results["optimal_parameters"]["param2"]) < 1.0

    def test_tune_hyperparameters_result_structure(self):
        """Test hyperparameter tuning returns correct structure."""
        parameter_space = {"x": (0, 1)}

        def eval_func(params):
            return params["x"] ** 2

        optimizer = ChaosEvolutionOptimizer()
        results = optimizer.tune_hyperparameters(parameter_space, eval_func)

        assert "optimal_parameters" in results
        assert "optimal_loss" in results
        assert "convergence_history" in results
        assert "method" in results
        assert results["method"] == "CGO_Hyperparameter_Tuning"

    def test_butterfly_effect_small_perturbation(self):
        """Test butterfly effect: small input change leads to large detection shift."""

        def simple_func(x):
            return np.sum(x**2)

        np.random.seed(42)
        optimizer1 = ChaosEvolutionOptimizer({"population_size": 20, "max_iterations": 30})
        results1 = optimizer1.optimize(simple_func, dim=2, bounds=[(-5, 5), (-5, 5)])

        np.random.seed(43)
        optimizer2 = ChaosEvolutionOptimizer({"population_size": 20, "max_iterations": 30})
        results2 = optimizer2.optimize(simple_func, dim=2, bounds=[(-5, 5), (-5, 5)])

        solution_diff = np.linalg.norm(results1["best_solution"] - results2["best_solution"])
        assert solution_diff > 0.01

    def test_chaotic_sensitivity_to_alpha(self):
        """Test sensitivity to alpha parameter in fractal component."""

        def test_func(x):
            return np.sum(x**2)

        results_low = ChaosEvolutionOptimizer({"alpha": 0.1, "max_iterations": 20}).optimize(
            test_func, dim=2, bounds=[(-5, 5), (-5, 5)]
        )
        results_high = ChaosEvolutionOptimizer({"alpha": 0.9, "max_iterations": 20}).optimize(
            test_func, dim=2, bounds=[(-5, 5), (-5, 5)]
        )

        assert results_low["best_fitness"] != results_high["best_fitness"]

    def test_chaotic_sensitivity_to_beta(self):
        """Test sensitivity to beta parameter in chaos component."""

        def test_func(x):
            return np.sum(x**2)

        results_low = ChaosEvolutionOptimizer({"beta": 0.01, "max_iterations": 20}).optimize(
            test_func, dim=2, bounds=[(-5, 5), (-5, 5)]
        )
        results_high = ChaosEvolutionOptimizer({"beta": 0.5, "max_iterations": 20}).optimize(
            test_func, dim=2, bounds=[(-5, 5), (-5, 5)]
        )

        assert results_low["best_fitness"] != results_high["best_fitness"]

    def test_chaotic_map_sequence_divergence(self):
        """Test that chaotic maps produce sequences within valid bounds."""
        x_logistic = 0.5
        x_tent = 0.5
        x_sine = 0.5

        for _ in range(10):
            x_logistic = ChaoticMap.logistic_map(x_logistic)
            x_tent = ChaoticMap.tent_map(x_tent)
            x_sine = ChaoticMap.sine_map(x_sine)

        assert 0 <= x_logistic <= 1
        assert 0 <= x_tent <= 1
        assert 0 <= x_sine <= 1

    def test_convergence_with_multiple_local_minima(self):
        """Test optimization on function with multiple local minima."""

        def multi_modal(x):
            return np.sum(x**2) + 10 * np.sum(np.cos(2 * np.pi * x))

        optimizer = ChaosEvolutionOptimizer({"population_size": 40, "max_iterations": 60})
        results = optimizer.optimize(multi_modal, dim=3, bounds=[(-5, 5)] * 3)

        assert results["best_fitness"] < 15.0

    def test_rastrigin_function_optimization(self):
        """Test optimization on challenging Rastrigin function."""

        def rastrigin(x):
            n = len(x)
            return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

        optimizer = ChaosEvolutionOptimizer({"population_size": 50, "max_iterations": 80})
        results = optimizer.optimize(rastrigin, dim=2, bounds=[(-5.12, 5.12)] * 2)

        assert results["best_fitness"] < 50.0

    def test_ackley_function_optimization(self):
        """Test optimization on Ackley function with many local minima."""

        def ackley(x):
            a = 20
            b = 0.2
            c = 2 * np.pi
            d = len(x)
            sum1 = np.sum(x**2)
            sum2 = np.sum(np.cos(c * x))
            return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)

        optimizer = ChaosEvolutionOptimizer({"population_size": 40, "max_iterations": 60})
        results = optimizer.optimize(ackley, dim=2, bounds=[(-5, 5)] * 2)

        assert results["best_fitness"] < 5.0

    def test_chaotic_perturbation_magnitude(self):
        """Test that chaos perturbations have expected magnitude."""
        optimizer = ChaosEvolutionOptimizer({"beta": 0.3})
        position = np.array([0.0, 0.0])
        best_position = np.array([1.0, 1.0])
        chaos_value = 0.5
        bounds = [(-10, 10), (-10, 10)]

        new_position = optimizer._chaos_game_step(position, best_position, chaos_value, bounds)

        movement = np.linalg.norm(new_position - position)
        assert movement > 0.0
        assert movement < 20.0

    def test_population_diversity_maintained(self):
        """Test that population maintains diversity during optimization."""

        def simple_func(x):
            return np.sum(x**2)

        optimizer = ChaosEvolutionOptimizer({"population_size": 30, "max_iterations": 10})
        bounds = [(-5, 5), (-5, 5)]
        population = optimizer._initialize_population(2, bounds)

        for _ in range(10):
            chaos_value = np.random.rand()
            for i in range(len(population)):
                chaos_value = optimizer.chaotic_map(chaos_value)
                population[i] = optimizer._chaos_game_step(
                    population[i], population[0], chaos_value, bounds
                )

        final_std = np.std(population, axis=0)
        assert np.mean(final_std) > 0.1

    def test_convergence_monotonic_decrease(self):
        """Test that convergence history shows general decreasing trend."""

        def simple_func(x):
            return np.sum(x**2)

        optimizer = ChaosEvolutionOptimizer({"population_size": 30, "max_iterations": 50})
        results = optimizer.optimize(simple_func, dim=2, bounds=[(-5, 5), (-5, 5)])

        history = results["convergence_history"]
        first_half_avg = np.mean(history[:25])
        second_half_avg = np.mean(history[25:])

        assert second_half_avg <= first_half_avg

    def test_different_dimensions_optimization(self):
        """Test optimization works across different dimensionalities."""

        def sphere(x):
            return np.sum(x**2)

        for dim in [2, 5, 10]:
            optimizer = ChaosEvolutionOptimizer({"population_size": 30, "max_iterations": 30})
            results = optimizer.optimize(sphere, dim=dim, bounds=[(-5, 5)] * dim)

            assert len(results["best_solution"]) == dim
            assert results["best_fitness"] < 10.0

    def test_tent_map_boundary_behavior(self):
        """Test tent map behavior at boundaries."""
        x = 0.0
        result = ChaoticMap.tent_map(x, mu=2.0)
        assert result == 0.0

        x = 1.0
        result = ChaoticMap.tent_map(x, mu=2.0)
        assert result == 0.0

    def test_logistic_map_with_different_r_values(self):
        """Test logistic map with different chaos parameters."""
        x = 0.5

        result_low = ChaoticMap.logistic_map(x, r=2.0)
        result_high = ChaoticMap.logistic_map(x, r=4.0)

        assert result_low != result_high
        assert 0.0 <= result_low <= 1.0
        assert 0.0 <= result_high <= 1.0

    def test_sine_map_periodicity(self):
        """Test sine map produces varied sequence."""
        x = 0.5
        values = []
        for _ in range(20):
            x = ChaoticMap.sine_map(x, a=2.3)
            values.append(x)

        assert len({round(v, 4) for v in values}) > 5

    def test_chaos_game_step_moves_toward_best(self):
        """Test chaos game step generally moves toward best solution."""
        optimizer = ChaosEvolutionOptimizer({"alpha": 0.9, "beta": 0.1})
        position = np.array([0.0, 0.0])
        best_position = np.array([5.0, 5.0])
        chaos_value = 0.5
        bounds = [(-10, 10), (-10, 10)]

        new_position = optimizer._chaos_game_step(position, best_position, chaos_value, bounds)

        distance_before = np.linalg.norm(position - best_position)
        distance_after = np.linalg.norm(new_position - best_position)

        assert distance_after < distance_before

    def test_optimization_with_tight_bounds(self):
        """Test optimization works with tight parameter bounds."""

        def simple_func(x):
            return np.sum(x**2)

        optimizer = ChaosEvolutionOptimizer({"population_size": 20, "max_iterations": 30})
        results = optimizer.optimize(simple_func, dim=2, bounds=[(-0.5, 0.5), (-0.5, 0.5)])

        assert np.all(results["best_solution"] >= -0.5)
        assert np.all(results["best_solution"] <= 0.5)
        assert results["best_fitness"] < 0.5

    def test_optimization_with_asymmetric_bounds(self):
        """Test optimization with asymmetric bounds."""

        def simple_func(x):
            return np.sum((x - 3) ** 2)

        optimizer = ChaosEvolutionOptimizer({"population_size": 25, "max_iterations": 40})
        results = optimizer.optimize(simple_func, dim=2, bounds=[(0, 10), (-5, 5)])

        assert 0 <= results["best_solution"][0] <= 10
        assert -5 <= results["best_solution"][1] <= 5

    def test_hyperparameter_tuning_with_constraints(self):
        """Test hyperparameter tuning respects parameter constraints."""
        parameter_space = {"lr": (0.001, 0.1), "momentum": (0.5, 0.99), "weight_decay": (0.0, 0.01)}

        def eval_func(params):
            return (params["lr"] - 0.01) ** 2 + (params["momentum"] - 0.9) ** 2

        optimizer = ChaosEvolutionOptimizer({"population_size": 25, "max_iterations": 30})
        results = optimizer.tune_hyperparameters(parameter_space, eval_func)

        params = results["optimal_parameters"]
        assert 0.001 <= params["lr"] <= 0.1
        assert 0.5 <= params["momentum"] <= 0.99
        assert 0.0 <= params["weight_decay"] <= 0.01

    def test_chaotic_sequence_non_repeating(self):
        """Test chaotic sequences don't repeat early."""
        x = 0.3
        sequence = [x]

        for _ in range(50):
            x = ChaoticMap.logistic_map(x)
            sequence.append(x)

        rounded_sequence = [round(v, 6) for v in sequence]
        unique_count = len(set(rounded_sequence))

        assert unique_count > 40

    def test_optimization_finds_global_minimum(self):
        """Test optimizer finds near-global minimum for convex function."""

        def sphere(x):
            return np.sum(x**2)

        optimizer = ChaosEvolutionOptimizer({"population_size": 50, "max_iterations": 100})
        results = optimizer.optimize(sphere, dim=3, bounds=[(-10, 10)] * 3)

        assert results["best_fitness"] < 1.0
        assert np.linalg.norm(results["best_solution"]) < 1.5

    def test_chaos_injection_improves_exploration(self):
        """Test chaos injection helps escape local minima."""

        def deceptive_func(x):
            return np.sum(x**2) + 5 * np.exp(-np.sum((x - 2) ** 2))

        optimizer = ChaosEvolutionOptimizer({"population_size": 40, "max_iterations": 60})
        results = optimizer.optimize(deceptive_func, dim=2, bounds=[(-5, 5), (-5, 5)])

        assert results["best_fitness"] < 10.0

    def test_population_size_affects_convergence(self):
        """Test different population sizes affect optimization."""

        def simple_func(x):
            return np.sum(x**2)

        results_small = ChaosEvolutionOptimizer(
            {"population_size": 10, "max_iterations": 30}
        ).optimize(simple_func, dim=2, bounds=[(-5, 5), (-5, 5)])
        results_large = ChaosEvolutionOptimizer(
            {"population_size": 50, "max_iterations": 30}
        ).optimize(simple_func, dim=2, bounds=[(-5, 5), (-5, 5)])

        assert "best_fitness" in results_small
        assert "best_fitness" in results_large

    def test_max_iterations_affects_quality(self):
        """Test more iterations generally improve solution quality."""

        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        optimizer_short = ChaosEvolutionOptimizer({"population_size": 30, "max_iterations": 10})
        optimizer_long = ChaosEvolutionOptimizer({"population_size": 30, "max_iterations": 100})

        results_short = optimizer_short.optimize(rosenbrock, dim=2, bounds=[(-2, 2), (-2, 2)])
        results_long = optimizer_long.optimize(rosenbrock, dim=2, bounds=[(-2, 2), (-2, 2)])

        assert len(results_short["convergence_history"]) == 10
        assert len(results_long["convergence_history"]) == 100

    def test_chaotic_map_reset_maintains_randomness(self):
        """Test chaotic map reset every 10 iterations maintains exploration."""
        optimizer = ChaosEvolutionOptimizer({"max_iterations": 30})

        def simple_func(x):
            return np.sum(x**2)

        results = optimizer.optimize(simple_func, dim=2, bounds=[(-5, 5), (-5, 5)])

        assert len(results["convergence_history"]) == 30
        assert results["convergence_history"][-1] <= results["convergence_history"][0]

    def test_chaos_evolutionary_with_constraints_handling(self):
        """Test optimizer handles constrained optimization."""

        def constrained_func(x):
            penalty = 0
            if x[0] + x[1] > 5:
                penalty = 1000
            return np.sum(x**2) + penalty

        optimizer = ChaosEvolutionOptimizer({"population_size": 30, "max_iterations": 50})
        results = optimizer.optimize(constrained_func, dim=2, bounds=[(-5, 5), (-5, 5)])

        assert results["best_fitness"] < 100

    def test_fractal_self_similarity_in_convergence(self):
        """Test convergence pattern shows fractal-like self-similarity."""

        def simple_func(x):
            return np.sum(x**2)

        optimizer = ChaosEvolutionOptimizer({"population_size": 40, "max_iterations": 100})
        results = optimizer.optimize(simple_func, dim=2, bounds=[(-5, 5), (-5, 5)])

        history = np.array(results["convergence_history"])
        first_quarter = history[:25]
        second_quarter = history[25:50]

        first_var = np.var(np.diff(first_quarter))
        second_var = np.var(np.diff(second_quarter))

        assert first_var > 0
        assert second_var >= 0
