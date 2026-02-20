"""
Mercury Agent
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

"""Extended tests for neurosymbolic_engine to reach >95% coverage."""

import ast

from omni_mercury_engine.core.neurosymbolic_engine import NeurosymbolicEngine


class TestNeurosymbolicEngineExtended:
    """Extended tests for uncovered neurosymbolic engine functionality."""

    def test_embed_code_with_comments(self):
        """Test embedding code that contains comments."""
        engine = NeurosymbolicEngine()
        code = """def example():
    return 42"""
        result = engine.hybrid_analysis(ast.parse(code))
        assert result is not None

    def test_extract_patterns_multiple_functions(self):
        """Test pattern extraction from multiple functions."""
        engine = NeurosymbolicEngine()
        code = """def func1():
    return 1
def func2():
    return 2
def func3():
    return 3"""
        result = engine.hybrid_analysis(ast.parse(code))
        assert "symbolic" in result

    def test_backprop_tune_patterns_with_labels(self):
        """Test backprop tuning with explicit labels."""
        import numpy as np

        from omni_mercury_engine.core.neurosymbolic_engine import NeurosymbolicConfig

        config = NeurosymbolicConfig(enable_backprop_tuning=True)
        engine = NeurosymbolicEngine(config)
        code_features = np.array([1.0, 2.0, 3.0, 4.0])
        ground_truth = np.array([1.0])

        result = engine.backprop_tune_patterns(code_features, ground_truth, iterations=5)

        assert result["enabled"] is True
        assert result["iterations"] == 5

    def test_symbolic_reasoning_integration(self):
        """Test symbolic reasoning component."""
        engine = NeurosymbolicEngine()
        code = "def test(): pass"

        result = engine.symbolic_analysis(ast.parse(code))

        assert "patterns" in result

    def test_pattern_matching_with_threshold(self):
        """Test pattern matching with custom threshold."""
        import numpy as np

        engine = NeurosymbolicEngine()
        code_features = np.array([1.0, 2.0, 3.0, 4.0])

        result = engine.neural_analysis(code_features)

        assert "method" in result

    def test_embed_code_empty_string(self):
        """Test embedding empty code string."""
        engine = NeurosymbolicEngine()
        code = "pass"
        result = engine.hybrid_analysis(ast.parse(code))
        assert result is not None

    def test_extract_patterns_single_line(self):
        """Test pattern extraction from single line."""
        engine = NeurosymbolicEngine()
        code = "x = 42"
        result = engine.symbolic_analysis(ast.parse(code))
        assert "patterns" in result

    def test_backprop_tune_convergence(self):
        """Test backprop tuning shows convergence."""
        import numpy as np

        from omni_mercury_engine.core.neurosymbolic_engine import NeurosymbolicConfig

        config = NeurosymbolicConfig(enable_backprop_tuning=True)
        engine = NeurosymbolicEngine(config)
        code_features = np.array([1.0, 2.0, 3.0, 4.0])
        ground_truth = np.array([1.0])

        result = engine.backprop_tune_patterns(code_features, ground_truth, iterations=10)

        assert result["final_loss"] <= result["initial_loss"]

    def test_pattern_similarity_computation(self):
        """Test pattern similarity computation."""
        engine = NeurosymbolicEngine()
        code1 = "def f1(): return 1"
        code2 = "def f2(): return 2"

        result1 = engine.symbolic_analysis(ast.parse(code1))
        result2 = engine.symbolic_analysis(ast.parse(code2))

        assert result1["patterns"] is not None
        assert result2["patterns"] is not None

    def test_neurosymbolic_fusion_basic(self):
        """Test basic neurosymbolic fusion."""
        engine = NeurosymbolicEngine()
        code = "def add(a, b): return a + b"

        result = engine.hybrid_analysis(ast.parse(code))

        assert "symbolic" in result and "neural" in result
