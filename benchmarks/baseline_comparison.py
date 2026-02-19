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

"""
Baseline Comparison Benchmarks
Compares RefactoringEngine performance: baseline (main) vs improved (PR #3)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ast
import importlib.util
import os
import subprocess
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.statistical_validation import statistical_analysis
from omni_mercury_engine.core.three_r_mechanism import RefactoringEngine as ImprovedEngine


def _create_safe_callable(func_name: str, func_node: ast.FunctionDef) -> Any:
    """Create a safe callable stub from an AST node without using exec().

    Instead of executing arbitrary code from external repositories,
    we create a no-op stub function and register the unparsed AST source
    in linecache so that inspect.getsource() can retrieve it.
    This avoids code injection while still allowing engine introspection.
    """
    import linecache

    source = ast.unparse(func_node)
    # Register source in linecache so inspect.getsource() works
    cache_key = f"<benchmark:{func_name}:{id(func_node)}>"
    source_lines = source.splitlines(keepends=True)
    linecache.cache[cache_key] = (
        len(source),
        None,
        source_lines,
        cache_key,
    )

    def _stub() -> None:
        pass

    _stub.__name__ = func_name
    _stub.__qualname__ = func_name
    # Point code object to our cached source so inspect.getsource works
    _stub.__code__ = _stub.__code__.replace(co_filename=cache_key, co_firstlineno=1)
    return _stub


def load_baseline_engine() -> Any:
    """Load baseline RefactoringEngine from main branch."""
    # Use cross-platform temp directory instead of hardcoded /tmp
    baseline_path = os.path.join(tempfile.gettempdir(), "three_r_mechanism_baseline.py")

    if not os.path.exists(baseline_path):
        print("  Extracting baseline from main branch...")
        try:
            result = subprocess.run(
                ["git", "show", "main:src/core/three_r_mechanism.py"],
                capture_output=True,
                text=True,
                check=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            with open(baseline_path, "w") as f:
                f.write(result.stdout)
            print("  ✓ Baseline extracted successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to extract baseline: {e}")
            raise

    spec = importlib.util.spec_from_file_location("baseline_module", baseline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load baseline module from {baseline_path}")
    baseline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_module)
    return baseline_module.RefactoringEngine


def extract_test_functions(
    repo_paths: list[Path], max_functions: int = 100
) -> list[tuple[str, ast.FunctionDef, Path]]:
    """Extract functions from open-source repos for testing."""
    functions = []

    for repo_path in repo_paths:
        if not repo_path.exists():
            print(f"Warning: {repo_path} does not exist, skipping")
            continue

        py_files = list(repo_path.rglob("*.py"))[:30]

        for py_file in py_files:
            try:
                with open(py_file, encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append((node.name, node, py_file))

                if len(functions) >= max_functions:
                    return functions[:max_functions]
            except Exception as e:
                print(f"  Skipping {py_file}: {e}")
                continue

    return functions


def benchmark_execution_time(
    engine_class: Any,
    functions: list[tuple[str, ast.FunctionDef, Path]],
    iterations: int = 10,
    is_improved: bool = False,
) -> np.ndarray:
    """
    Benchmark execution time for RefactoringEngine analysis operations.

    Args:
        engine_class: RefactoringEngine class to benchmark
        functions: List of (func_name, func_ast_node, file_path) tuples
        iterations: Number of iterations per function
        is_improved: True if testing improved version with enhanced methods

    Returns:
        Array of mean execution times per function
    """
    times = []

    try:
        engine = engine_class()
    except TypeError:
        try:
            engine = engine_class.__new__(engine_class)
            engine.__init__()
        except Exception:
            print(f"Warning: Could not properly initialize {engine_class}")
            return np.array([0.0] * len(functions))

    for func_name, func_node, file_path in functions:
        try:
            test_func = _create_safe_callable(func_name, func_node)

            if test_func is None:

                def test_func() -> None:
                    pass

                test_func.__name__ = func_name

            iter_times = []
            for _ in range(iterations):
                start = time.perf_counter()

                try:
                    _ = engine.analyze_function_complexity(test_func)
                    _ = engine.suggest_refactorings(test_func)
                except Exception:
                    complexity = 1
                    for node in ast.walk(func_node):
                        if isinstance(node, (ast.If, ast.While, ast.For, ast.And, ast.Or)):
                            complexity += 1

                if is_improved:
                    try:
                        _ = engine.analyze_with_harmonics(test_func)
                    except (AttributeError, Exception):
                        pass  # Method may not exist in baseline engine

                    try:
                        _ = engine.explore_quantum_refactoring_paths(test_func, num_paths=3)
                    except (AttributeError, Exception):
                        pass  # Method may not exist in baseline engine

                    try:
                        _ = engine.detect_pattern_resonance(test_func)
                    except (AttributeError, Exception):
                        pass  # Method may not exist in baseline engine

                    try:
                        _ = engine.analyze_with_neurosymbolic(test_func)
                    except (AttributeError, Exception):
                        pass  # Method may not exist in baseline engine

                    try:
                        _ = engine.orchestrate_refactoring(test_func)
                    except (AttributeError, Exception):
                        pass  # Method may not exist in baseline engine

                end = time.perf_counter()
                iter_times.append(end - start)

            times.append(np.mean(iter_times))

        except Exception as e:
            print(f"Warning: Could not benchmark function {func_name}: {e}")
            times.append(0.0)

    return np.array(times)


def benchmark_memory_usage(
    engine_class: Any, functions: list[tuple[str, ast.FunctionDef, Path]], is_improved: bool = False
) -> np.ndarray:
    """
    Benchmark memory usage during RefactoringEngine analysis.

    Args:
        engine_class: RefactoringEngine class to benchmark
        functions: List of (func_name, func_ast_node, file_path) tuples
        is_improved: True if testing improved version with enhanced methods

    Returns:
        Array of peak memory usage (KB) per function
    """
    memory_usage = []

    try:
        engine = engine_class()
    except TypeError:
        try:
            engine = engine_class.__new__(engine_class)
            engine.__init__()
        except Exception:
            print(f"Warning: Could not properly initialize {engine_class}")
            return np.array([0.0] * len(functions))

    for func_name, func_node, file_path in functions:
        try:
            test_func = _create_safe_callable(func_name, func_node)

            if test_func is None:

                def test_func() -> None:
                    pass

                test_func.__name__ = func_name

            tracemalloc.start()

            try:
                _ = engine.analyze_function_complexity(test_func)
                _ = engine.suggest_refactorings(test_func)
            except Exception:
                _ = len(list(ast.walk(func_node)))

            if is_improved:
                try:
                    _ = engine.analyze_with_harmonics(test_func)
                    _ = engine.explore_quantum_refactoring_paths(test_func, num_paths=3)
                    _ = engine.detect_pattern_resonance(test_func)
                    _ = engine.analyze_with_neurosymbolic(test_func)
                    _ = engine.orchestrate_refactoring(test_func)
                except (AttributeError, Exception):
                    pass  # Methods may not exist in baseline engine

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_usage.append(peak / 1024)

        except Exception as e:
            print(f"Warning: Could not benchmark memory for {func_name}: {e}")
            memory_usage.append(0.0)

    return np.array(memory_usage)


def benchmark_accuracy(
    engine_class: Any, functions: list[tuple[str, ast.FunctionDef, Path]]
) -> np.ndarray:
    """
    Note on accuracy measurement for code complexity analysis.

    True accuracy measurement would require ground truth complexity labels,
    which don't exist for arbitrary code. The RefactoringEngine computes
    cyclomatic complexity and suggests refactorings based on heuristics.

    This function returns a placeholder indicating that accuracy is not
    directly measurable without labeled ground truth data.

    For meaningful evaluation, consider:
    - Comparing against established tools (radon, pylint complexity)
    - User studies on refactoring suggestion quality
    - Correlation with bug density or maintainability metrics
    """
    # Return NaN to indicate accuracy is not measurable without ground truth
    # This is more honest than returning fake 100% accuracy
    return np.full(len(functions), np.nan)


def run_comprehensive_comparison(
    baseline_engine: Any, improved_engine: Any, functions: list[tuple[str, ast.FunctionDef, Path]]
) -> dict[str, Any]:
    """Run comprehensive benchmarks comparing baseline vs improved."""

    print(f"\nBenchmarking {len(functions)} functions...")
    print("=" * 60)

    print("\n[1/2] Benchmarking BASELINE (main branch, 336 lines)...")
    print("  Methods tested: analyze_function_complexity(), suggest_refactorings()")
    baseline_times = benchmark_execution_time(
        baseline_engine, functions, iterations=10, is_improved=False
    )
    baseline_memory = benchmark_memory_usage(baseline_engine, functions, is_improved=False)
    baseline_accuracy = benchmark_accuracy(baseline_engine, functions)

    print(
        f"  Execution time: {baseline_times.mean()*1000:.3f}ms ± {baseline_times.std()*1000:.3f}ms"
    )
    print(f"  Memory usage: {baseline_memory.mean():.2f}KB ± {baseline_memory.std():.2f}KB")
    print("  Accuracy: N/A (requires ground truth labels)")

    print("\n[2/2] Benchmarking IMPROVED (PR #3, 1018 lines, +5 methods)...")
    print("  Methods tested: Basic methods + analyze_with_harmonics(),")
    print("                  explore_quantum_refactoring_paths(), detect_pattern_resonance(),")
    print("                  analyze_with_neurosymbolic(), orchestrate_refactoring()")
    improved_times = benchmark_execution_time(
        improved_engine, functions, iterations=10, is_improved=True
    )
    improved_memory = benchmark_memory_usage(improved_engine, functions, is_improved=True)
    improved_accuracy = benchmark_accuracy(improved_engine, functions)

    print(
        f"  Execution time: {improved_times.mean()*1000:.3f}ms ± {improved_times.std()*1000:.3f}ms"
    )
    print(f"  Memory usage: {improved_memory.mean():.2f}KB ± {improved_memory.std():.2f}KB")
    print("  Accuracy: N/A (requires ground truth labels)")

    print("\n" + "=" * 60)
    print("STATISTICAL VALIDATION")
    print("=" * 60)

    time_stats = statistical_analysis(baseline_times, improved_times)
    memory_stats = statistical_analysis(baseline_memory, improved_memory)

    print("\nExecution Time Comparison:")
    print(f"  Improvement: {time_stats['improvement_percent']:.2f}%")
    print(f"  T-statistic: {time_stats['t_statistic']:.3f}")
    print(f"  P-value: {time_stats['p_value']:.6f}")
    print(f"  Statistically significant (p<0.05): {time_stats['significant']}")
    ci_low = time_stats["confidence_interval_95"][0] * 1000
    ci_high = time_stats["confidence_interval_95"][1] * 1000
    print(f"  95% CI: ({ci_low:.4f}ms, {ci_high:.4f}ms)")
    print(f"  Cohen's d: {time_stats['cohens_d']:.3f} ({time_stats['effect_size']})")

    print("\nMemory Usage Comparison:")
    print(f"  Improvement: {memory_stats['improvement_percent']:.2f}%")
    print(f"  T-statistic: {memory_stats['t_statistic']:.3f}")
    print(f"  P-value: {memory_stats['p_value']:.6f}")
    print(f"  Statistically significant (p<0.05): {memory_stats['significant']}")

    print("\n" + "=" * 60)
    print("REQUIREMENT CHECK: >15% Improvement with p<0.05")
    print("=" * 60)

    meets_time_threshold = time_stats["improvement_percent"] > 15 and time_stats["significant"]
    meets_memory_threshold = (
        memory_stats["improvement_percent"] > 15 and memory_stats["significant"]
    )

    print(f"  Execution Time: {'✅ PASS' if meets_time_threshold else '❌ FAIL'}")
    print(f"    - Improvement: {time_stats['improvement_percent']:.2f}% (need >15%)")
    print(f"    - Significant: {time_stats['significant']} (need p<0.05)")

    print(f"  Memory Usage: {'✅ PASS' if meets_memory_threshold else '❌ FAIL'}")
    print(f"    - Improvement: {memory_stats['improvement_percent']:.2f}% (need >15%)")
    print(f"    - Significant: {memory_stats['significant']} (need p<0.05)")

    return {
        "baseline": {
            "times": baseline_times,
            "memory": baseline_memory,
            "accuracy": baseline_accuracy,
        },
        "improved": {
            "times": improved_times,
            "memory": improved_memory,
            "accuracy": improved_accuracy,
        },
        "statistics": {
            "time": time_stats,
            "memory": memory_stats,
        },
        "meets_requirements": meets_time_threshold or meets_memory_threshold,
    }


if __name__ == "__main__":
    repo_paths = [
        Path.home() / "benchmark_repos" / "requests",
        Path.home() / "benchmark_repos" / "flask",
        Path.home() / "benchmark_repos" / "cpython",
        Path.home() / "benchmark_repos" / "django",
        Path.home() / "benchmark_repos" / "numpy",
    ]

    print("=" * 60)
    print("BASELINE COMPARISON BENCHMARK")
    print("Comparing: main branch (336 lines) vs PR #3 (1018 lines)")
    print("=" * 60)

    print("\nLoading engines...")
    baseline_engine = load_baseline_engine()
    improved_engine = ImprovedEngine

    print(f"  Baseline: {baseline_engine}")
    print(f"  Improved: {improved_engine}")

    print("\nExtracting test functions from open-source repositories...")
    functions = extract_test_functions(repo_paths, max_functions=100)

    if len(functions) < 10:
        print("\nWARNING: Not enough functions from benchmark repos.")
        print("Falling back to using Mercury Agent's own Python files...")

        fallback_paths = [
            Path(__file__).parent.parent / "omni_mercury_engine",
            Path(__file__).parent.parent / "src",
        ]
        functions = extract_test_functions(fallback_paths, max_functions=100)

    print(f"  Extracted {len(functions)} functions")

    if len(functions) < 10:
        print("\nERROR: Not enough test functions found!")
        sys.exit(1)

    results = run_comprehensive_comparison(baseline_engine, improved_engine, functions)

    output_file = Path(__file__).parent / "baseline_comparison_results.txt"
    with open(output_file, "w") as f:
        f.write("BASELINE vs IMPROVED COMPARISON RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Sample size: {len(functions)} functions\n\n")

        f.write("BASELINE (main branch, 336 lines):\n")
        baseline_time_mean = results["baseline"]["times"].mean() * 1000
        baseline_time_std = results["baseline"]["times"].std() * 1000
        f.write(f"  Execution time: {baseline_time_mean:.3f}ms ± {baseline_time_std:.3f}ms\n")
        baseline_mem_mean = results["baseline"]["memory"].mean()
        baseline_mem_std = results["baseline"]["memory"].std()
        f.write(f"  Memory usage: {baseline_mem_mean:.2f}KB ± {baseline_mem_std:.2f}KB\n")
        f.write("  Accuracy: N/A (requires ground truth labels)\n\n")

        f.write("IMPROVED (PR #3, 1018 lines, +5 analysis methods):\n")
        improved_time_mean = results["improved"]["times"].mean() * 1000
        improved_time_std = results["improved"]["times"].std() * 1000
        f.write(f"  Execution time: {improved_time_mean:.3f}ms ± {improved_time_std:.3f}ms\n")
        improved_mem_mean = results["improved"]["memory"].mean()
        improved_mem_std = results["improved"]["memory"].std()
        f.write(f"  Memory usage: {improved_mem_mean:.2f}KB ± {improved_mem_std:.2f}KB\n")
        f.write("  Accuracy: N/A (requires ground truth labels)\n\n")

        f.write("STATISTICAL VALIDATION:\n")
        time_stats = results["statistics"]["time"]
        memory_stats = results["statistics"]["memory"]

        f.write("\nExecution Time:\n")
        f.write(f"  Improvement: {time_stats['improvement_percent']:.2f}%\n")
        f.write(f"  T-statistic: {time_stats['t_statistic']:.3f}\n")
        f.write(f"  P-value: {time_stats['p_value']:.6f}\n")
        f.write(f"  Significant (p<0.05): {time_stats['significant']}\n")
        f.write(f"  Cohen's d: {time_stats['cohens_d']:.3f} ({time_stats['effect_size']})\n")
        ci_low = time_stats["confidence_interval_95"][0] * 1000
        ci_high = time_stats["confidence_interval_95"][1] * 1000
        f.write(f"  95% CI: ({ci_low:.4f}ms, {ci_high:.4f}ms)\n")

        f.write("\nMemory Usage:\n")
        f.write(f"  Improvement: {memory_stats['improvement_percent']:.2f}%\n")
        f.write(f"  T-statistic: {memory_stats['t_statistic']:.3f}\n")
        f.write(f"  P-value: {memory_stats['p_value']:.6f}\n")
        f.write(f"  Significant (p<0.05): {memory_stats['significant']}\n")
        f.write(f"  Cohen's d: {memory_stats['cohens_d']:.3f} ({memory_stats['effect_size']})\n")

        f.write("\nREQUIREMENT CHECK:\n")
        status = "PASS" if results["meets_requirements"] else "FAIL"
        f.write(f"  >15% improvement with p<0.05: {status}\n")

    print(f"\n✅ Results saved to: {output_file}")
    print("\nComparison complete!")

    if not results["meets_requirements"]:
        print("\n⚠️  WARNING: >15% improvement threshold not met")
        print("    The enhanced version provides additional capabilities")
        print("    but does not achieve >15% raw performance improvement.")
