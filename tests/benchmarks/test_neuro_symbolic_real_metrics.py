# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pin: the neuro-symbolic benchmark reports real metrics, never fabrication.

The ``domain_performance`` block and the memory-growth panel were previously
filled with ``np.random`` numbers and a fixed-ratio split of the memory
total - synthetic values rendered into a committed report image. These
tests pin the honest-aggregation contract so that regression cannot return.
"""

from __future__ import annotations

from benchmarks.neuro_symbolic_benchmark import run_neuro_symbolic_benchmark


def test_domain_performance_is_real_aggregation_not_fabrication() -> None:
    """Per-domain metrics aggregate real observations, not an RNG draw.

    The cognitive stack is legitimately stochastic, so the numbers vary
    run to run - that is real variation, not fabrication. The structural
    signature that pins "real" is that ``anomaly_accuracy`` is the mean of
    per-observation 0/1 correctness, so it must be exactly ``k /
    n_observations`` for some integer k. The old ``0.85 + np.random``
    fabrication produced values independent of the run that would almost
    never land on such a fraction, and reported confidences above 0.85
    regardless of the scenario's real (~0.5-0.76) confidence.
    """
    performance = run_neuro_symbolic_benchmark(epochs=2)["domain_performance"]
    assert performance, "expected at least one measured domain"
    for domain, metrics in performance.items():
        n = metrics["n_observations"]
        k = metrics["anomaly_accuracy"] * n
        assert abs(k - round(k)) < 1e-9, (
            f"{domain}.anomaly_accuracy={metrics['anomaly_accuracy']} is not k/n "
            f"(n={n}); it is not a real mean of 0/1 correctness"
        )
        for key in ("avg_confidence", "avg_benevolence", "anomaly_accuracy"):
            assert 0.0 <= metrics[key] <= 1.0


def test_domain_performance_has_no_structural_success_rate_axis() -> None:
    """success_rate is a structural 0.0 here (no tools bound); not reported."""
    performance = run_neuro_symbolic_benchmark(epochs=2)["domain_performance"]
    for domain, metrics in performance.items():
        assert "avg_success_rate" not in metrics, (
            f"{domain} still carries avg_success_rate; it reads as failure for "
            "pure-reasoning scenarios rather than 'not applicable'"
        )
        assert metrics["n_observations"] == 2  # one scenario per domain, two epochs


def test_memory_growth_uses_real_per_type_counts() -> None:
    """Each epoch records the agent's real per-type memory counts."""
    summaries = run_neuro_symbolic_benchmark(epochs=2)["epoch_summaries"]
    last = summaries[-1]
    for key in ("short_term_count", "long_term_count", "episodic_count", "semantic_count"):
        assert key in last, f"missing real memory field {key}"
    # The parts reconcile with the total the benchmark reports.
    parts = (
        last["short_term_count"]
        + last["long_term_count"]
        + last["episodic_count"]
        + last["semantic_count"]
    )
    assert parts == last["memory_entries"]
