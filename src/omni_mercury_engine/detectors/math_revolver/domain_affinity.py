# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Domain-variant affinity mapping for all 21 Anomaly Math Revolver probes.

Each domain ranks the 21 probes by relevance. Probes at the top of the list
receive higher Phi-weighted contributions during fusion.
"""

from __future__ import annotations

DOMAIN_AFFINITY_21: dict[str, list[str]] = {
    "earthquake": [
        "additive",
        "momentum",
        "wave_propagation",
        "energy_minimization",
        "variance_adapted",
        "ethical_constrained",
        "svd_projection",
        "lyapunov_chaos",
        "r3_recursion_resonance",
        "harmonic_oscillator",
        "topology_homology",
        "iqr_robust",
        "modified_zscore",
        "helix_multiplicative",
        "exponential_decay",
        "catalan_optimized",
        "quantum_annealing",
        "boltzmann_coupling",
        "fractal_self_similarity",
        "zeta_harmonic",
        "quantum_superposition",
    ],
    "tsunami": [
        "harmonic_oscillator",
        "wave_propagation",
        "ethical_constrained",
        "topology_homology",
        "momentum",
        "additive",
        "svd_projection",
        "energy_minimization",
        "variance_adapted",
        "iqr_robust",
        "modified_zscore",
        "r3_recursion_resonance",
        "helix_multiplicative",
        "exponential_decay",
        "catalan_optimized",
        "lyapunov_chaos",
        "quantum_annealing",
        "boltzmann_coupling",
        "fractal_self_similarity",
        "zeta_harmonic",
        "quantum_superposition",
    ],
    "pandemic": [
        "exponential_decay",
        "variance_adapted",
        "helix_multiplicative",
        "quantum_annealing",
        "momentum",
        "additive",
        "r3_recursion_resonance",
        "boltzmann_coupling",
        "catalan_optimized",
        "iqr_robust",
        "modified_zscore",
        "lyapunov_chaos",
        "svd_projection",
        "energy_minimization",
        "harmonic_oscillator",
        "ethical_constrained",
        "fractal_self_similarity",
        "topology_homology",
        "wave_propagation",
        "zeta_harmonic",
        "quantum_superposition",
    ],
    "marine": [
        "helix_multiplicative",
        "harmonic_oscillator",
        "wave_propagation",
        "fractal_self_similarity",
        "variance_adapted",
        "momentum",
        "exponential_decay",
        "topology_homology",
        "additive",
        "svd_projection",
        "r3_recursion_resonance",
        "boltzmann_coupling",
        "catalan_optimized",
        "ethical_constrained",
        "iqr_robust",
        "modified_zscore",
        "lyapunov_chaos",
        "energy_minimization",
        "quantum_annealing",
        "zeta_harmonic",
        "quantum_superposition",
    ],
    "geomagnetic": [
        "harmonic_oscillator",
        "ethical_constrained",
        "wave_propagation",
        "lyapunov_chaos",
        "variance_adapted",
        "additive",
        "svd_projection",
        "quantum_superposition",
        "momentum",
        "exponential_decay",
        "zeta_harmonic",
        "fractal_self_similarity",
        "helix_multiplicative",
        "catalan_optimized",
        "r3_recursion_resonance",
        "iqr_robust",
        "modified_zscore",
        "energy_minimization",
        "quantum_annealing",
        "boltzmann_coupling",
        "topology_homology",
    ],
    "conflict": [
        "additive",
        "momentum",
        "ethical_constrained",
        "energy_minimization",
        "variance_adapted",
        "boltzmann_coupling",
        "exponential_decay",
        "r3_recursion_resonance",
        "helix_multiplicative",
        "svd_projection",
        "lyapunov_chaos",
        "iqr_robust",
        "catalan_optimized",
        "modified_zscore",
        "harmonic_oscillator",
        "quantum_annealing",
        "topology_homology",
        "wave_propagation",
        "fractal_self_similarity",
        "zeta_harmonic",
        "quantum_superposition",
    ],
    "default": [
        "variance_adapted",
        "momentum",
        "additive",
        "modified_zscore",
        "iqr_robust",
        "ethical_constrained",
        "harmonic_oscillator",
        "svd_projection",
        "catalan_optimized",
        "helix_multiplicative",
        "exponential_decay",
        "wave_propagation",
        "energy_minimization",
        "r3_recursion_resonance",
        "lyapunov_chaos",
        "topology_homology",
        "boltzmann_coupling",
        "quantum_annealing",
        "fractal_self_similarity",
        "zeta_harmonic",
        "quantum_superposition",
    ],
}


def get_affinity_order(domain: str, probe_names: list[str]) -> list[int]:
    """Return index ordering of *probe_names* by domain affinity rank.

    Probes not in the affinity map are appended at the end in their
    original order so that no probe is ever dropped.

    Args:
        domain: Domain key (e.g. ``"earthquake"``).  Falls back to
            ``"default"`` if not found.
        probe_names: Current probe names in their natural order.

    Returns:
        List of indices into *probe_names* sorted by affinity rank.
    """
    affinity_list = DOMAIN_AFFINITY_21.get(domain, DOMAIN_AFFINITY_21["default"])
    name_to_idx: dict[str, int] = {name: i for i, name in enumerate(probe_names)}

    ordered: list[int] = []
    seen: set[int] = set()

    for name in affinity_list:
        if name in name_to_idx:
            idx = name_to_idx[name]
            if idx not in seen:
                ordered.append(idx)
                seen.add(idx)

    # Append any probes not in the affinity list
    for i in range(len(probe_names)):
        if i not in seen:
            ordered.append(i)

    return ordered
