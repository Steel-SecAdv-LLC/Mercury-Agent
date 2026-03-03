# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Domain-variant affinity mapping for all 17 Anomaly Math Arrest probes.

Each domain ranks the 17 probes by relevance. Probes at the top of the list
receive higher Phi-weighted contributions during fusion.
"""

from __future__ import annotations

DOMAIN_AFFINITY_17: dict[str, list[str]] = {
    "earthquake": [
        "additive_harmonic",
        "momentum",
        "wave_propagation",
        "energy_minimization",
        "variance_adapted",
        "ethical_iqr",
        "svd_projection",
        "lyapunov_chaos",
        "r3_recursion_resonance",
        "topology_homology",
        "annealed_zscore",
        "helix_multiplicative",
        "catalan_decay",
        "boltzmann_coupling",
        "fractal_self_similarity",
        "zeta_harmonic",
        "quantum_superposition",
    ],
    "tsunami": [
        "additive_harmonic",
        "wave_propagation",
        "ethical_iqr",
        "topology_homology",
        "momentum",
        "svd_projection",
        "energy_minimization",
        "variance_adapted",
        "annealed_zscore",
        "r3_recursion_resonance",
        "helix_multiplicative",
        "catalan_decay",
        "lyapunov_chaos",
        "boltzmann_coupling",
        "fractal_self_similarity",
        "zeta_harmonic",
        "quantum_superposition",
    ],
    "pandemic": [
        "catalan_decay",
        "variance_adapted",
        "helix_multiplicative",
        "annealed_zscore",
        "momentum",
        "additive_harmonic",
        "r3_recursion_resonance",
        "boltzmann_coupling",
        "ethical_iqr",
        "lyapunov_chaos",
        "svd_projection",
        "energy_minimization",
        "fractal_self_similarity",
        "topology_homology",
        "wave_propagation",
        "zeta_harmonic",
        "quantum_superposition",
    ],
    "marine": [
        "helix_multiplicative",
        "additive_harmonic",
        "wave_propagation",
        "fractal_self_similarity",
        "variance_adapted",
        "momentum",
        "catalan_decay",
        "topology_homology",
        "svd_projection",
        "r3_recursion_resonance",
        "boltzmann_coupling",
        "ethical_iqr",
        "annealed_zscore",
        "lyapunov_chaos",
        "energy_minimization",
        "zeta_harmonic",
        "quantum_superposition",
    ],
    "geomagnetic": [
        "additive_harmonic",
        "ethical_iqr",
        "wave_propagation",
        "lyapunov_chaos",
        "variance_adapted",
        "svd_projection",
        "quantum_superposition",
        "momentum",
        "catalan_decay",
        "zeta_harmonic",
        "fractal_self_similarity",
        "helix_multiplicative",
        "r3_recursion_resonance",
        "annealed_zscore",
        "energy_minimization",
        "boltzmann_coupling",
        "topology_homology",
    ],
    "conflict": [
        "additive_harmonic",
        "momentum",
        "ethical_iqr",
        "energy_minimization",
        "variance_adapted",
        "boltzmann_coupling",
        "catalan_decay",
        "r3_recursion_resonance",
        "helix_multiplicative",
        "svd_projection",
        "lyapunov_chaos",
        "annealed_zscore",
        "topology_homology",
        "wave_propagation",
        "fractal_self_similarity",
        "zeta_harmonic",
        "quantum_superposition",
    ],
    "default": [
        "variance_adapted",
        "momentum",
        "additive_harmonic",
        "annealed_zscore",
        "ethical_iqr",
        "svd_projection",
        "catalan_decay",
        "helix_multiplicative",
        "wave_propagation",
        "energy_minimization",
        "r3_recursion_resonance",
        "lyapunov_chaos",
        "topology_homology",
        "boltzmann_coupling",
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
    affinity_list = DOMAIN_AFFINITY_17.get(domain, DOMAIN_AFFINITY_17["default"])
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
