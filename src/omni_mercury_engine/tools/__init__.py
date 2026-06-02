"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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

------------------------------------------------------------------------

One-shot operator tools that run outside the engine.

Modules in this package are intentionally not imported by
``omni_mercury_engine.__init__`` and never exercised by the runtime
detection / training / inference paths. They exist only so operators
can perform offline maintenance tasks (such as migrating legacy
``.pkl`` training payloads) without the engine ever loading the
dangerous code paths involved.

This module exposes a stable :data:`TOOL_REGISTRY` mapping ``name`` →
``module.main`` for every tool wired through ``_base.run_cli``.  The
``mercury-agent tool <name>`` CLI subcommand and the parametrised
exit-code contract test in ``tests/tools/test_tool_contract.py`` both
key off this registry, so a tool added without a registry entry is a
hard test failure.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from typing import Final, cast

# Canonical list of tools.  Order is purely cosmetic (mirrors the order
# in docs/TOOLS.md); the runtime sorts before iteration.  Every entry
# must be a module under ``omni_mercury_engine.tools`` with a
# ``main(argv: list[str] | None) -> int`` entry-point.
_TOOL_NAMES: Final[tuple[str, ...]] = (
    # Ethical / mathematical certifiers
    "lyapunov_validator",
    "benevolence_certifier",
    "oae_weight_certifier",
    "convergence_proof_emitter",
    "benevolence_calibration_report",
    "oae_dimensionality_probe",
    "oae_eigen_monitor",
    "ethical_gate_coverage_report",
    "sigma_immutable_drift_monitor",
    "fairness_subgroup_explorer",
    # Cryptography & supply chain
    "sigma_immutable_verifier",
    "pqc_capability_probe",
    "pqc_handshake_simulator",
    "kat_runner_standalone",
    "algorithm_name_drift_gate",
    "corpus_resigner",
    "hsm_attestation_probe",
    "tls_posture_probe",
    "slsa_provenance_emitter",
    "signed_release_bundle",
    "secret_scan_baseline",
    "audit_log_signer",
    "audit_log_verifier",
    "hwrng_audit",
    # Datasets
    "loader_reachability_probe",
    "dataset_checksum_manifest",
    "bias_audit_standalone",
    "synthetic_fallback_auditor",
    "live_dataset_protection_gate",
    "dataset_license_auditor",
    "pii_scrubber_probe",
    "loader_schema_pinner",
    "synthetic_provenance_tag",
    "network_egress_recorder",
    # Benchmarks / performance
    "run_hardware_benchmark",
    "benchmark_diff",
    "detector_profiler",
    "gosnn_latency_sla_gate",
    "memory_leak_sentinel",
    "gpu_capability_probe",
    "thermal_throttle_probe",
    # Configuration & deployment
    "config_validator",
    "helm_values_linter",
    "image_surface_auditor",
    "workflow_version_drift_gate",
    "network_policy_synthesiser",
    "pod_security_standard_gate",
    "dockerfile_lockfile_gate",
    "reproducible_build_probe",
    "config_secret_redactor",
    # Observability & runtime evidence
    "gate_trace_probe",
    "disconnect_tester",
    "gosnn_scalar_dump",
    "federated_round_simulator",
    "opentelemetry_span_emitter",
    "prometheus_metrics_exporter",
    "time_source_probe",
    # Release & ML governance
    "release_manifest_builder",
    "sbom_emitter",
    "api_contract_diff",
    "changelog_enforcer",
    "model_card_generator",
    "dataset_card_generator",
    "adversarial_probe",
)


def _resolve(name: str) -> Callable[[list[str] | None], int]:
    """Lazily import a tool module and return its ``main`` entry-point.

    The ``cast`` narrows ``module.main`` (typed as ``Any`` by the
    ``importlib`` stubs because the module is loaded by string) to the
    documented contract every tool obeys: ``(argv: list[str] | None) -> int``.
    The contract is enforced at runtime by
    :mod:`tests.tools.test_tool_contract`, so the cast does not lose
    real type safety.
    """
    module = importlib.import_module(f"omni_mercury_engine.tools.{name}")
    main = getattr(module, "main", None)
    if not callable(main):
        raise ImportError(f"tool {name!r} has no callable main() entry-point")
    return cast("Callable[[list[str] | None], int]", main)


class _Registry:
    """Lazy mapping of tool-name → ``main`` callable.

    Implemented as a small ``Mapping``-like class rather than a dict
    populated at import time so that pulling in the registry does not
    eagerly import every tool — many tools optionally depend on torch,
    cryptography backends, or external services and we want
    ``mercury-agent tool list`` to work even when those are absent.
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = tuple(sorted(names))

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._names

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, name: str) -> Callable[[list[str] | None], int]:
        if name not in self._names:
            raise KeyError(name)
        return _resolve(name)

    def names(self) -> tuple[str, ...]:
        """Return the sorted tuple of registered tool names."""
        return self._names


TOOL_REGISTRY: Final[_Registry] = _Registry(_TOOL_NAMES)
"""Canonical registry of operator tools.

Iterate to discover tools, index to obtain the ``main`` callable.  The
``mercury-agent tool list`` CLI and the parametrised exit-code contract
test in :mod:`tests.tools.test_tool_contract` both key off this
registry.
"""


__all__ = ["TOOL_REGISTRY"]
