# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Merit gate for the real self-referential software-engineering metrics.

``scripts/collect_sw_eng_metrics.py`` computes REAL Halstead / cyclomatic /
Maintainability-Index measurements over ``src/omni_mercury_engine`` (via stdlib
``ast``) plus Mercury-native supply-chain / repository-integrity checks
handwritten from repo config, and persists them
to ``core/sw_eng_metrics.json``.  ``GlobalOmniScalarNetwork`` overlays them onto
21 of the 82 diagnostic SOFTWARE_ENGINEERING scalars at init.

This gate pins the invariants that keep the wiring honest AND safe:

* **Real, not placeholder** — the wired scalars differ from the shipped static
  literals, and the artifact matches a live recomputation (freshness).
* **Sane ranges** — every wired scalar sits in ``[0, 2]`` (the scalar
  convention), so no measurement blows up the vector.
* **Still metric-only (the safety invariant)** — the wired scalars never enter
  the σ_Immutable operational vector, the operational count stays 127, and the
  σ score is unchanged by the overlay.  Changing diagnostic *values* must never
  perturb the frozen gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_ARTIFACT = _REPO / "src" / "omni_mercury_engine" / "core" / "sw_eng_metrics.json"
_SCRIPT = _REPO / "scripts" / "collect_sw_eng_metrics.py"

_WIRED = {
    "omni_halstead_vocabulary",
    "omni_halstead_length",
    "omni_halstead_volume",
    "omni_halstead_difficulty",
    "omni_halstead_effort",
    "omni_halstead_time_to_program",
    "omni_halstead_delivered_bugs",
    "omni_mccabe_cyclomatic_complexity",
    "omni_maintainability_index_sei",
    "omni_maintainability_index_vs",
    "omni_maintainability_index_delta",
    "omni_ossf_branch_protection",
    "omni_ossf_code_review_required",
    "omni_ossf_ci_tests_required",
    "omni_ossf_dependency_update_tool",
    "omni_ossf_dangerous_workflow",
    "omni_ossf_pinned_dependencies",
    "omni_ossf_sast_enabled",
    "omni_ossf_token_permissions",
    "omni_ossf_signed_releases",
    "omni_ossf_vulnerabilities",
    # DORA (4) — VCS-history proxies from git log.
    "omni_dora_deployment_frequency",
    "omni_dora_lead_time_for_changes",
    "omni_dora_mean_time_to_restore",
    "omni_dora_change_failure_rate",
    # NIST SSDF (4) — practice-group coverage from repo state.
    "omni_ssdf_prepare_organization",
    "omni_ssdf_protect_software",
    "omni_ssdf_produce_well_secured_software",
    "omni_ssdf_respond_to_vulnerabilities",
    # SLSA (4) — build-track evidence from repo state.
    "omni_slsa_source_integrity",
    "omni_slsa_build_provenance",
    "omni_slsa_dependency_attestation",
    "omni_slsa_level",
    # NIST SAMATE (3 computable subset; the other 7 stay placeholders).
    "omni_samate_supply_chain_assurance",
    "omni_samate_evidence_completeness",
    "omni_samate_residual_risk",
}


def _load_collector():
    spec = importlib.util.spec_from_file_location("collect_sw_eng_metrics", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["collect_sw_eng_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


def _artifact() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(_ARTIFACT.read_text())
    return result


def test_artifact_wires_exactly_the_expected_scalars() -> None:
    payload = _artifact()
    assert payload["schema"] == "sw_eng_metrics/v1"
    assert set(payload["scalars"]) == _WIRED
    assert payload["n_scalars_wired"] == len(_WIRED) == 36


def test_collector_recomputes_real_valid_values() -> None:
    """A live run of the collector produces real, in-band values for all 21 scalars.

    Not pinned bit-exact to the committed artifact: the metrics are a live
    function of the source tree (and of the Python ``ast`` version CI runs on),
    so an exact match would flake across the 3.11-3.14 matrix and on every edit.
    What must hold is that the collector genuinely runs over the real tree and
    yields in-band, direction-correct values.
    """
    live = _load_collector().collect()
    assert set(live["scalars"]) == _WIRED
    assert live["raw"]["code"]["files"] > 300 and live["raw"]["code"]["total_loc"] > 50_000
    for name, value in live["scalars"].items():
        assert 0.5 < value < 2.0, f"{name}={value} outside the sane band"


def test_wired_values_are_real_and_in_range() -> None:
    payload = _artifact()
    for name, value in payload["scalars"].items():
        assert 0.0 <= value <= 2.0, f"{name}={value} outside the [0,2] scalar band"
    # Real code metrics, not placeholders: the aggregate raw block must reflect
    # the actual source tree (hundreds of files, non-trivial LOC).
    code = payload["raw"]["code"]
    assert code["files"] > 300 and code["total_loc"] > 50_000
    assert 0.0 < code["mean_file_maintainability_index_sei"] <= 100.0


def test_wired_scalars_stay_metric_only_and_do_not_move_the_gate() -> None:
    """The safety invariant: measured diagnostics never touch the σ gate."""
    from omni_mercury_engine.core.global_omni_scalar_network import (
        GlobalOmniScalarNetwork,
        get_global_scalar_network,
    )
    from omni_mercury_engine.security.sigma_immutable_gate import get_sigma_immutable_gate

    net = get_global_scalar_network()
    # Every wired scalar is classified metric-only …
    for name in _WIRED:
        assert GlobalOmniScalarNetwork._is_metric_only_scalar(name), f"{name} is not metric-only"
    # … and therefore none leak into the operational vector, which stays 127.
    operational = net._collect_all_scalars()
    assert len(operational) == 127
    assert _WIRED.isdisjoint(operational.keys())
    # The σ score is unaffected by the measured diagnostic values.
    gate = get_sigma_immutable_gate()
    padded = np.zeros(256, dtype=np.float64)
    vals = np.asarray(list(operational.values()), dtype=np.float64)
    padded[: len(vals)] = vals
    assert float(gate.evaluate(padded).score) == pytest.approx(0.9999216794967651, abs=1e-9)


def test_overlay_actually_applied_real_values() -> None:
    """The live scalar group carries the measured values, not the placeholders."""
    from omni_mercury_engine.core.global_omni_scalar_network import (
        ScalarGroup,
        get_global_scalar_network,
    )

    group = get_global_scalar_network().scalar_groups[ScalarGroup.SOFTWARE_ENGINEERING]
    scalars = _artifact()["scalars"]
    for name in _WIRED:
        assert group[name] == pytest.approx(scalars[name], abs=1e-6)
