# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every built-in GOSNN scalar carries the ``omni`` prefix, without moving σ.

GOSNN treats scalar names as opaque keys; what σ_Immutable depends on is the
order of the operational vector and which names its prefix filters exclude.
The two operational scalars renamed to the prefix keep their position, so the
harvested baseline only changes its name lists and the signed training corpus
(features and labels, no names) is untouched. The old names still resolve.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.security.sigma_immutable_corpus import BASELINE_PATH

RENAMED = {
    "civilization_first_principle": ("omni_civilization_first_principle", 1.35),
    "bias_audit_compliance": ("omni_bias_audit_compliance", 1.25),
}


@pytest.fixture
def gosnn() -> Any:
    reset_global_network()
    yield GlobalOmniScalarNetwork()
    reset_global_network()


def test_every_built_in_scalar_carries_the_omni_prefix(gosnn: Any) -> None:
    unprefixed = sorted(
        name
        for group in gosnn.scalar_groups.values()
        for name in group
        if not name.startswith("omni")
    )
    assert unprefixed == []


@pytest.mark.parametrize("legacy", sorted(RENAMED))
def test_renamed_scalars_still_resolve_by_their_old_names(gosnn: Any, legacy: str) -> None:
    current, value = RENAMED[legacy]
    assert gosnn.resolve_scalar_name(legacy) == current
    assert gosnn.get_scalar(legacy) == gosnn.get_scalar(current) == value


def test_rename_kept_the_operational_vector_order(gosnn: Any) -> None:
    ethical = list(gosnn.scalar_groups[ScalarGroup.ETHICAL])
    assert ethical[-2:] == [
        RENAMED[n][0] for n in ("civilization_first_principle", "bias_audit_compliance")
    ]
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    for legacy, (current, _value) in RENAMED.items():
        assert legacy not in baseline["names"]
        assert current in baseline["names"]
