# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OAE dimensionality probe must be able to fail.

The tool this replaced drove random Gaussian tensors through an inline
``w_R * a + w_H * b + w_O * c`` and asserted the result kept its shape, that
the fused L2 norm was bounded by the largest input's, and that equal inputs
fused to themselves. All three are convex-combination identities: measured,
it returned ``ok`` for ``(1, 0, 0)``, ``(0, 0, 1)``, uniform and
``(0.9, 0.05, 0.05)``, failing only on a negative weight -- which
``oae_weight_certifier`` already covers. It never imported the OAE.

So a passing certificate is not evidence on its own. These tests inject one
defect at a time into the real
:class:`~omni_mercury_engine.core.three_r.fusion.OmniAvaEquation` and require
the probe to catch each, which is what makes its ``ok`` mean something.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from omni_mercury_engine.core.three_r import fusion as fusion_module
from omni_mercury_engine.tools import oae_dimensionality_probe as probe

_ARGS = argparse.Namespace(batch=8, dim=16, trials=8, seed=0)


def _collect() -> Any:
    return probe._collect(_ARGS)


def test_the_healthy_equation_certifies_and_influence_matches_the_table() -> None:
    """Served behaviour agrees with the published weights to float precision."""
    cert = _collect()
    assert cert.status == "ok", cert.body["failures"]
    assert cert.body["weights"]["sum"] == pytest.approx(1.0, abs=1e-9)
    for stream, record in cert.body["influence"].items():
        assert record["measured_influence"] == pytest.approx(
            record["declared_weight"], abs=1e-9
        ), f"{stream}: measured influence disagrees with its declared weight"
        assert record["measured_influence"] > 0.0, f"{stream}: no influence on the fused score"


def test_a_dead_stream_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """The defect class that actually shipped: a stream carrying no signal.

    ``H(omega)`` was a saturated constant 0.0 on recorded NAB data while
    holding 27.6% of the OAE's weight. This is the gate that would have
    reported it.
    """
    real = fusion_module.OmniAvaEquation

    class DeadResonance(real):  # type: ignore[valid-type,misc]
        def compute(self, **kwargs: Any) -> Any:
            kwargs["resonance_score"] = 0.0
            return super().compute(**kwargs)

    monkeypatch.setattr(fusion_module, "OmniAvaEquation", DeadResonance)
    cert = _collect()
    assert cert.status == "fail"
    assert any("resonance_score is a dead stream" in f for f in cert.body["failures"]), cert.body[
        "failures"
    ]


def test_served_arithmetic_drifting_from_the_declared_table_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The table stays canonical while the served path mis-scales a stream."""
    real = fusion_module.OmniAvaEquation

    class ArithmeticDrift(real):  # type: ignore[valid-type,misc]
        def compute(self, **kwargs: Any) -> Any:
            kwargs["recursion_score"] = kwargs["recursion_score"] * 0.5
            return super().compute(**kwargs)

    monkeypatch.setattr(fusion_module, "OmniAvaEquation", ArithmeticDrift)
    cert = _collect()
    assert cert.status == "fail"
    assert any("drifted from the published weights" in f for f in cert.body["failures"]), cert.body[
        "failures"
    ]


def test_recoupling_eta_to_the_fused_score_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """``eta ** Phi`` must not scale the fused score.

    A per-call scalar that does cannot reorder a single batch, but it makes
    scores from calls at different eta incomparable -- so the decoupling is
    what lets the score be thresholded at all.
    """
    real = fusion_module.OmniAvaEquation

    class EtaCoupled(real):  # type: ignore[valid-type,misc]
        def compute(self, **kwargs: Any) -> Any:
            result = super().compute(**kwargs)
            object.__setattr__(
                result, "fusion_score", result.fusion_score * self.ethical_compliance_threshold
            )
            return result

    monkeypatch.setattr(fusion_module, "OmniAvaEquation", EtaCoupled)
    cert = _collect()
    assert cert.status == "fail"
    assert cert.body["failures"], "eta coupling went unreported"


def test_eta_is_measured_as_decoupled_on_the_shipped_equation() -> None:
    """The positive half of the above, on the real object."""
    cert = _collect()
    assert cert.body["eta_span"] == pytest.approx(
        0.0, abs=1e-9
    ), f"the fused score moved {cert.body['eta_span']} across the permitted eta range"
    assert len(cert.body["eta_sweep"]) > 1, "an eta sweep of one point measures nothing"


def test_the_probe_reads_the_real_equation_rather_than_reimplementing_it() -> None:
    """Vacuity guard: the certificate must depend on the shipped class.

    The replaced tool was immune to every defect above because it reimplemented
    the arithmetic inline. If this ever stops tracking the real object, the
    injections above would pass and the probe would be theatre again.
    """
    real = fusion_module.OmniAvaEquation

    class Broken(real):  # type: ignore[valid-type,misc]
        def compute(self, **kwargs: Any) -> Any:
            result = super().compute(**kwargs)
            object.__setattr__(result, "fusion_score", 0.5)
            return result

    import unittest.mock

    with unittest.mock.patch.object(fusion_module, "OmniAvaEquation", Broken):
        cert = _collect()
    assert cert.status == "fail", (
        "a fused score pinned to a constant certified clean; the probe is not "
        "measuring the shipped equation"
    )
