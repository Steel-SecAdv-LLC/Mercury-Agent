# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury intelligence layer: closed-loop learning + decision geometry.

This package is the *learning and geometry* layer that sits on top of the Tier-0
safety foundation (the harm gate, the calibrator, the oracle-validated
verifiers, the rolling corpus). It adds six streams, each shipping a **measured
value metric** (baseline + target, see :mod:`.value_metrics`):

* :mod:`.self_consistency` -- N-sample reasoning-path sampling and a
  disagreement-based uncertainty signal that the calibrator consumes.
* :mod:`.cascade` -- calibrated-uncertainty routing between a cheap template path
  and a heavy model path, with compute-cost / latency instrumentation.
* :mod:`.verifier_loop` -- routes applicable generative claims through the
  symbolic verifiers and blocks emission when an oracle refutes a claim.
* :mod:`.provenance` -- provenance carried as a typed companion through the
  pipeline and enforced at the output boundary (the timeboxed fallback), with a
  forward path to a fully unrepresentable-without-provenance type.
* :mod:`.red_team` -- an adversarial co-training generator whose surviving gate
  bypasses are appended to ``corpus/pending`` with triage metadata.
* :mod:`.feedback_loop` -- an accept-gated closed loop: a durable labeled queue,
  a signed/audited retrain trigger, an OOF/adversarial regression gate that
  blocks a regressing (or poisoned) candidate model, and a one-command rollback.

Submodules are imported lazily via :func:`__getattr__` so pulling one stream (or
just :mod:`.value_metrics`) never eagerly imports the heavy ML / gate paths of
the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omni_mercury_engine.intel.value_metrics import (
    VALUE_METRICS,
    Direction,
    ValueMetric,
    get_value_metric,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from omni_mercury_engine.intel import (
        cascade,
        feedback_loop,
        provenance,
        red_team,
        self_consistency,
        verifier_loop,
    )

_LAZY_SUBMODULES = frozenset(
    {
        "cascade",
        "feedback_loop",
        "provenance",
        "red_team",
        "self_consistency",
        "verifier_loop",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily import a stream submodule on first attribute access (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VALUE_METRICS",
    "Direction",
    "ValueMetric",
    "get_value_metric",
]
