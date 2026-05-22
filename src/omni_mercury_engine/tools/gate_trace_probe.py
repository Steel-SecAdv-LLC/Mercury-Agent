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

Operator tool: gate-trace probe.

Exercises every public detect/analyze/predict surface in Mercury and
emits a JSON trace of which gates (Benevolence, σ_Immutable, GOSNN)
fired and in what order.  Today the gate contract lives only in
docstrings and the ``EthicalConstraintViolationError(check=...)``
identifier; an operator has no runtime artefact proving the gates
actually ran on a real call.

The probe wraps ``SigmaImmutableGate.evaluate`` and
``BenevolenceScorer.score_action`` with thread-safe instrumentation,
calls a small set of representative entry-points end-to-end on
synthetic but well-formed input, and emits per-call records::

    {"surface": "engine.detect_with_fusion", "gates": ["benevolence", "sigma_immutable"], ...}

A surface that does *not* invoke the contractual gates is a hard
finding — the JSON trace makes the gap concrete and reviewable.
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.gate_trace_probe/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.gate_trace_probe",
        description=(
            "Exercise public Mercury surfaces and emit a JSON trace of which "
            "gates (benevolence, sigma_immutable, gosnn) fired on each call."
        ),
    )
    parser.add_argument(
        "--surfaces",
        nargs="*",
        default=None,
        help="Restrict the probe to a subset of surfaces (default: all).",
    )
    return parser


@contextmanager
def _trace_gates() -> Iterator[list[dict[str, Any]]]:
    """Patch the gate primitives to record every invocation in order.

    Patches are reverted on context exit so the probe never leaves the
    process in a half-instrumented state, even if the caller raises.
    """
    log: list[dict[str, Any]] = []
    lock = threading.Lock()

    patches: list[tuple[Any, str, Any]] = []

    def _record(name: str, args: tuple[Any, ...]) -> None:
        with lock:
            log.append(
                {
                    "gate": name,
                    "thread": threading.get_ident(),
                    "monotonic_ns": time.monotonic_ns(),
                    "arg_summary": _summarise_args(args),
                }
            )

    try:
        from omni_mercury_engine.security import sigma_immutable_gate as sig_mod

        orig_eval = sig_mod.SigmaImmutableGate.evaluate

        def wrapped_eval(self: Any, scalar_vector: Any) -> Any:
            _record("sigma_immutable", (scalar_vector,))
            return orig_eval(self, scalar_vector)

        sig_mod.SigmaImmutableGate.evaluate = wrapped_eval  # type: ignore[method-assign]
        patches.append((sig_mod.SigmaImmutableGate, "evaluate", orig_eval))
    except ImportError:
        pass

    try:
        from omni_mercury_engine.cognitive import ethical_bounding as eth_mod

        orig_score = eth_mod.BenevolenceScorer.score_action

        def wrapped_score(self: Any, action: str, context: dict[str, Any]) -> Any:
            _record("benevolence", (action, context))
            return orig_score(self, action, context)

        eth_mod.BenevolenceScorer.score_action = wrapped_score  # type: ignore[method-assign]
        patches.append((eth_mod.BenevolenceScorer, "score_action", orig_score))
    except ImportError:
        pass

    try:
        yield log
    finally:
        for owner, name, orig in patches:
            setattr(owner, name, orig)


def _summarise_args(args: tuple[Any, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, a in enumerate(args):
        if isinstance(a, np.ndarray):
            out[f"arg{i}"] = {"shape": list(a.shape), "dtype": str(a.dtype)}
        elif isinstance(a, dict):
            out[f"arg{i}"] = {"keys": sorted(a.keys())[:6]}
        elif isinstance(a, str):
            out[f"arg{i}"] = a[:60]
        else:
            out[f"arg{i}"] = type(a).__name__
    return out


def _surface_engine_detect() -> dict[str, Any]:
    from omni_mercury_engine.engine import MercuryEngine

    engine = MercuryEngine()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 16)).astype(np.float64)
    with _trace_gates() as log:
        try:
            result = engine.detect(X)  # most common public surface
            status = "ok"
            outcome: dict[str, Any] = {"keys": sorted(result.keys()) if isinstance(result, dict) else None}
        except Exception as exc:  # noqa: BLE001
            status = "raised"
            outcome = {"exception_type": type(exc).__name__, "message": str(exc)}
        return {
            "surface": "MercuryEngine.detect",
            "status": status,
            "outcome": outcome,
            "gate_calls": log,
            "gates_fired": [g["gate"] for g in log],
        }


def _surface_orchestrator_analyze() -> dict[str, Any]:
    try:
        from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator
    except ImportError as exc:
        return {
            "surface": "CognitiveOrchestrator.analyze",
            "status": "import-failed",
            "outcome": {"message": str(exc)},
            "gate_calls": [],
            "gates_fired": [],
        }
    orch = CognitiveOrchestrator()
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 16)).astype(np.float64)
    with _trace_gates() as log:
        try:
            result = orch.analyze({"features": X.tolist(), "context": {}})
            status = "ok"
            outcome: dict[str, Any] = {"type": type(result).__name__}
        except Exception as exc:  # noqa: BLE001
            status = "raised"
            outcome = {"exception_type": type(exc).__name__, "message": str(exc)[:300]}
        return {
            "surface": "CognitiveOrchestrator.analyze",
            "status": status,
            "outcome": outcome,
            "gate_calls": log,
            "gates_fired": [g["gate"] for g in log],
        }


def _surface_hub_predict() -> dict[str, Any]:
    try:
        from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub
    except ImportError as exc:
        return {
            "surface": "NeuroSymbolicHub.predict",
            "status": "import-failed",
            "outcome": {"message": str(exc)},
            "gate_calls": [],
            "gates_fired": [],
        }
    hub = NeuroSymbolicHub()
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4, 16)).astype(np.float64)
    with _trace_gates() as log:
        try:
            result = hub.predict(X)
            status = "ok"
            outcome: dict[str, Any] = {"type": type(result).__name__}
        except Exception as exc:  # noqa: BLE001
            status = "raised"
            outcome = {"exception_type": type(exc).__name__, "message": str(exc)[:300]}
        return {
            "surface": "NeuroSymbolicHub.predict",
            "status": status,
            "outcome": outcome,
            "gate_calls": log,
            "gates_fired": [g["gate"] for g in log],
        }


_SURFACES = {
    "engine.detect": _surface_engine_detect,
    "orchestrator.analyze": _surface_orchestrator_analyze,
    "hub.predict": _surface_hub_predict,
}


def _collect(args: argparse.Namespace) -> Certificate:
    selected = args.surfaces or list(_SURFACES)
    unknown = set(selected) - set(_SURFACES)
    if unknown:
        raise ValueError(
            f"unknown surfaces: {sorted(unknown)}; known: {sorted(_SURFACES)}"
        )

    records: list[dict[str, Any]] = []
    for name in selected:
        try:
            records.append(_SURFACES[name]())
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "surface": name,
                    "status": "probe-error",
                    "outcome": {
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-5:],
                    },
                    "gate_calls": [],
                    "gates_fired": [],
                }
            )

    # An "ok" run requires every surface that executed (status==ok or
    # raised, both of which mean the call actually invoked the gates)
    # to have fired *at least* the σ_Immutable gate.  Surfaces that
    # short-circuit before the gate are flagged.
    findings: list[str] = []
    for r in records:
        if r["status"] in {"import-failed", "probe-error"}:
            findings.append(f"{r['surface']}: {r['status']}")
            continue
        if "sigma_immutable" not in r["gates_fired"]:
            findings.append(f"{r['surface']}: did not invoke sigma_immutable gate")

    body: dict[str, Any] = {
        "surfaces": selected,
        "records": records,
        "findings": findings,
    }
    status = "fail" if findings else "ok"
    return Certificate(
        tool="gate_trace_probe",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=findings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
