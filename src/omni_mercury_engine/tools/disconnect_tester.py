# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: Mercury/AMA Disconnect response-latency tester.

Engages the Phase-5 OODA Mercury/AMA Disconnect on ``AutonomousAgent``
under load and measures the wall-clock latency between engagement and
the next in-flight step standing down.  The contractual SLA quoted in
the 7-Phase table is sub-second; this tool turns the claim into a
measurable, reproducible operator artefact.

Methodology
-----------
1. Construct an ``AutonomousAgent`` and start a worker thread that
   repeatedly calls ``observe → orient → decide → act`` until the
   Disconnect is engaged.
2. After ``--warmup`` iterations, fire ``activate_disconnect()`` from
   the main thread.  Capture ``t_activate`` (perf_counter_ns).
3. The worker checks ``self._disconnect_engaged`` at every step
   boundary; the first step that observes it records ``t_observed``
   and stands down.
4. ``disconnect_latency_ns = t_observed - t_activate`` is the
   SLA-relevant measurement.

The Disconnect is a ``bool`` read inside the worker's hot loop;
Python's GIL guarantees the write is atomic, but propagation latency
is what we're measuring — typically <1ms on a modern CPU.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.disconnect_tester/v1"
_DEFAULT_SLA_MS = 1000.0  # the README's documented sub-second SLA


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.disconnect_tester",
        description=(
            "Engage the Phase-5 OODA Mercury/AMA Disconnect under load and "
            "measure engagement→observation latency against the documented SLA."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Iterations before engaging the Disconnect (default 50).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10_000,
        help="Safety upper bound; the worker exits after this many steps even if the Disconnect never engages.",
    )
    parser.add_argument(
        "--sla-ms",
        type=float,
        default=_DEFAULT_SLA_MS,
        help="Disconnect response-latency SLA in milliseconds (default 1000ms).",
    )
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    try:
        from omni_mercury_engine.cognitive.autonomous_agent import OODAAgent
    except ImportError as exc:
        from omni_mercury_engine.tools._base import DependencyMissing

        raise DependencyMissing(f"OODAAgent import failed (missing extras?): {exc}") from exc

    agent = OODAAgent()

    warmup_event = threading.Event()
    t_activate_holder: list[int] = []
    t_observed_holder: list[int] = []
    steps_holder = [0]

    def worker() -> None:
        # Hot loop mirroring the OODA contract.  Each iteration checks
        # the Disconnect at the step boundary; the agent's own
        # ``observe`` also raises ``RuntimeError`` when it is engaged,
        # so we treat that as an observation of engagement too.
        i = 0
        # A tiny per-iteration sleep keeps the loop slow enough that
        # the main thread reliably wins the warmup race on fast CPUs
        # without dominating the latency measurement (1µs ≪ SLA ms).
        per_step_sleep = 1e-6
        while i < args.max_iterations:
            i += 1
            steps_holder[0] = i
            if i == args.warmup:
                warmup_event.set()
            if agent._disconnect_engaged:
                t_observed_holder.append(time.perf_counter_ns())
                return
            try:
                obs = agent.observe({"i": i, "value": float(i % 7)})
                ori = agent.orient(obs)
                dec = agent.decide(ori)
                agent.act(dec)
            except RuntimeError:
                # ``observe`` raises this once the Disconnect is engaged.
                t_observed_holder.append(time.perf_counter_ns())
                return
            except Exception:
                t_observed_holder.append(time.perf_counter_ns())
                return
            time.sleep(per_step_sleep)

    thread = threading.Thread(target=worker, name="disconnect-worker", daemon=True)
    thread.start()

    # Wait for the worker to reach warmup; bounded so a broken agent
    # cannot hang the operator indefinitely.
    if not warmup_event.wait(timeout=10.0):
        return Certificate(
            tool="disconnect_tester",
            schema=_SCHEMA,
            status="fail",
            body={
                "warmup": args.warmup,
                "steps_completed": steps_holder[0],
                "worker_alive": thread.is_alive(),
            },
            warnings=["worker never reached the warmup checkpoint"],
        )

    t_activate_holder.append(time.perf_counter_ns())
    agent.activate_disconnect()
    thread.join(timeout=10.0)

    if not t_observed_holder:
        return Certificate(
            tool="disconnect_tester",
            schema=_SCHEMA,
            status="fail",
            body={
                "warmup": args.warmup,
                "max_iterations": args.max_iterations,
                "steps_completed": steps_holder[0],
                "worker_alive": thread.is_alive(),
            },
            warnings=["Mercury/AMA Disconnect was engaged but the worker never observed it"],
        )

    disconnect_latency_ns = t_observed_holder[0] - t_activate_holder[0]
    disconnect_latency_ms = disconnect_latency_ns / 1_000_000.0
    body: dict[str, Any] = {
        "warmup": args.warmup,
        "steps_completed": steps_holder[0],
        "activation_t_ns": t_activate_holder[0],
        "observed_t_ns": t_observed_holder[0],
        "disconnect_latency_ns": int(disconnect_latency_ns),
        "disconnect_latency_ms": disconnect_latency_ms,
        "sla_ms": args.sla_ms,
        "within_sla": disconnect_latency_ms <= args.sla_ms,
    }
    warnings: list[str] = []
    if disconnect_latency_ms > args.sla_ms:
        warnings.append(
            f"disconnect latency {disconnect_latency_ms:.3f}ms exceeds SLA {args.sla_ms:.1f}ms"
        )
    status = "ok" if body["within_sla"] else "fail"
    return Certificate(
        tool="disconnect_tester",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
