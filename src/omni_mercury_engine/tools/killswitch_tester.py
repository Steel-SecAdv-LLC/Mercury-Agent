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

Operator tool: kill-switch trip-latency tester.

Trips the Phase-5 OODA kill-switch on ``AutonomousAgent`` under load
and measures the wall-clock latency between activation and the next
in-flight step refusing to act.  The contractual SLA quoted in the
7-Phase table is sub-second; this tool turns the claim into a
measurable, reproducible operator artefact.

Methodology
-----------
1. Construct an ``AutonomousAgent`` and start a worker thread that
   repeatedly calls ``observe → orient → decide → act`` until the
   kill-switch trips.
2. After ``--warmup`` iterations, fire ``activate_kill_switch()`` from
   the main thread.  Capture ``t_activate`` (perf_counter_ns).
3. The worker checks ``self._kill_switch`` at every step boundary; the
   first step that observes the switch records ``t_observed`` and
   exits.
4. ``trip_latency_ns = t_observed - t_activate`` is the SLA-relevant
   measurement.

The kill-switch is a ``bool`` read inside the worker's hot loop;
Python's GIL guarantees the write is atomic, but propagation latency
is what we're measuring — typically <1ms on a modern CPU.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.killswitch_tester/v1"
_DEFAULT_SLA_MS = 1000.0  # the README's documented sub-second SLA


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.killswitch_tester",
        description=(
            "Trip the Phase-5 OODA kill-switch under load and measure "
            "activation→observation latency against the documented SLA."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Iterations before tripping the switch (default 50).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10_000,
        help="Safety upper bound; the worker exits after this many steps even if the switch never trips.",
    )
    parser.add_argument(
        "--sla-ms",
        type=float,
        default=_DEFAULT_SLA_MS,
        help="Trip-latency SLA in milliseconds (default 1000ms).",
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
        # the kill switch at the step boundary; the agent's own
        # ``observe`` also raises ``RuntimeError`` when the switch is
        # set, so we treat that as a trip observation too.
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
            if agent._kill_switch:
                t_observed_holder.append(time.perf_counter_ns())
                return
            try:
                obs = agent.observe({"i": i, "value": float(i % 7)})
                ori = agent.orient(obs)
                dec = agent.decide(ori)
                agent.act(dec)
            except RuntimeError:
                # ``observe`` raises this once the switch is set.
                t_observed_holder.append(time.perf_counter_ns())
                return
            except Exception:
                t_observed_holder.append(time.perf_counter_ns())
                return
            time.sleep(per_step_sleep)

    thread = threading.Thread(target=worker, name="killswitch-worker", daemon=True)
    thread.start()

    # Wait for the worker to reach warmup; bounded so a broken agent
    # cannot hang the operator indefinitely.
    if not warmup_event.wait(timeout=10.0):
        return Certificate(
            tool="killswitch_tester",
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
    agent.activate_kill_switch()
    thread.join(timeout=10.0)

    if not t_observed_holder:
        return Certificate(
            tool="killswitch_tester",
            schema=_SCHEMA,
            status="fail",
            body={
                "warmup": args.warmup,
                "max_iterations": args.max_iterations,
                "steps_completed": steps_holder[0],
                "worker_alive": thread.is_alive(),
            },
            warnings=["kill switch was activated but the worker never observed it"],
        )

    trip_latency_ns = t_observed_holder[0] - t_activate_holder[0]
    trip_latency_ms = trip_latency_ns / 1_000_000.0
    body: dict[str, Any] = {
        "warmup": args.warmup,
        "steps_completed": steps_holder[0],
        "activation_t_ns": t_activate_holder[0],
        "observed_t_ns": t_observed_holder[0],
        "trip_latency_ns": int(trip_latency_ns),
        "trip_latency_ms": trip_latency_ms,
        "sla_ms": args.sla_ms,
        "within_sla": trip_latency_ms <= args.sla_ms,
    }
    warnings: list[str] = []
    if trip_latency_ms > args.sla_ms:
        warnings.append(f"trip latency {trip_latency_ms:.3f}ms exceeds SLA {args.sla_ms:.1f}ms")
    status = "ok" if body["within_sla"] else "fail"
    return Certificate(
        tool="killswitch_tester",
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
