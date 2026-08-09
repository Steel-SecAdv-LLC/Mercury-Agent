#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mutation-test ``tests/pillars/``: break each control, prove the suite notices.

Why this exists
---------------

``tests/pillars/`` asserts that Mercury's enforced controls hold. A suite written
alongside the controls it guards has correlated blind spots — it can pass because
the control works, or because the test never really looked. Reading the tests
cannot tell those apart. Killing the control can.

Each mutation below disables or inverts exactly one enforced control, then runs
the pillar module that claims to observe it. The expected outcome is a **failing**
suite. A mutation that SURVIVES is a vacuous test and a real finding: the pillar
it belongs to is not actually observed, and the control could regress silently.

This complements the σ_Immutable hot-path mutation gate in CI
(``.github/workflows/mutation-testing.yml``), which measures kill rate over
``security/sigma_immutable_gate.py``. This one covers the *pillar* controls.

Usage::

    python scripts/mutate_pillar_controls.py            # all mutations
    python scripts/mutate_pillar_controls.py --list     # names only

Exit code is non-zero if any mutation survives.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutation:
    """One control-disabling patch and the pillar module that should catch it.

    Attributes:
        name: Human-readable description of what is broken.
        pillar: The property that would silently regress if this survives.
        patch: Python executed as a pytest plugin before collection, monkeypatching
            the control at its import site.
        selection: pytest selection expected to go red.
    """

    name: str
    pillar: str
    patch: str
    selection: str


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="harm gate always returns ALLOW",
        pillar="non-maleficence",
        patch="""
        import omni_mercury_engine.cognitive.decision_gate as g
        from omni_mercury_engine.cognitive.ethical_bounding import (
            WeaponsDisposition, WeaponsRiskAssessment, HazardDomain, OperationalIntent,
        )

        def _always_allow(action, context=None, harm_classifier=None):
            return WeaponsRiskAssessment(
                disposition=WeaponsDisposition.ALLOW,
                hazard_domain=HazardDomain.NONE,
                intent_tier=OperationalIntent.NONE,
                confidence=0.0,
                signals=(),
            )

        g.assess_weapons_uplift = _always_allow
        """,
        selection="tests/pillars/test_non_maleficence.py",
    ),
    Mutation(
        name="decision boundary swallows its own refusal",
        pillar="non-maleficence (fail-closed)",
        patch="""
        import omni_mercury_engine.cognitive.decision_gate as g

        _real = g.enforce_decision_boundary

        def _never_raise(subject, *, advisory_scorer=None, harm_classifier=None):
            try:
                return _real(
                    subject, advisory_scorer=advisory_scorer, harm_classifier=harm_classifier
                )
            except Exception:
                class _Verdict:
                    requires_provenance = False
                    benevolence = None
                    assessment = None
                    subject = None
                return _Verdict()

        g.enforce_decision_boundary = _never_raise
        """,
        selection="tests/pillars/test_non_maleficence.py",
    ),
    Mutation(
        name="grave-harm axis disabled",
        pillar="non-maleficence (harm to a person)",
        patch="""
        import omni_mercury_engine.cognitive.ethical_bounding as eb

        eb._direct_harm_present = lambda *a, **k: False
        eb._euphemism_harm_present = lambda *a, **k: False
        """,
        selection="tests/pillars/test_non_maleficence.py",
    ),
    Mutation(
        name="GATED_BOUNDARY capability contract becomes a no-op decorator",
        pillar="non-maleficence (routing)",
        patch="""
        import omni_mercury_engine.agentic.capabilities.contract as c

        def _noop_contract(*invariants, **kwargs):
            def deco(fn):
                return fn
            return deco

        c.capability_contract = _noop_contract
        """,
        selection="tests/pillars/test_non_maleficence.py",
    ),
    Mutation(
        name="decision records become mutable",
        pillar="control (immutable ledger)",
        patch="""
        import omni_mercury_engine.decision.record as r

        for _obj in list(vars(r).values()):
            if isinstance(_obj, type):
                try:
                    _obj.__setattr__ = object.__setattr__
                except Exception:
                    pass
        """,
        selection="tests/pillars/test_control.py",
    ),
    Mutation(
        name="governor tripwire becomes reversible",
        pillar="corrigibility",
        patch="""
        import omni_mercury_engine.agentic.subagents.governor as gv

        for _obj in list(vars(gv).values()):
            if isinstance(_obj, type):
                for _method in ("halt", "trip", "engage_tripwire"):
                    if hasattr(_obj, _method):
                        setattr(_obj, _method, lambda self, *a, **k: None)
        """,
        selection="tests/pillars/test_corrigibility.py",
    ),
    Mutation(
        name="response policy emits a destructive verb",
        pillar="non-maleficence (non-destructive responses)",
        patch="""
        import omni_mercury_engine.decision.response as rp

        _real_plan = rp.ResponsePolicy.plan

        def _destructive(self, disposition, **kwargs):
            plan = _real_plan(self, disposition, **kwargs)
            poisoned = plan.rationale + " delete and destroy the host"
            try:
                object.__setattr__(plan, "rationale", poisoned)
            except Exception:
                try:
                    plan.rationale = poisoned
                except Exception:
                    pass
            return plan

        rp.ResponsePolicy.plan = _destructive
        """,
        selection="tests/pillars/test_non_maleficence.py",
    ),
)


def run_mutation(mutation: Mutation) -> str:
    """Apply one mutation and report how the pillar suite responded.

    Args:
        mutation: The control-disabling patch to apply.

    Returns:
        ``"KILLED"`` when a test actually failed (pytest exit 1 -- the mutation
        was caught by an assertion), ``"SURVIVED"`` when every test still passed
        (exit 0), or ``"ERROR"`` for any other pytest exit (2 = collection/usage
        error, 5 = no tests collected, ...). ``ERROR`` is NOT a kill: the suite
        never ran a pass-to-fail transition, so a mutation that breaks import or
        collection would otherwise be miscounted as observed and inflate the
        kill rate. The mutation gate must fail loudly on it instead.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin = Path(tmpdir) / "mercury_mutant_plugin.py"
        plugin.write_text(textwrap.dedent(mutation.patch))

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [tmpdir, str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
        )
        proc = subprocess.run(  # noqa: S603 - fixed argv; selection is a literal in MUTATIONS
            [
                sys.executable,
                "-m",
                "pytest",
                mutation.selection,
                "-q",
                "-p",
                "no:randomly",
                "-p",
                "mercury_mutant_plugin",
                "--timeout=900",
                "-x",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    # Only pytest exit 1 -- a test ran and FAILED -- is a genuine kill. Exit 0
    # is a survivor; any other code (2 collection/usage error, 3 internal, 5 no
    # tests collected) means the transition we are testing for never happened,
    # so it must not be scored as a kill.
    if proc.returncode == 0:
        result = "SURVIVED"
    elif proc.returncode == 1:
        result = "KILLED"
    else:
        result = "ERROR"
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(no output)"
    label = {"KILLED": "KILLED  ", "SURVIVED": "SURVIVED", "ERROR": "ERROR   "}[result]
    print(f"  {label}  [{mutation.pillar}] {mutation.name}")
    if result == "ERROR":
        print(
            f"              pytest exit {proc.returncode} (suite did not run a kill) -- {tail[:80]}"
        )
    else:
        print(f"              {tail[:100]}")
    return result


def main() -> int:
    """Run every mutation and fail if any survived.

    Returns:
        Process exit code: 0 when every mutation was killed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print mutation names and exit")
    args = parser.parse_args()

    if args.list:
        for mutation in MUTATIONS:
            print(f"{mutation.pillar}: {mutation.name}")
        return 0

    print("=" * 78)
    print("Mutation test: break each pillar control, confirm tests/pillars/ catches it")
    print("=" * 78)

    results = [(m, run_mutation(m)) for m in MUTATIONS]
    killed = [m for m, r in results if r == "KILLED"]
    survivors = [m for m, r in results if r == "SURVIVED"]
    errored = [m for m, r in results if r == "ERROR"]

    print()
    print(f"kill rate: {len(killed)}/{len(MUTATIONS)} (killed by a genuine test failure)")
    if errored:
        print("ERRORS -- the suite could not be run for these mutations, so coverage is")
        print("UNPROVEN (a collection/usage error is not a kill):")
        for mutation in errored:
            print(f"  - [{mutation.pillar}] {mutation.name}")
    if survivors:
        print("SURVIVORS -- these pillars are not actually observed:")
        for mutation in survivors:
            print(f"  - [{mutation.pillar}] {mutation.name}")
    if survivors or errored:
        return 1
    print("Every disabled control was caught. No vacuous pillar coverage found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
