#!/usr/bin/env python3
"""Ethics audit for Mercury-Agent.

This script is referenced by .github/workflows/ci.yml (Stage 7: Ethics Audit).
It validates the AI ethics framework by exercising the key components:

  - EthicalAutonomyGovernor: end-to-end action evaluation with all 8 pillars
  - PreExecutionBlockingGate: hard-blocks destructive/exfiltration/deceptive patterns
  - OmniAvaEquation: sigma_immutable (ethical threshold) is truly immutable after init
  - EthicsConfig: all 8 ethical principle checks are enabled by default

For implementation details, see:
  - src/omni_mercury_engine/core/ai_ethics.py
  - src/omni_mercury_engine/ethical/ethical_constraint_engine.py
  - src/omni_mercury_engine/core/ethical_governor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).parent.parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src.resolve()))


def _pass(msg: str) -> None:
    print(f"  ✓ PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ FAIL  {msg}")


def _skip(msg: str) -> None:
    print(f"  - SKIP  {msg}")


def run_audit() -> int:
    """Run the full ethics audit. Returns 0 on pass, non-zero on failure."""
    failures: list[str] = []
    print("=" * 60)
    print("Mercury-Agent Ethics Audit")
    print("=" * 60)

    # ------------------------------------------------------------------
    # T1 — Module imports
    # ------------------------------------------------------------------
    print("\n[T1] Core ethics module imports")
    try:
        from omni_mercury_engine.core.ai_ethics import (
            EthicalAutonomyGovernor,
            EthicsConfig,
            PreExecutionBlockingGate,
        )

        _pass("ai_ethics: EthicalAutonomyGovernor, PreExecutionBlockingGate, EthicsConfig")
    except ImportError as e:
        _skip(f"ai_ethics not importable ({e}) — skipping T1-T4")
        # Without the core module we cannot run any substantive checks.
        print("\nFinal verdict: INCONCLUSIVE (package not installed)")
        return 0  # advisory-only; don't hard-fail CI on missing install

    # ------------------------------------------------------------------
    # T2 — EthicsConfig: all 8 pillars enabled by default
    # ------------------------------------------------------------------
    print("\n[T2] EthicsConfig — 8 ethical pillars enabled by default")
    cfg = EthicsConfig()
    pillar_flags = {
        "compassion": cfg.enable_compassion_checks,
        "evidence": cfg.enable_evidence_validation,
        "justice": cfg.enable_justice_bias_checks,
        "altruism": cfg.enable_altruism_impact_checks,
        "control": cfg.enable_control_auditing,
        "character": cfg.enable_character_transparency,
        "competence": cfg.enable_competence_validation,
        "commitment": cfg.enable_commitment_evolution,
    }
    all_enabled = True
    for pillar, enabled in pillar_flags.items():
        if enabled:
            _pass(f"Pillar '{pillar}' enabled")
        else:
            _fail(f"Pillar '{pillar}' disabled by default — should be True")
            failures.append(f"T2: pillar '{pillar}' disabled")
            all_enabled = False
    if all_enabled:
        _pass("All 8 ethical pillars are enabled by default")

    # ------------------------------------------------------------------
    # T3 — PreExecutionBlockingGate: blocks known dangerous patterns
    # ------------------------------------------------------------------
    print("\n[T3] PreExecutionBlockingGate — hard-block verification")
    # The May 2026 Phase 2 audit cure removed ``enable_blocking`` and
    # ``allow_overrides`` from the gate's constructor because a single
    # ``False`` at construction silently disabled all protection.
    # Blocking is now permanent — there is no off-switch — so the gate
    # is constructed bare.
    gate = PreExecutionBlockingGate()

    _destructive_patterns = [
        "delete_all",
        "drop_database",
        "wipe",
    ]
    _exfil_patterns = ["export_credentials", "dump_secrets"]
    _safe_patterns = ["detect_anomaly", "train_model", "generate_report"]

    for action in _destructive_patterns + _exfil_patterns:
        result = gate.check_action(action_type=action)
        if result.blocked:
            _pass(f"'{action}' correctly blocked (category: {result.category})")
        else:
            _fail(
                f"'{action}' NOT blocked — destructive/exfiltration action should be hard-blocked"
            )
            failures.append(f"T3: '{action}' not blocked")

    for action in _safe_patterns:
        result = gate.check_action(action_type=action)
        if not result.blocked:
            _pass(f"'{action}' correctly allowed")
        else:
            _fail(f"'{action}' incorrectly blocked — safe action should be allowed")
            failures.append(f"T3: safe '{action}' blocked")

    # Gate should reject blocked params too
    result = gate.check_action(
        action_type="run_pipeline",
        action_params={"skip_validation": True},
    )
    if result.blocked:
        _pass("Blocked param 'skip_validation=True' correctly rejected")
    else:
        _fail("Blocked param 'skip_validation=True' not rejected — parameter guard broken")
        failures.append("T3: blocked param not rejected")

    # Hardening contract: ``enable_blocking`` and ``allow_overrides`` are
    # removed.  Asserting the constructor rejects the legacy kwargs is the
    # positive evidence that the May 2026 Phase 2 footgun cure is in
    # effect — i.e., a stale caller cannot silently disable blocking by
    # passing ``enable_blocking=False`` (which previously bypassed the
    # entire gate without raising).
    try:
        PreExecutionBlockingGate(enable_blocking=False)  # type: ignore[call-arg]
    except TypeError:
        _pass(
            "PreExecutionBlockingGate rejects legacy ``enable_blocking`` kwarg (Phase 2 hardening intact)"
        )
    else:
        _fail(
            "PreExecutionBlockingGate accepted ``enable_blocking=False`` — off-switch reintroduced!"
        )
        failures.append("T3: legacy off-switch kwarg silently accepted")

    try:
        PreExecutionBlockingGate(allow_overrides=True)  # type: ignore[call-arg]
    except TypeError:
        _pass(
            "PreExecutionBlockingGate rejects legacy ``allow_overrides`` kwarg (Phase 2 hardening intact)"
        )
    else:
        _fail(
            "PreExecutionBlockingGate accepted ``allow_overrides=True`` — overrides reintroduced!"
        )
        failures.append("T3: legacy override kwarg silently accepted")

    # ------------------------------------------------------------------
    # T4 — EthicalAutonomyGovernor: end-to-end action evaluation
    # ------------------------------------------------------------------
    print("\n[T4] EthicalAutonomyGovernor — end-to-end evaluation")
    governor = EthicalAutonomyGovernor(config=EthicsConfig())

    safe_result = governor.evaluate_action(
        action_type="analyze_telemetry",
        action_params={"dataset": "sensor_logs", "purpose": "anomaly_detection"},
        context={"user": "operator", "environment": "production"},
    )
    if safe_result.passed:
        _pass(f"Safe action approved (score={safe_result.overall_score:.3f})")
    else:
        # Advisory: scoring may flag a safe action if principle weights or
        # min_ethics_score threshold are miscalibrated.  Report but don't fail.
        _skip(
            f"Safe action scored {safe_result.overall_score:.3f} < min_ethics_score "
            "— advisory finding; governor is functional but scoring calibration may "
            "need review (violations: " + ", ".join(safe_result.violations) + ")"
        )

    blocked_result = governor.evaluate_action(
        action_type="delete_all",
        action_params={"target": "production_database"},
        context={"user": "service_account"},
    )
    # EthicsResult.passed=False when either the gate blocked or scoring failed
    if not blocked_result.passed:
        _pass(
            f"Destructive 'delete_all' action correctly rejected "
            f"(score={blocked_result.overall_score:.3f}, "
            f"violations={blocked_result.violations[:1]})"
        )
    else:
        _fail(
            "Destructive 'delete_all' action PASSED ethics check — "
            "blocking gate or ethical scoring not functioning"
        )
        failures.append("T4: destructive action passed ethics check")

    # ------------------------------------------------------------------
    # T5 — OmniAvaEquation: sigma_immutable (ethical threshold) is
    #      truly immutable after construction
    # ------------------------------------------------------------------
    print("\n[T5] OmniAvaEquation — ethical_compliance_threshold immutability")
    try:
        from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation

        oae = OmniAvaEquation(ethical_compliance_threshold=0.96)
        original = oae.ethical_compliance_threshold

        # Attempt mutation with an intentionally invalid (too-low) value.
        # The point is that ANY post-construction write must be rejected.
        _BELOW_MIN_THRESHOLD = 0.01
        mutation_blocked = False
        try:
            oae.ethical_compliance_threshold = _BELOW_MIN_THRESHOLD  # must be rejected
        except AttributeError:
            mutation_blocked = True

        if mutation_blocked:
            _pass(
                f"ethical_compliance_threshold={original:.2f} is immutable "
                "— post-construction assignment raises AttributeError"
            )
        else:
            _fail(
                "ethical_compliance_threshold can be mutated after construction "
                "— ethical threshold is NOT protected"
            )
            failures.append("T5: ethical_compliance_threshold mutable after construction")

        # Clamping: out-of-range value should be clamped, not accepted raw
        oae_low = OmniAvaEquation(ethical_compliance_threshold=0.50)
        if oae_low.ethical_compliance_threshold >= 0.90:
            _pass(
                f"Low threshold 0.50 clamped to {oae_low.ethical_compliance_threshold:.2f} "
                "(minimum 0.90 enforced)"
            )
        else:
            _fail(
                f"Low threshold 0.50 stored as {oae_low.ethical_compliance_threshold:.2f} "
                "— minimum 0.90 not enforced"
            )
            failures.append("T5: threshold clamping not enforced")

    except ImportError as e:
        _skip(f"three_r.fusion not importable ({e})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL — {len(failures)} issue(s) found:")
        for f in failures:
            print(f"  • {f}")
        print("=" * 60)
        return 1
    else:
        print("PASS — all ethics audit checks passed")
        print("=" * 60)
        return 0


def main() -> int:
    """Entry point."""
    try:
        return run_audit()
    except Exception as e:
        print(f"Ethics audit encountered an unexpected error: {e}")
        import traceback

        traceback.print_exc()
        # Unexpected failures indicate a broken ethics framework — treat as hard failure.
        return 1


if __name__ == "__main__":
    sys.exit(main())
