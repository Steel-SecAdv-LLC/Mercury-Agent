# Decision / Abstention / Response Layer

> Turning calibrated confidence into autonomy with a conscience:
> a closed **identify → interpret → decide → deter → verify** loop with a
> first-class *don't-know* gate.

`omni_mercury_engine.decision` is the layer that lets Mercury **act** on a
detection — proportionately, reversibly, and only when it is honestly sure —
instead of emitting a bare `is_anomaly` boolean that no one closes the loop on.
It is built entirely on substrate the engine already had (calibrated
probabilities, conformal prediction sets, the three-state honesty contract, and
the dual hard ethical gate), so every part of it is verifiable, not aspirational.

---

## Why it exists

Before this layer, the engine's decision boundary was always two-valued:

```python
"is_anomaly": bool(float(anomaly_prob_val) > threshold)   # engine.py
```

Two outcomes, no honest deferral, and **no response**. The conformal machinery
even computed an `abstain` bit — and then discarded it at the boundary. There
was no closed loop from a detection to a proportionate action, and nothing that
could say *"I don't know — get more evidence"* instead of guessing.

This layer adds exactly those two missing halves:

1. **Decision with abstention** — a typed verdict that can be `ABSTAIN`.
2. **Response ("deter")** — a graded, reversible-by-default, ethically-gated
   action, recorded to an append-only audit ledger.

---

## The pipeline

```
            detect_with_fusion / score_fusion_conformal
                              │   calibrated P(anomaly) (+ conformal set)
                              ▼
   ConfidenceSignal  ──►  AbstentionPolicy  ──►  Decision {POSITIVE|NEGATIVE|ABSTAIN}
   (interpret)             (decide)               │  state ∈ {GROUNDED, UNAVAILABLE}
                                                  ▼
                              ResponsePlanner  ──►  ResponseAction  (proportionate)
                              (deter)               │
                                                    ▼
                              ResponseActuator  ──►  ResponseOutcome {APPLIED|DEFERRED|BLOCKED|NOOP}
                              (fail-closed gate)     │
                                                     ▼
                              AuditLedger  ──►  LoopResult  (verifiable certificate)
                              (verify)
```

### 1. Interpret — `ConfidenceSignal`

A source-agnostic carrier: a calibrated `P(anomaly)` and, when available, the
conformal label set over `{0 = normal, 1 = anomaly}`. The adapters read the
engine surfaces that exist today and are **forward-compatible** with PR #278's
richer calibration: if a result carries `calibrated_probabilities` (Beta-MCA) or
a `reconciled_operating_point`, those are preferred automatically.

### 2. Decide — `AbstentionPolicy` (the don't-know gate)

Deterministic and pure. Two paths:

| Input | Verdict | Three-state |
|---|---|---|
| conformal singleton `{1}` | `POSITIVE` | `GROUNDED` |
| conformal singleton `{0}` | `NEGATIVE` | `GROUNDED` |
| conformal `{0, 1}` (both admissible) | `ABSTAIN` | `UNAVAILABLE` |
| conformal `{}` (empty / atypical) | `ABSTAIN` (+ `novelty`) | `UNAVAILABLE` |
| no set, `p ≥ positive` | `POSITIVE` | `GROUNDED` |
| no set, `p ≤ negative` | `NEGATIVE` | `GROUNDED` |
| no set, `negative < p < positive` | `ABSTAIN` | `UNAVAILABLE` |

The conformal path is preferred because the set carries a **distribution-free
coverage guarantee** — the policy *honours* the set rather than overriding it
with a point threshold. Set `require_conformal=True` to abstain whenever no
guaranteed set is present (the strict posture for high-stakes domains).

**Honesty invariant:** a detection abstention is always `ThreeState.UNAVAILABLE`
(decidable in principle — more data/coverage could settle it), **never**
`UNDECIDABLE`. That state is reserved by the cross-repo contract for claims with
no decision procedure in principle, which an anomaly call is not. Every
abstention records *what would decide it*.

### 3. Deter — `ResponsePlanner` + `ResponseActuator`

The planner maps `(verdict, severity)` to one action on an ascending ladder:

| Verdict | Severity | Action | Tier | Reversible | Human auth? |
|---|---|---|---|---|---|
| `NEGATIVE` | — | continue monitoring | `MONITOR` | ✓ | — |
| `ABSTAIN` | low/med | reduce uncertainty | `GATHER_EVIDENCE` | ✓ | — |
| `ABSTAIN` | high+ | inform a human | `NOTIFY` | ✓ | — |
| `POSITIVE` | low/med | flag for attention | `NOTIFY` | ✓ | — |
| `POSITIVE` | substantial/high | reversible quarantine/throttle | `SOFT_CONTAIN` | ✓ | — |
| `POSITIVE` | severe/critical | escalate to a human | `ESCALATE` | ✗ | **required** |

The actuator enforces the safety contract, in order:

1. `NONE` tier → `NOOP`.
2. **An abstention may never actuate a deterrent** (`SOFT_CONTAIN` / `ESCALATE`)
   — enforced in the planner *and* re-checked here (defence in depth) → `DEFERRED`.
3. **Fail-closed ethical gate** runs before any effectful action; a veto →
   `BLOCKED`. The engine binds this to its own benevolence + σ_Immutable boundary.
4. Escalatory or irreversible actions without an explicit `Authorization` →
   `DEFERRED`.
5. Otherwise the registered handler runs → `APPLIED`.

Default handlers are **safe, recordable placeholders** — a deployment plugs real
effectors (a rate-limiter, a quarantine queue, a CAP alert) via
`ResponseActuator.register_handler`. The layer's guarantees hold whatever the
effector does.

### 4. Verify — `AuditLedger`

Every pass becomes a frozen, JSON-serialisable `LoopResult` appended to an
append-only ledger: what was decided, why, the response disposition, and the
honesty state. `ledger.summary()` reports verdict / status / state counts and the
abstention rate. The optional `feedback` sink is the omnidirectional seam to push
outcomes back to calibration, RL, or a human queue.

---

## Usage

### On the engine (closed loop on a real detection)

```python
from omni_mercury_engine.engine import OmniMercuryEngine
from omni_mercury_engine.decision import Authorization

engine = OmniMercuryEngine(mode="fusion", device="cpu")
engine.fit_fusion(X_train, y_train)
engine.calibrate_fusion_conformal(X_cal, y_cal, coverage=0.9)   # enables the conformal path

result = engine.decide_and_respond(sample, domain="network_security")
result["decision"]   # {'verdict': 'abstain', 'state': 'unavailable', 'reason': ..., ...}
result["response"]   # {'action': {...}, 'status': 'deferred', 'ethical_gate_passed': True}
result["loop"]       # the full JSON-serialisable certificate

# A severe/critical anomaly escalates and DEFERS until a human signs off:
engine.decide_and_respond(
    sample,
    domain="network_security",
    authorization=Authorization(authority="soc-operator-7", reason="confirmed incident"),
)
```

### Standalone (no detector — testable anywhere)

```python
from omni_mercury_engine.decision import (
    AbstentionPolicy, ConfidenceSignal, DecisionResponseLoop, permit_all_gate,
)

loop = DecisionResponseLoop(ethical_gate=permit_all_gate)   # bind a real gate in production
loop.step(ConfidenceSignal(0.55, prediction_set=(0, 1), coverage=0.9))   # -> ABSTAIN / gather-evidence
loop.step(ConfidenceSignal(0.98, prediction_set=(1,),   coverage=0.9))   # -> POSITIVE / escalate (deferred)
loop.ledger.summary()
```

---

## Design commitments (all enforced in tests)

- **Honest abstention** is first-class and maps to the shared three-state contract.
- **Verifiable-only:** decisions are pure functions of their inputs; every pass is
  JSON-serialisable with full provenance.
- **Reversible-by-default, fail-closed:** abstentions never deter; irreversible /
  escalatory actions require explicit human authorization; every effectful action
  passes the ethical gate first.
- **Forward-compatible:** consumes #278's richer calibration when present, depends
  on none of it.

See `tests/decision/` for the truth tables and the safety-contract tests, and
`docs/CAPABILITY_VS_VISION_MATRIX.md` for where this layer sits in the roadmap.
