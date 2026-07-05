# Provenance-as-Type: Migration Plan and Timebox Decision

The design record for `omni_mercury_engine.intel.provenance`: the endpoint we
want (provenance as an *unrepresentable-without-it* type), the three-week timebox
that forced a decision, what shipped, and the phased path from the shipped
fallback to the full type. Companion to
[`ESCALATION_AUDIT_RUNBOOK.md`](ESCALATION_AUDIT_RUNBOOK.md) (the harm-gate audit
trail this module writes into) and [`HARM_POLICY.md`](HARM_POLICY.md).

## 1. The goal

Provenance as an **unrepresentable-without-it type**: a value on a
provenance-required (hazardous) topic that *cannot exist in the type system*
unless it carries its sources. Not a runtime check that can be forgotten — a
compile-time guarantee, enforced whole-pipeline by mypy. A stage that drops
provenance would fail to type-check against the next stage's parameter, so an
uncited hazardous emission is not a refusal at the boundary but a **build
failure** long before it runs.

That is the endpoint. It is a deep, multi-module refactor: every capability
output, every pipeline stage signature, and every emission point has to move to
the typed companion at once for the guarantee to hold end-to-end.

## 2. The timebox and the decision

The work was timeboxed to **three weeks**. A full pipeline conversion to the
type endpoint exceeds that box — it touches `web_research`, `text_synthesis`,
`assistant`, and the aggregate gate, and cannot land behind a flag as a single
mergeable unit.

**Decision (per governance): ship the fallback.** The boundary-fallback delivers
~80% of the safety value — nothing hazardous leaves uncited — and does **not
block merge**. The stricter type endpoint is seeded now and reached
incrementally (Section 5). Strength is selected at runtime by the
`MERCURY_PROVENANCE_MODE` env var:

```bash
# Shipped default: provenance carried as metadata, enforced at the boundary.
export MERCURY_PROVENANCE_MODE=boundary-fallback

# Stricter seed: the boundary only accepts a Provenanced payload on a
# provenance-required topic; a bare value is refused at emission.
export MERCURY_PROVENANCE_MODE=type
```

An unset or unrecognized value resolves to `boundary-fallback`
(`ProvenanceMode.from_env`, warned on unknown input). `type` mode is the seed of
the endpoint, not the endpoint: it enforces at the boundary at runtime, whereas
the endpoint enforces across the pipeline at compile time.

## 3. What ships now

All in `src/omni_mercury_engine/intel/provenance.py`.

### 3.1 The provenance record — `Provenance`

Frozen dataclass carried alongside a value:

- `origin: ProvenanceOrigin` — `ORACLE_VERIFIED` (strongest) `> HUMAN >
  EXTRACTIVE > MODEL_GENERATED > SYNTHETIC` (weakest), by `.rank`.
- `sources: tuple[str, ...]`, `verified: bool`, `notes: str`.
- `has_citations()` — true when at least one non-empty source is attached.
- `is_adequate(require_verified=False)` — adequacy requires **both** an
  attributed origin (`ORACLE_VERIFIED` / `HUMAN` / `EXTRACTIVE`) **and** at least
  one citation; an unattributed `MODEL_GENERATED` / `SYNTHETIC` value is never
  adequate even when cited (this closes the fail-open where a fabricated citation
  on the weakest origin would launder synthetic content). `require_verified=True`
  additionally demands `verified` sources.
- `merge(other)` — the pipeline-join rule: **weakest origin wins**, sources are
  unioned (order-stable, deduped), and `verified` holds **only if both** inputs
  were verified. A stage can never launder an unverified input into a verified
  output.

### 3.2 The typed companion — `Provenanced[T]`

Frozen `Generic[T]` pairing `value: T` with `provenance: Provenance`. This is the
type seed:

- `map(fn, step="")` — transform the value, carrying (and optionally annotating)
  provenance through a pipeline step.
- `combine(other, fn)` — join two provenanced values, merging provenance
  weakest-wins.

Because a stage takes and returns `Provenanced`, provenance cannot be silently
dropped between stages — a stage that forgets it fails to type-check against the
next stage's `Provenanced` parameter. That property is what Section 5 leverages.

### 3.3 The type-seed boundary — `require_provenanced`

- `ensure_provenanced(value, provenance)` — lift a bare value into a
  `Provenanced` companion.
- `require_provenanced(payload)` — assert `payload` is a `Provenanced`; raises
  `TypeError` on a bare value. A function that accepts only `Provenanced` makes a
  bare value *unrepresentable at that boundary* — the seed the migration grows.

### 3.4 The output boundary — `enforce_at_boundary`

```python
from omni_mercury_engine.intel.provenance import enforce_at_boundary

decision = enforce_at_boundary(
    payload,                       # a Provenanced, or a bare value + provenance=
    text=candidate_text,           # decides required-ness via the gate
    provenance=prov,               # for a bare payload
    require_verified=False,        # demand independently-checked sources
)
```

Returns a `BoundaryDecision` (`emitted`, `payload`, `reason`, `enforced`,
`provenance_required`, `provenance`, `mode`). When provenance is **required** and
missing or inadequate, the emission is withheld: `emitted=False`, `payload`
replaced by `REFUSAL_NOTICE`, and the refusal durably audited. In `type` mode a
bare (non-`Provenanced`) value on a required topic is refused outright, before
the adequacy check. When not required, the value passes through unchanged.

### 3.5 What "provenance-required" means — `provenance_required_for`

`provenance_required_for(text)` **reuses the shipped weapons/mass-casualty gate**
(`assess_weapons_uplift` in `omni_mercury_engine.cognitive.ethical_bounding`): a
topic dispositioned `ALLOW_PROVENANCE` — or any refusal disposition (`ESCALATE`,
`REFUSE_REDACT`, `HARD_REFUSE`) — is exactly the set that must be
source-attributed. This does **not** invent a second notion of what needs
citations; it inherits the gate's. The gate is fail-closed, so an internal gate
error yields `True` (require provenance), never `False`.

## 4. Value metric

`boundary_provenance_enforcement_rate` (stream `provenance` in
`intel/value_metrics.py`): the fraction of provenance-required, unprovenanced
emissions that were withheld. **Baseline 0.0, target 1.0** — every
provenance-required emission lacking adequate provenance is refused/redacted.
Computed by `boundary_enforcement_rate(emissions, ...)` over `(payload,
provenance)` pairs; an emission that *does* carry adequate provenance is
legitimately emittable and is excluded from the denominator.

## 5. Migration to the type endpoint (the residual ~20%)

The residual over the shipped fallback is the compile-time, whole-pipeline
**unrepresentability** guarantee. Reach it in five phases. **Each phase is
independently shippable and monotonic** — it never weakens the current boundary;
the worst case at any point is the shipped runtime refusal.

**Phase 1 — create provenance at the source.** Adopt `Provenanced[T]` at
capability outputs — `WebResearcher` (`web_research`), `ExtractiveSynthesizer`
(`text_synthesis`), `GeneralAssistant` (`assistant`) — so a value is born with
its `Provenance` (`EXTRACTIVE`/`ORACLE_VERIFIED` for cited fetches,
`MODEL_GENERATED` for unattributed synthesis). No signature downstream changes
yet; the boundary still enforces at runtime.

**Phase 2 — thread `Provenanced` through stage signatures.** Change each pipeline
stage to take and return `Provenanced`, carrying provenance with `map()` and
joining with `combine()`. From here a stage that drops provenance **fails mypy**
against the next stage's `Provenanced` parameter — the guarantee becomes local
and compile-time, one stage boundary at a time.

**Phase 3 — replace bare-value boundary calls with `require_provenanced`.** At
every emission point, swap the bare-value `enforce_at_boundary` call for
`require_provenanced(payload)` so the emission site accepts only a `Provenanced`.
A bare value at an emission point is now a `TypeError` at construction, not a
late boundary refusal.

**Phase 4 — turn on mypy strictness on the required paths.** The repo already
runs strict mypy (`[tool.mypy]` in `pyproject.toml`: `disallow_untyped_defs`,
`disallow_untyped_calls`, `disallow_any_generics`). With Phases 1–3 landed, a
bare value on a provenance-required path is a **type error, not a runtime
refusal** — the emission is unrepresentable, caught in CI:

```bash
PYTHONPATH=src mypy src/omni_mercury_engine/intel/provenance.py \
                    src/omni_mercury_engine/agentic/capabilities/
```

**Phase 5 — delete the runtime boundary-fallback.** Once the type fully subsumes
the boundary — every provenance-required emission provably carries provenance by
construction — retire the runtime path: `boundary-fallback` mode, the
`enforce_at_boundary` withhold branch, and `REFUSAL_NOTICE`. `MERCURY_PROVENANCE_MODE`
collapses to a single guaranteed strength. **Do not start Phase 5 until Phase 4
is green across the whole pipeline** — the runtime boundary is the safety net and
is removed last.

## 6. Risk and governance

- **The fallback must never block merge.** `boundary-fallback` is the shipped
  default and ships ~80% of the value; the migration is additive on top.
- **The runtime boundary stays until the type fully subsumes it.** Phases 1–4 are
  monotonic — they only add compile-time guarantees over the runtime one; the
  runtime withhold is deleted only in Phase 5, and only after Phase 4 is green.
- **Every withheld emission is audited.** `enforce_at_boundary` records each
  refusal via `record_gate_decision` (decision `provenance_withheld`,
  disposition `refuse_redact`) through `omni_mercury_engine.cognitive.gate_audit`
  — the same durable, tamper-evident trail the escalation runbook describes.
  Migration progress is legible: as phases land, withhold events fall toward zero
  because bare values stop reaching the boundary.

## 7. Quick reference

| Concern | Symbol / knob |
|---|---|
| Strength selector | `MERCURY_PROVENANCE_MODE` = `boundary-fallback` (default) \| `type` |
| Carried record | `Provenance` — `has_citations()`, `is_adequate(require_verified=)`, `merge()` |
| Typed companion | `Provenanced[T]` — `map()`, `combine()` |
| Type-seed assert | `require_provenanced(payload)` → `TypeError` on a bare value |
| Lift a bare value | `ensure_provenanced(value, provenance)` |
| Output boundary | `enforce_at_boundary(...) -> BoundaryDecision` |
| Required-ness rule | `provenance_required_for(text)` (reuses the weapons gate, fail-closed) |
| Value metric | `boundary_provenance_enforcement_rate` (0.0 → 1.0) via `boundary_enforcement_rate` |
| Audit sink | `record_gate_decision(decision="provenance_withheld", ...)` |
