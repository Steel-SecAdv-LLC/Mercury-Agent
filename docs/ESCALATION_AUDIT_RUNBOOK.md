# Escalation & Harm-Gate Audit Runbook

Operator guide for the human-in-the-loop escalation control and the durable,
tamper-evident audit trail behind Mercury's harm gate. Companion to
[`HARM_POLICY.md`](HARM_POLICY.md) (the policy) and
[`INSTALLATION.md`](INSTALLATION.md) (Tier 0 configuration).

## 1. What escalation is

An `ESCALATE` disposition marks a genuine gray-zone request that a *human* could
authorize — a licensed engineer's production-adjacent query, or an accretion
pattern worth a second look. It is **not** a silent allow and **not** a plain
refusal: it is routed to an injectable reviewer under a bounded-autonomy cap, and
every outcome is durably audited.

- Control: `omni_mercury_engine.cognitive.escalation.EscalationBroker`.
- Reviewer type: `HumanReviewCallback = Callable[[EscalationRecord], bool]`.
- **Fail-closed**: with no reviewer wired, or on a reviewer error, the escalation
  is *denied* (the gray-zone request is refused).
- **Bounded autonomy**: at most `max_approvals` escalations may be approved per
  session; beyond the ceiling every escalation is denied regardless of the
  reviewer, so a compromised or looping reviewer cannot rubber-stamp without
  bound.

## 2. Wiring a reviewer

The general-capability surface accepts the hook directly:

```python
from omni_mercury_engine.agentic.capabilities import GeneralAssistant
from omni_mercury_engine.cognitive.escalation import EscalationRecord

def reviewer(record: EscalationRecord) -> bool:
    # Consult a queue / approval webhook / SOAR action / interactive prompt.
    # Return True to authorize the gray-zone request, False to refuse.
    return approve_via_your_system(record.query, record.hazard_domain)

assistant = GeneralAssistant(
    escalation_reviewer=reviewer,
    escalation_max_approvals=3,   # bounded-autonomy ceiling for this session
)
```

With no `escalation_reviewer`, escalations are denied fail-closed — the safe
default. An approved gray-zone answer is additionally treated as
*provenance-required* (it must be emitted from cited sources).

## 3. The audit trail

Every harm-gate decision (`refused`, `escalated`, `approved`, `escalation_denied`,
`allow_provenance`, accretion detections, capability-contract enforcement) is
written durably by `record_gate_decision`.

### 3.1 Primary sink (JSON-Lines)

- Path: `MERCURY_GATE_AUDIT_LOG`, else `<repo>/artifacts/audit/gate_decisions.jsonl`
  in a source checkout, else `<XDG_STATE_HOME>/mercury-agent/audit/…` when
  installed.
- One decision per line, flushed and `fsync`ed. Free-text fields are capped so a
  full procedure is never stored verbatim.
- Fail-safe: an audit write failure is logged and swallowed — auditing can never
  break the control it records. Disable entirely (not recommended) with
  `MERCURY_GATE_AUDIT_DISABLED=1`.

Tail it live:

```bash
export MERCURY_GATE_AUDIT_LOG=/var/lib/mercury/audit/gate_decisions.jsonl
tail -f "$MERCURY_GATE_AUDIT_LOG" | jq '{ts, decision, source, disposition, hazard_domain, reason}'
```

### 3.2 Tamper-evident sink (hash-chained, opt-in)

Set `MERCURY_GATE_AUDIT_SECURELOG=1` to *also* forward each decision to the
hash-chained, PII-masking `SecureAuditLogger`. Verification is two-fold and
constant-time: each event's stored hash must recompute from its own content
(so an **edit** to a hashed field is caught, unforgeable without the HMAC key),
and each event's back-pointer must match the prior event's hash (so a
**deletion or reordering** breaks the chain).

> **Cross-process verification needs a stable key.** The hash recompute uses the
> HMAC key the log was written with; the default is a per-process ephemeral key.
> To verify a log written by another process, configure a stable key
> (`AMA_MASTER_SEED`, or `configure_audit_logger(hmac_key=...)`).

> **Note.** `MERCURY_GATE_AUDIT_LOG` steers only the plain JSONL. The secure
> sink writes to its own directory — set `MERCURY_SECURE_AUDIT_DIR` to point it
> (default `./audit_logs/audit.jsonl`). The sink is (re)configured at most once,
> so the hash chain is never reset on the hot path.

Verify integrity (and detect tampering):

```python
from omni_mercury_engine.security.secure_audit_logging import get_audit_logger

logger = get_audit_logger()
logger.flush()
ok, message = logger.verify_log_integrity()   # or pass an explicit Path
print(ok, message)   # (True, "Log integrity verified (N events)") — or a broken-chain report
```

A `False` result names the indices where the chain breaks. Treat any broken
chain as a security incident: the audit store was mutated out of band.

## 4. Capability contracts

The general capabilities are wrapped by `@capability_contract`
(`agentic/capabilities/contract.py`), which enforces three invariants at runtime
and audits any breach before repairing it to the safe result:

- `fail_closed` — an unexpected error becomes the capability's typed
  honest-negative, never an unguarded exception.
- `cite_or_refuse` — emitted content on a provenance-required topic must carry
  citations, or it is downgraded to a refusal.
- `monotone_harm` — output never contains a gate-unsafe span; adding harmful
  input can only increase redaction.

Breaches surface in the audit log with `decision` values
`capability_fail_closed` / `capability_cite_or_refuse` / `capability_monotone_harm`.

## 5. Rolling corpus & calibration

The weapons-gate evaluation corpus is versioned and rolled forward with a
stdlib-only CLI; calibration is measured out-of-fold so the fit and the metric
never see the same row.

```bash
# Ingest new labeled examples (validated, deduped, kept disjoint from the
# held-out adversarial set); bumps the content-hash version + manifest.
python scripts/ingest_weapons_gate_corpus.py --add new_cases.jsonl

# Integrity gate (corpus matches manifest hash + class balance).
python scripts/ingest_weapons_gate_corpus.py --check

# Out-of-fold ECE/Brier + rolling-origin + held-out adversarial recall.
PYTHONPATH=src python benchmarks/rolling_corpus_eval.py            # print metrics
PYTHONPATH=src python benchmarks/rolling_corpus_eval.py --check    # regression gate
PYTHONPATH=src python benchmarks/rolling_corpus_eval.py --update   # re-pin baseline + report
```

Artifacts: `benchmarks/weapons_gate_corpus_manifest.json` (version + hash),
`benchmarks/weapons_gate_oof_baseline.json` (pinned OOF metrics),
`benchmarks/weapons_gate_calibration_report.md` (human-readable report).

> The held-out adversarial recall is intentionally reported: with lexical-only
> features it is low (paraphrase/obfuscation slip a lexicon), which is exactly the
> gap the served-model `ci/meaning-level` lane closes. Run that lane with a real
> local reasoning model (see [`INSTALLATION.md`](INSTALLATION.md) Tier 0) to
> measure the meaning-level false-negative budget.

## 6. Quick triage

| Symptom | Check |
|---|---|
| Gray-zone requests always refused | No `escalation_reviewer` wired (fail-closed default), or `max_approvals` ceiling reached this session. |
| No audit entries | `MERCURY_GATE_AUDIT_DISABLED=1` set, or `MERCURY_GATE_AUDIT_LOG` points at an unwritable path (write failure is logged, not raised). |
| `verify_log_integrity` returns False | Secure log mutated out of band — treat as an incident; the named indices locate the break. |
| Meaning-level lane skips instead of measuring | No real served model detected; set `MERCURY_CI_REQUIRE_REAL_CLASSIFIER=1` and serve a local model at `MERCURY_MODEL_ENDPOINT`. |
| `import omni_mercury_engine` raises at startup | PQC gate: AMA missing/partial or version ≠ `3.2.0` — rebuild with `scripts/build_ama_native.sh`. |
