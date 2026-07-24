# Closed-Loop Retrain Runbook

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-24.

Operator guide for Mercury's **accept-gated closed feedback loop**: the gated
retrain and one-command rollback behind the harm gate. Package:
`omni_mercury_engine.intel.feedback_loop`. Companion to
[`ESCALATION_AUDIT_RUNBOOK.md`](ESCALATION_AUDIT_RUNBOOK.md) (the audit trail this
loop feeds from) and [`WEAPONS_GATE_EVALUATION.md`](WEAPONS_GATE_EVALUATION.md)
(the rolling corpus + OOF metrics this loop's gate reuses).

## 1. What the loop is

Gate-audit events and red-team survivors are a learning signal **and** a
poisoning surface: the same stream that teaches the gate what it missed can be
used to teach it to miss on purpose. The loop closes that gap by learning only
from **human-verified labels**, and admits **no model update** without all three
of:

1. a verifying **signature** (an unforgeable, queue-bound authorization),
2. a human **sign-off** (`human_verified=True`),
3. a candidate that passes the **OOF/adversarial regression gate**.

Any one missing, fail-closed: the baseline model stands. Accepted runs write a
**staged** candidate only — promotion to production is a separate, deliberate
operator step.

## 2. Environment & configuration

| Variable | Purpose | Default / fail mode |
|---|---|---|
| `MERCURY_FEEDBACK_QUEUE_URL` | Durable labeled queue sink. A `file://` URL or a bare path. | `<repo>/artifacts/feedback/labeled_queue.jsonl` (`DEFAULT_QUEUE_PATH`). |
| `MERCURY_RETRAIN_TRIGGER_SECRET` | HMAC key that signs/authorizes a retrain. | Unset ⇒ authorization impossible (fail-closed). |

- The queue URL is resolved by `resolve_queue_path`. Only the empty/`file`
  schemes are supported; a non-`file` scheme raises `NotImplementedError` rather
  than silently dropping labels — a loud, fail-closed extension point for
  SQS/PubSub/Kafka adapters.
- The signing secret is read by `secret_from_env` (`SECRET_ENV =
  "MERCURY_RETRAIN_TRIGGER_SECRET"`). With no secret, `sign_trigger` raises
  `RuntimeError` and `verify_trigger` returns `False`: a retrain cannot be
  authorized at all.

```bash
export MERCURY_FEEDBACK_QUEUE_URL=file:///var/lib/mercury/feedback/labeled_queue.jsonl
export MERCURY_RETRAIN_TRIGGER_SECRET="$(openssl rand -hex 32)"   # keep out of logs/artifacts
```

## 3. The three gates

`staged_refit` checks these in order, each fail-closed. Every outcome is audited
(§7).

### 3.1 Signed trigger (`trigger.py`)

A `RetrainTrigger` is an HMAC-SHA256 signature over a canonical payload that
**binds the exact queue snapshot** the retrain is authorized for. The signed
fields are `queue_hash`, `corpus_version`, `requested_by`, `n_examples`, and a
caller `nonce`.

`verify_trigger(trigger, secret=, expected_queue_hash=, audit=)` returns `True`
only if a secret is configured, the HMAC matches in constant time, and (when
`expected_queue_hash` is supplied) the trigger's `queue_hash` still equals the
live queue hash. `staged_refit` always passes the **live** `queue.snapshot_hash()`
as `expected_queue_hash`, so a signature **replayed against a changed queue**
(e.g. one poisoned after signing) fails with a queue-hash mismatch. Every
verification is recorded as `retrain_trigger_verified` or
`retrain_trigger_rejected`.

### 3.2 Human verification (`retrain.py`)

`staged_refit(..., human_verified=True)` is required. Human-in-the-loop sign-off
must happen **before** any model update; the pipeline refuses with
`human_verified=False` even when the trigger verifies.

This gate is anchored upstream in labeling: a `LabeledExample` with an empty
`reviewer` id is refused at construction — anonymous or malformed labels never
enter the queue.

### 3.3 OOF/adversarial regression gate (`regression_gate.py`)

`evaluate_candidate(examples, base_rows=None)` fits a **candidate** on
`base corpus + queue examples` and a **baseline** on the base corpus alone, then
compares out-of-fold calibration (`oof_ece`, `oof_brier`, `oof_auroc`) and
held-out `adversarial_recall` under the **Tier-0 rolling-corpus `MARGINS`** —
the same folds, holdout, and margins the `ci/rolling-corpus-eval` lane enforces,
so a candidate can never pass a weaker bar than the foundation requires:

```
MARGINS = {"oof_ece": 0.05, "oof_brier": 0.03, "oof_auroc": -0.03, "adversarial_recall": -0.05}
```

A positive margin caps how far a metric may **rise** (ECE/Brier); a negative
margin caps how far it may **fall** (AUROC/recall). A NaN metric fails closed.
`gate_reports` returns a `RegressionVerdict`; a regressing **or** poisoned
candidate has `accepted=False`, is refused, and the baseline stands.

## 4. Operator procedure

Run under `PYTHONPATH=src`. The base corpus / OOF evaluator requires the AMA/PQC
backend (see [`INSTALLATION.md`](INSTALLATION.md) Tier 0).

**1. Ingest audit events and label them.** Parse a gate-audit record, then attach
a human-verified label. Direct overrides and triaged red-team survivors go
through `override_to_example`.

```python
from omni_mercury_engine.intel.feedback_loop import (
    ingest_audit_event, apply_human_label, override_to_example,
    DurableLabeledQueue, sign_trigger, staged_refit,
)

event = ingest_audit_event(record)                       # dict from the gate-audit JSONL
ex1 = apply_human_label(event, label="offensive", reviewer="analyst-7",
                        reason="actionable chemical synthesis")
ex2 = override_to_example("how do vaccines trigger immunity?", label="benign",
                          reviewer="analyst-7", reason="benign explainer")
```

Labels are `"offensive"` or `"benign"`; `reviewer` is mandatory and non-empty.

**2. Enqueue into the durable queue.**

```python
queue = DurableLabeledQueue()          # or DurableLabeledQueue("file:///path/queue.jsonl")
queue.enqueue(ex1)                      # -> True if newly stored (deduped, fsync'd)
queue.enqueue_many([ex2])              # -> count newly stored
```

**3. Sign a trigger bound to the current queue snapshot.** Sign against the exact
hash you intend to retrain on; any later enqueue changes the hash and invalidates
the signature.

```python
qhash = queue.snapshot_hash()
trigger = sign_trigger(
    queue_hash=qhash, corpus_version="v1.7", requested_by="analyst-7",
    n_examples=len(queue), nonce="2026-07-04T00:00Z#1",
)   # secret= defaults to MERCURY_RETRAIN_TRIGGER_SECRET
```

**4. Run the accept-gated staged refit.**

```python
result = staged_refit(
    queue, trigger, human_verified=True,
    staging_dir="artifacts/closed_loop/staging",
    corpus_version="v1.7",
)   # secret= defaults to env
print(result.as_dict())
```

On acceptance, `staged_refit` writes to `staging_dir`, keyed by
`version = "<corpus_version>+<queue_hash[:12]>"`:

- the staged candidate model — `candidate_<version>.json` (logistic weights +
  `feature_order`), and
- the retrain artifact — `retrain_<version>.json` (metrics, trigger fingerprint,
  requester, regression verdict), stamped
  `"STAGED ONLY -- promotion to production is a separate operator step."`

It also registers the candidate as `active` in the staging `ModelRegistry`. These
are **staged**; nothing reaches production until an operator promotes it
deliberately.

A refusal returns `accepted=False` with the failing gate in `reason` (bad/replayed
signature, missing human verification, empty queue, or regression-gate
violations).

## 5. Rollback (one command)

If a staged candidate misbehaves, roll the staging registry's `active` pointer
back to `previous` in a single audited, atomic swap:

```bash
python scripts/mercury_retrain_rollback.py --staging-dir artifacts/closed_loop/staging
python scripts/mercury_retrain_rollback.py --staging-dir artifacts/closed_loop/staging --status
```

`--status` prints the `active`/`previous` pointers and exits without changing
anything. The rollback itself is `ModelRegistry.rollback` (also
`rollback_staging(staging_dir)`): the registry keeps exactly two live pointers
plus an append-only history, `registry.json` is written atomically (temp-file +
`os.replace`), and the swap is recorded as `model_rollback`. Fail-safe: with no
`active` or no `previous` there is nothing to restore, so it reports
`rolled_back=False` rather than corrupting state. Exit code is `0` on a swap, `1`
otherwise.

## 6. Data-poisoning defense

The attack: enqueue mislabeled examples — offensive text labeled `benign` — so
the candidate fit shifts its decision boundary to allow what the gate should
block. Human gating alone would let this through, because a compromised or
mistaken reviewer signs it off.

The defense is Gate 3. Folding poisoned rows into the fit drives out-of-fold
`oof_ece`/`oof_brier`/`oof_auroc` and held-out `adversarial_recall` past the
`MARGINS`; `evaluate_candidate` returns `accepted=False`, `staged_refit` refuses,
and the baseline stands. This is the measured value:
`VALUE_METRICS["closed_feedback_loop"].metric = "poisoned_candidate_block_rate"`,
**target 1.0** (every poisoned candidate blocked).

## 7. Audit

Every loop decision is durably recorded via
`omni_mercury_engine.cognitive.gate_audit.record_gate_decision`:

| Decision | Emitted when |
|---|---|
| `retrain_trigger_verified` / `retrain_trigger_rejected` | a trigger is verified (§3.1) |
| `retrain_accepted` / `retrain_refused` | a staged refit passes / fails a gate |
| `model_registered` | a staged candidate is registered `active` |
| `model_rollback` | `active` is swapped back to `previous` (§5) |

See [`ESCALATION_AUDIT_RUNBOOK.md`](ESCALATION_AUDIT_RUNBOOK.md) §3 for the sink
paths, the tamper-evident hash-chained option, and how to tail/verify the log.

## 8. Demo & CI

End-to-end staging demo (requires the AMA/PQC backend):

```bash
PYTHONPATH=src python scripts/closed_loop_demo.py
PYTHONPATH=src python scripts/closed_loop_demo.py --poisoned   # show the gate BLOCK a poisoned candidate
```

The demo ingests an audit event, labels + enqueues it, signs a queue-bound
trigger, runs the gated staged refit, and exercises register + one-command
rollback. `--poisoned` enqueues mislabeled (offensive→benign) examples and shows
the regression gate refuse the poisoned candidate.

The `ci/closed-loop-integration` lane runs this staging demo end-to-end and
asserts both a gated retrain artifact (`retrain_<version>.json`) and the
poisoned-candidate block.

## 9. Quick triage

| Symptom | Check |
|---|---|
| `sign_trigger` raises `RuntimeError` | `MERCURY_RETRAIN_TRIGGER_SECRET` unset — no signing key (fail-closed). |
| `retrain_trigger_rejected` with "queue hash mismatch" | Queue changed since signing (enqueue after `sign_trigger`); re-`snapshot_hash()` and re-sign. |
| Refit refused, `human_verified=False` | Sign-off gate: pass `human_verified=True` to `staged_refit`. |
| Refit refused with regression violations | Candidate regressed OOF/adversarial vs baseline (possibly poisoned); baseline stands — inspect the queue labels. |
| `LabeledExample` raises on construction | Empty `reviewer`/`text` or a label outside `{offensive, benign}` — anonymous/malformed labels are refused. |
| `NotImplementedError` from `resolve_queue_path` | `MERCURY_FEEDBACK_QUEUE_URL` uses a non-`file` scheme — use a `file://` URL or path. |
| `rollback` reports `rolled_back=False` | No `previous` pointer to restore; check `--status`. |
