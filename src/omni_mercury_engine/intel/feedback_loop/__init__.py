# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Accept-gated closed feedback loop for Mercury's harm gate.

Audit events and red-team survivors are a learning signal *and* a poisoning
surface. This subpackage closes the loop safely: it turns gate decisions and
human overrides into human-verified labels (:mod:`.labeling`), stores them in a
durable, snapshot-hashable queue (:mod:`.queue`), authorizes a retrain with a
signed/audited, queue-bound trigger (:mod:`.trigger`), and admits a candidate
model only after an OOF/adversarial **regression gate** (:mod:`.regression_gate`)
clears it -- with a one-command **rollback** (:mod:`.rollback`) if a staged model
misbehaves. :func:`.retrain.staged_refit` composes all of it.

No model update happens without: a verifying signature, a human sign-off, and a
candidate that does not regress calibration or adversarial recall.
"""

from __future__ import annotations

from omni_mercury_engine.intel.feedback_loop.labeling import (
    AuditEvent,
    ExampleSource,
    LabeledExample,
    apply_human_label,
    ingest_audit_event,
    override_to_example,
)
from omni_mercury_engine.intel.feedback_loop.queue import (
    DEFAULT_QUEUE_PATH,
    DurableLabeledQueue,
    resolve_queue_path,
)
from omni_mercury_engine.intel.feedback_loop.regression_gate import (
    CandidateReport,
    RegressionVerdict,
    evaluate_candidate,
    gate_reports,
    load_base_corpus,
)
from omni_mercury_engine.intel.feedback_loop.retrain import RetrainResult, staged_refit
from omni_mercury_engine.intel.feedback_loop.rollback import (
    ModelEntry,
    ModelRegistry,
    RollbackResult,
    rollback_staging,
)
from omni_mercury_engine.intel.feedback_loop.trigger import (
    RetrainTrigger,
    secret_from_env,
    sign_trigger,
    verify_trigger,
)

__all__ = [
    "DEFAULT_QUEUE_PATH",
    "AuditEvent",
    "CandidateReport",
    "DurableLabeledQueue",
    "ExampleSource",
    "LabeledExample",
    "ModelEntry",
    "ModelRegistry",
    "RegressionVerdict",
    "RetrainResult",
    "RetrainTrigger",
    "RollbackResult",
    "apply_human_label",
    "evaluate_candidate",
    "gate_reports",
    "ingest_audit_event",
    "load_base_corpus",
    "override_to_example",
    "resolve_queue_path",
    "rollback_staging",
    "secret_from_env",
    "sign_trigger",
    "staged_refit",
    "verify_trigger",
]
