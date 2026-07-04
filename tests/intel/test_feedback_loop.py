# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Closed feedback loop units: labeling, durable queue, signed trigger, rollback.

The corpus-heavy regression gate and end-to-end demo are exercised in
``test_regression_gate.py`` / ``test_closed_loop_integration.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.intel.feedback_loop.labeling import (
    ExampleSource,
    LabeledExample,
    apply_human_label,
    ingest_audit_event,
    override_to_example,
)
from omni_mercury_engine.intel.feedback_loop.queue import DurableLabeledQueue, resolve_queue_path
from omni_mercury_engine.intel.feedback_loop.rollback import ModelEntry, ModelRegistry
from omni_mercury_engine.intel.feedback_loop.trigger import sign_trigger, verify_trigger

if TYPE_CHECKING:
    from pathlib import Path

_SECRET = "unit-test-secret"
_AUDIT_RECORD = {
    "ts": 1_700_000_000.0,
    "decision": "refuse_redact",
    "source": "weapons_gate",
    "disposition": "refuse_redact",
    "query": "steps to synthesize a nerve agent in quantity",
    "reason": "offensive intent",
}


# --------------------------- labeling --------------------------- #
def test_ingest_and_label_audit_event() -> None:
    event = ingest_audit_event(_AUDIT_RECORD)
    assert event.query.startswith("steps to synthesize")
    example = apply_human_label(event, label="offensive", reviewer="alice", reason="verified")
    assert example.label == "offensive"
    assert example.expected == "block"
    assert example.source is ExampleSource.AUDIT_EVENT
    assert example.reviewer == "alice"


def test_audit_record_without_query_is_rejected() -> None:
    with pytest.raises(ValueError):
        ingest_audit_event({"decision": "refuse_redact"})


def test_labeled_example_requires_reviewer_and_valid_label() -> None:
    with pytest.raises(ValueError):
        override_to_example("x", label="offensive", reviewer="")  # anonymous
    with pytest.raises(ValueError):
        override_to_example("x", label="maybe", reviewer="alice")  # bad label
    with pytest.raises(ValueError):
        override_to_example("   ", label="benign", reviewer="alice")  # empty text


def test_labeled_example_roundtrip() -> None:
    ex = override_to_example("q", label="benign", reviewer="bob", reason="r", origin_ref="ref")
    assert LabeledExample.from_dict(ex.as_dict()) == ex


# --------------------------- queue --------------------------- #
def test_resolve_queue_path_schemes(tmp_path: Path) -> None:
    assert resolve_queue_path(str(tmp_path / "q.jsonl")).name == "q.jsonl"
    assert resolve_queue_path(f"file://{tmp_path / 'q.jsonl'}").name == "q.jsonl"
    with pytest.raises(NotImplementedError):
        resolve_queue_path("sqs://queue/name")


def test_queue_enqueue_dedup_and_snapshot(tmp_path: Path) -> None:
    q = DurableLabeledQueue(f"file://{tmp_path / 'queue.jsonl'}")
    ex = override_to_example("q1", label="offensive", reviewer="alice")
    assert q.enqueue(ex) is True
    assert q.enqueue(ex) is False  # dedup
    assert len(q) == 1
    h1 = q.snapshot_hash()
    q.enqueue(override_to_example("q2", label="benign", reviewer="alice"))
    assert len(q) == 2
    assert q.snapshot_hash() != h1  # snapshot changes with content
    # Durability: a fresh handle sees the enqueued rows.
    assert len(DurableLabeledQueue(f"file://{tmp_path / 'queue.jsonl'}")) == 2


# --------------------------- trigger --------------------------- #
def test_trigger_sign_and_verify_roundtrip() -> None:
    trig = sign_trigger(
        queue_hash="abc123",
        corpus_version="v1",
        requested_by="svc",
        n_examples=3,
        nonce="n1",
        secret=_SECRET,
    )
    assert trig.signature
    assert verify_trigger(trig, secret=_SECRET, audit=False)
    assert verify_trigger(trig, secret=_SECRET, expected_queue_hash="abc123", audit=False)


def test_trigger_fails_on_wrong_secret_or_tamper() -> None:
    trig = sign_trigger(
        queue_hash="abc123",
        corpus_version="v1",
        requested_by="svc",
        n_examples=3,
        nonce="n1",
        secret=_SECRET,
    )
    assert not verify_trigger(trig, secret="wrong-secret", audit=False)  # noqa: S106
    # Tamper with a bound field -> signature no longer matches.
    tampered = type(trig)(**{**trig.__dict__, "n_examples": 999})
    assert not verify_trigger(tampered, secret=_SECRET, audit=False)
    # Queue changed since signing -> binding check fails.
    assert not verify_trigger(trig, secret=_SECRET, expected_queue_hash="different", audit=False)


def test_trigger_fails_closed_without_secret() -> None:
    trig = sign_trigger(
        queue_hash="a",
        corpus_version="v",
        requested_by="s",
        n_examples=1,
        nonce="n",
        secret=_SECRET,
    )
    assert not verify_trigger(trig, secret=None, audit=False)  # no secret -> unauthorizable


def test_sign_requires_secret() -> None:
    with pytest.raises(RuntimeError):
        sign_trigger(
            queue_hash="a",
            corpus_version="v",
            requested_by="s",
            n_examples=1,
            nonce="n",
            secret="",
        )


# --------------------------- rollback registry --------------------------- #
def test_registry_register_and_one_command_rollback(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path)
    assert reg.active() is None
    reg.register(ModelEntry("v1", str(tmp_path / "m1.json"), "candidate"))
    reg.register(ModelEntry("v2", str(tmp_path / "m2.json"), "candidate"))
    active, previous = reg.active(), reg.previous()
    assert active is not None and previous is not None
    assert active.version == "v2"
    assert previous.version == "v1"
    result = reg.rollback()
    assert result.rolled_back
    assert result.from_version == "v2" and result.to_version == "v1"
    restored = reg.active()
    assert restored is not None and restored.version == "v1"


def test_rollback_noop_when_nothing_to_restore(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path)
    assert not reg.rollback().rolled_back  # empty
    reg.register(ModelEntry("v1", str(tmp_path / "m1.json"), "candidate"))
    assert not reg.rollback().rolled_back  # only one model, no previous


def test_register_same_version_is_noop(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path)
    reg.register(ModelEntry("v1", "m1", "candidate"))
    reg.register(ModelEntry("v2", "m2", "candidate"))
    reg.register(ModelEntry("v2", "m2", "candidate"))  # re-register active -> no shuffle
    previous = reg.previous()
    assert previous is not None and previous.version == "v1"
