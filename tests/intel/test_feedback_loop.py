# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Closed feedback loop units: labeling, durable queue, signed trigger, rollback.

The corpus-heavy regression gate and end-to-end demo are exercised in
``test_regression_gate.py`` / ``test_closed_loop_integration.py``.
"""

from __future__ import annotations

import logging
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
from omni_mercury_engine.intel.feedback_loop.trigger import (
    NonceLedger,
    sign_trigger,
    verify_trigger,
)

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


def test_resolve_queue_path_relative_file_url_stays_relative() -> None:
    """A file:// URL built from a *relative* path must resolve relative to CWD.

    Regression: ``file://artifacts/x`` is split by ``//`` into netloc="artifacts",
    path="/x"; dropping the netloc minted a spurious absolute ``/x`` at the
    filesystem root (an unwritable path in a sandboxed CI runner -> PermissionError
    when the demo passed a relative --staging-dir).
    """
    from pathlib import Path as _Path

    resolved = resolve_queue_path("file://artifacts/closed_loop/staging/labeled_queue.jsonl")
    assert not resolved.is_absolute()
    assert resolved == _Path("artifacts/closed_loop/staging/labeled_queue.jsonl")
    # Absolute file URLs and the localhost host form still resolve absolute.
    assert resolve_queue_path("file:///abs/q.jsonl") == _Path("/abs/q.jsonl")
    assert resolve_queue_path("file://localhost/abs/q.jsonl") == _Path("/abs/q.jsonl")


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


def test_rollback_is_monotonic_not_a_toggle(tmp_path: Path) -> None:
    """A second rollback must NOT swap the rolled-back (bad) model back in."""
    reg = ModelRegistry(tmp_path)
    reg.register(ModelEntry("v1_good", str(tmp_path / "v1.json"), "candidate"))
    reg.register(ModelEntry("v2_bad", str(tmp_path / "v2.json"), "candidate"))
    first = reg.rollback()
    assert first.rolled_back and first.to_version == "v1_good"
    restored = reg.active()
    assert restored is not None and restored.version == "v1_good"
    # The bug: a repeated rollback re-arming v2_bad. It must be a no-op instead.
    second = reg.rollback()
    assert not second.rolled_back
    still = reg.active()
    assert still is not None and still.version == "v1_good"  # v2_bad NOT restored
    # A fresh gated register repopulates the previous pointer (rollback works again).
    reg.register(ModelEntry("v3", str(tmp_path / "v3.json"), "candidate"))
    third = reg.rollback()
    assert third.rolled_back and third.to_version == "v1_good"


# --------------------------- nonce ledger (single-use) --------------------------- #
def test_nonce_ledger_single_use(tmp_path: Path) -> None:
    ledger = NonceLedger(tmp_path)
    assert ledger.consume("n1", queue_hash="h", requested_by="svc") is True
    assert ledger.consume("n1") is False  # replay refused
    assert ledger.is_consumed("n1")
    assert ledger.consume("n2") is True  # a distinct nonce is fine
    # Durability: a fresh handle still sees the consumed nonce.
    assert NonceLedger(tmp_path).is_consumed("n1")
    ledger.clear()
    assert ledger.consume("n1") is True  # reset


# --------------------------- atomic queue snapshot (TOCTOU) --------------------------- #
def test_queue_snapshot_is_atomic_and_matches_hash(tmp_path: Path) -> None:
    q = DurableLabeledQueue(f"file://{tmp_path / 'q.jsonl'}")
    q.enqueue(override_to_example("a", label="offensive", reviewer="alice"))
    q.enqueue(override_to_example("b", label="benign", reviewer="alice"))
    examples, h = q.snapshot()
    assert len(examples) == 2
    # The snapshot hash provably covers exactly the returned examples.
    assert h == q.snapshot_hash()


# --------------------------- nonce ledger (inter-process single-use) --------------------------- #
def _consume_worker(args: tuple[str, str]) -> bool:
    """Top-level (picklable) worker: consume ``nonce`` against the ledger at ``path``."""
    path_str, nonce = args
    from omni_mercury_engine.intel.feedback_loop.trigger import NonceLedger

    return NonceLedger(path_str).consume(nonce, requested_by="proc")


def test_nonce_ledger_single_use_across_processes(tmp_path: Path) -> None:
    """Two concurrent *processes* must not both consume the same nonce.

    The in-process ``threading.Lock`` is per-process, so without the inter-process
    advisory file lock several forked workers could each observe the nonce as
    absent and each append it -- consuming a single-use authorization more than
    once. With the ``flock`` guard exactly one worker wins.
    """
    import multiprocessing as mp

    nonce = "shared-nonce"
    n_workers = 12
    ctx = mp.get_context("fork")  # engine is Linux-only (native AMA .so gate)
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_consume_worker, [(str(tmp_path), nonce)] * n_workers)
    assert sum(1 for r in results if r) == 1  # exactly one process consumed it
    # And it is durably recorded exactly once.
    assert NonceLedger(tmp_path).is_consumed(nonce)


# --------------------------- queue tolerates a corrupt/truncated line --------------------------- #
def test_queue_read_tolerates_corrupt_or_truncated_line(tmp_path: Path) -> None:
    """A truncated/corrupt line must not crash every queue read (fail-closed).

    A crash mid-append leaves a partial final line; the previous ``json.loads``
    per line would raise on it, crashing dedup/snapshot/len/retrain. The valid
    rows must survive (an unreadable example is simply not trained on) and reads
    must not raise.
    """
    qpath = tmp_path / "queue.jsonl"
    q = DurableLabeledQueue(f"file://{qpath}")
    q.enqueue(override_to_example("good1", label="offensive", reviewer="alice"))
    q.enqueue(override_to_example("good2", label="benign", reviewer="alice"))
    with qpath.open("a", encoding="utf-8") as fh:
        fh.write('{"id": "trunc", "text": "partial record with no closing')  # invalid, no newline
    fresh = DurableLabeledQueue(f"file://{qpath}")
    assert len(fresh) == 2  # does not raise; corrupt tail skipped
    assert {e.text for e in fresh.pending()} == {"good1", "good2"}
    examples, h = fresh.snapshot()
    assert len(examples) == 2 and h  # snapshot tolerant too

    # A valid-JSON but non-object line (a bare scalar) is also skipped, not crashed.
    with qpath.open("a", encoding="utf-8") as fh:
        fh.write("\n42\n")
    assert len(DurableLabeledQueue(f"file://{qpath}")) == 2


# --------------------------- verify_trigger: no secret-derived data in clear-text logs --------- #
def test_verify_trigger_no_secret_configured_logs_only_reason_code(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The operator warning must carry only the enumerated reason_code.

    The composed ``reason`` names ``SECRET_ENV`` (a secret-flagged token); it must
    reach the structured audit record, never the clear-text logger.
    """
    monkeypatch.delenv("MERCURY_RETRAIN_TRIGGER_SECRET", raising=False)
    trig = sign_trigger(
        queue_hash="abc123",
        corpus_version="v",
        requested_by="svc",
        n_examples=1,
        nonce="n",
        secret=_SECRET,
    )
    with caplog.at_level(logging.WARNING):
        assert not verify_trigger(trig, secret=None, audit=False)  # no secret -> unauthorizable
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "secret_unconfigured" in logged
    assert "MERCURY_RETRAIN_TRIGGER_SECRET" not in logged  # env-var name never in clear-text log
    assert _SECRET not in logged


def test_verify_trigger_audit_preserves_full_reason_and_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detailed reason + reason_code go to the audit sink (observability kept)."""
    from omni_mercury_engine.intel.feedback_loop import trigger as trig_mod

    captured: dict[str, object] = {}
    monkeypatch.setattr(trig_mod, "record_gate_decision", lambda **kw: captured.update(kw))
    trig = sign_trigger(
        queue_hash="abc123",
        corpus_version="v",
        requested_by="svc",
        n_examples=1,
        nonce="n",
        secret=_SECRET,
    )
    assert not verify_trigger(trig, secret="wrong-secret", audit=True)  # noqa: S106
    assert captured["reason"] == "HMAC signature mismatch"
    extra = captured["extra"]
    assert isinstance(extra, dict) and extra["reason_code"] == "signature_mismatch"
