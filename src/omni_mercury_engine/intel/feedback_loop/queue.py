# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""A durable, append-only queue of human-verified labeled examples.

The learning loop's memory: labels a reviewer stood behind are enqueued here and
survive process exit, so a retrain run consumes exactly the accumulated,
verified feedback -- no more, no less. The queue is content-addressed and deduped
so the same example enqueued twice is stored once, and it exposes a **snapshot
hash** that the retrain trigger binds to, making "retrain on this queue state"
tamper-evident.

The sink is a JSON-Lines file resolved from ``MERCURY_FEEDBACK_QUEUE_URL``
(``file://`` URL or a bare path; default
``<repo>/artifacts/feedback/labeled_queue.jsonl``). Non-``file`` schemes are a
loud, fail-closed ``NotImplementedError`` rather than a silent no-op -- a durable
queue that silently drops labels would break the loop's core guarantee. The
scheme is the documented extension point (SQS/PubSub/Kafka adapters plug in
here).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from urllib.parse import urlparse

from omni_mercury_engine.intel.feedback_loop.labeling import LabeledExample

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_QUEUE_PATH = _REPO_ROOT / "artifacts" / "feedback" / "labeled_queue.jsonl"

_LOCK = threading.Lock()


def _content_id(example: LabeledExample) -> str:
    """Stable dedup id: hash of ``(text, label)`` (a re-label is a new example)."""
    key = f"{example.label}\x00{example.text}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


def resolve_queue_path(url: str | None = None) -> Path:
    """Resolve the queue file path from ``url`` / ``MERCURY_FEEDBACK_QUEUE_URL``.

    Accepts a ``file://`` URL or a bare filesystem path. A non-``file`` scheme
    raises :class:`NotImplementedError` (fail-closed extension point).
    """
    raw = url if url is not None else os.environ.get("MERCURY_FEEDBACK_QUEUE_URL", "").strip()
    if not raw:
        return DEFAULT_QUEUE_PATH
    parsed = urlparse(raw)
    if parsed.scheme == "":
        return Path(raw)  # a bare filesystem path
    if parsed.scheme == "file":
        # `file:///abs/path` -> netloc="", path="/abs/path". But a file URL built
        # from a *relative* path (`file://artifacts/x`) is split by the `//` at
        # its first segment into netloc="artifacts", path="/x"; dropping the
        # netloc would mint a spurious absolute `/x` at the filesystem root (an
        # unwritable path in a sandboxed CI runner). Rejoin netloc+path so a
        # relative file URL resolves relative to CWD as intended. A `localhost`
        # netloc (`file://localhost/abs`) is the standard local-host form -> /abs.
        if parsed.netloc and parsed.netloc != "localhost":
            return Path(parsed.netloc + parsed.path)
        return Path(parsed.path)
    raise NotImplementedError(
        f"MERCURY_FEEDBACK_QUEUE_URL scheme {parsed.scheme!r} is not supported; "
        "use a file:// URL or a path (SQS/PubSub/Kafka adapters are the documented "
        "extension point -- see docs/RETRAIN_RUNBOOK.md)"
    )


class DurableLabeledQueue:
    """Append-only, deduped, fsync'd queue of :class:`LabeledExample`."""

    def __init__(self, url: str | None = None) -> None:
        """Open (or lazily create) the queue at the resolved path."""
        self.path = resolve_queue_path(url)

    def enqueue(self, example: LabeledExample) -> bool:
        """Durably append ``example`` (deduped). Returns ``True`` if newly stored.

        The write is flushed + ``fsync``'d so an enqueued label survives a crash.
        """
        content_id = _content_id(example)
        with _LOCK:
            if content_id in self._ids_unlocked():
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {"id": content_id, **example.as_dict()}
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return True

    def enqueue_many(self, examples: list[LabeledExample]) -> int:
        """Enqueue several examples; return the number newly stored."""
        return sum(1 for e in examples if self.enqueue(e))

    def _rows_unlocked(self) -> list[dict[str, object]]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def _ids_unlocked(self) -> set[str]:
        return {str(r.get("id")) for r in self._rows_unlocked()}

    def pending(self) -> list[LabeledExample]:
        """Return every enqueued example (in insertion order)."""
        with _LOCK:
            rows = self._rows_unlocked()
        return [LabeledExample.from_dict(r) for r in rows]

    def __len__(self) -> int:
        """Number of enqueued examples."""
        with _LOCK:
            return len(self._rows_unlocked())

    @staticmethod
    def _hash_ids(ids: set[str]) -> str:
        """Order-independent content hash of a set of dedup ids."""
        return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()

    def snapshot(self) -> tuple[list[LabeledExample], str]:
        """Atomically read the pending examples **and** their snapshot hash.

        Returns ``(examples, hash)`` computed from a *single* locked read of the
        queue, so the hash provably covers exactly the returned examples. The
        retrain pipeline binds a signed trigger to this hash and trains on these
        exact examples -- eliminating the TOCTOU window that a separate
        :meth:`snapshot_hash` + :meth:`pending` pair would leave open (a row
        enqueued between the two calls would be trained on but not covered by the
        authorized hash).
        """
        with _LOCK:
            rows = self._rows_unlocked()
        examples = [LabeledExample.from_dict(r) for r in rows]
        ids = {str(r.get("id")) for r in rows}
        return examples, self._hash_ids(ids)

    def snapshot_hash(self) -> str:
        """Content hash of the whole queue (dedup ids, order-independent).

        The retrain trigger binds this so a signature authorizes exactly the
        queue state it was signed against; any later enqueue changes the hash.
        Prefer :meth:`snapshot` in the retrain path so the hash and the trained
        rows come from one atomic read.
        """
        with _LOCK:
            ids = self._ids_unlocked()
        return self._hash_ids(ids)

    def clear(self) -> None:
        """Delete the queue file (staging/test convenience, not a loop operation)."""
        with _LOCK:
            if self.path.is_file():
                self.path.unlink()


__all__ = ["DEFAULT_QUEUE_PATH", "DurableLabeledQueue", "resolve_queue_path"]
