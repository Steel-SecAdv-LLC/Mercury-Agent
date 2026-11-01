# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agent's detection memory survives a restart when a path is configured.

Unset, nothing changes: the retriever keeps an in-process list. Set,
``MERCURY_AGENT_MEMORY_PATH`` names a SQLite file that every logged
detection is appended to and that a new retriever loads at construction, so
"what was the latest anomaly?" has an answer after a redeploy and every worker
on the same path shares one memory. Principal scoping is stored with the
entry, so the boundary holds across the restart too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_mercury_engine.narrative.detection_memory import (
    DEFAULT_KEEP,
    ENV_VAR,
    DetectionMemoryStore,
)
from omni_mercury_engine.narrative.retriever import KnowledgeRetriever

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _flagged(score: float, method: str = "multivariate") -> dict[str, object]:
    return {
        "method": method,
        "anomaly_detected": True,
        "anomaly_count": 1,
        "anomaly_score": score,
        "n_samples": 10,
        "threshold": 2.0,
        "indices": [3],
    }


def test_unset_path_keeps_memory_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    retriever = KnowledgeRetriever()
    assert retriever._detection_store is None
    retriever.log_detection(_flagged(5.0), domain="general", principal="a")
    assert len(KnowledgeRetriever().detections(principal="a")) == 0


def test_memory_survives_a_new_retriever(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "state" / "memory.sqlite"))
    first = KnowledgeRetriever()
    first.log_detection(_flagged(5.0, "univariate"), domain="general", principal="a")
    first.log_detection(_flagged(9.0), domain="medical", principal="b")

    second = KnowledgeRetriever()  # a restart
    mine = second.detections(principal="a")
    assert len(mine) == 1 and mine[0]["detection"]["anomaly_score"] == 5.0
    assert mine[0]["domain"] == "general" and mine[0]["principal"] == "a"
    assert [e["detection"]["method"] for e in second.detections(principal="b")] == ["multivariate"]
    assert second.detections(principal="nobody") == []
    assert second.search("latest anomaly events", domain="general", principal="a").total_found == 1


def test_store_is_bounded_and_ordered(tmp_path: Path) -> None:
    store = DetectionMemoryStore(tmp_path / "m.sqlite", keep=3)
    for i in range(5):
        store.append(
            {"detection": {"i": i}, "domain": None, "principal": None, "timestamp": 100.0 + i}
        )
    assert store.count() == 3
    loaded = store.load()
    assert [e["detection"]["i"] for e in loaded] == [2, 3, 4]
    assert [e["timestamp"] for e in loaded] == [102.0, 103.0, 104.0]
    store.close()


def test_unusable_path_degrades_to_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path))  # a directory, not a file
    assert DetectionMemoryStore.from_env() is None
    assert KnowledgeRetriever()._detection_store is None


def test_default_window_matches_the_retriever_log() -> None:
    assert DEFAULT_KEEP == 1000
