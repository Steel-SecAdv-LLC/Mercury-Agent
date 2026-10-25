# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The served fusion engine learns its reliabilities from the ledger's training split."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.assessment import service
from omni_mercury_engine.security.adjudication import (
    AdjudicationLedger,
    AgreementPolicy,
    Case,
    Outcome,
    Rating,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def fresh_engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(service, "_ENGINE", None)
    service._ENGINE_STATUS.clear()
    yield
    service._ENGINE = None
    service._ENGINE_STATUS.clear()


def _ledger(path: Path, n: int) -> AdjudicationLedger:
    rng = np.random.default_rng(0)
    ledger = AdjudicationLedger(path, policy=AgreementPolicy(n_boot=50))
    items: list[Case | Outcome | Rating] = []
    for i in range(n):
        level = int(rng.integers(1, 6))
        midpoint = (level - 0.5) / 5.0
        case_id = f"case-{i:04d}"
        reports = {
            "signals": {"threat_score": float(np.clip(midpoint + rng.normal(0, 0.03), 0, 1))},
            "open_source": {"threat_score": float(rng.uniform())},
        }
        items += [
            Case(case_id, T0, reports, ("Reuters",), T0 - timedelta(hours=1)),
            Outcome(case_id, T0 + timedelta(days=1), "observed", ("UN OCHA",)),
            Rating("alice", case_id, level, "consistent with the outcome", T0 + timedelta(days=2)),
            Rating("bob", case_id, level, "agrees", T0 + timedelta(days=2)),
        ]
    ledger.extend(items)
    return ledger


def test_served_engine_learns_reliabilities_from_training_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_engine: None
) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = _ledger(path, 240)
    assert len(ledger.training_cases()) >= 120
    monkeypatch.setenv("MERCURY_ADJUDICATION_PATH", str(path))
    monkeypatch.setenv("MERCURY_INTEL_LEARN_RELIABILITY", "1")
    monkeypatch.delenv("MERCURY_INTEL_FUSION_CHECKPOINT", raising=False)
    status = service.fusion_status()
    reliability = status["reliability"]
    assert reliability["installed"], reliability["reason"]
    assert reliability["table"]["open_source"] < reliability["previous_table"]["open_source"]
    assert service.fusion_engine().int_reliability_scores == reliability["table"]


def test_without_a_usable_ledger_the_declared_table_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_engine: None
) -> None:
    monkeypatch.setenv("MERCURY_ADJUDICATION_PATH", str(tmp_path / "empty.jsonl"))
    monkeypatch.setenv("MERCURY_INTEL_LEARN_RELIABILITY", "1")
    status = service.fusion_status()
    assert status["reliability"]["installed"] is False
    assert status["decides_with"] == "evidence"
