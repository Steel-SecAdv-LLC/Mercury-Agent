# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The committed Detector Evidence Ledger matches its generator exactly.

The ledger (``artifacts/detector_evidence/ledger.json``) is the
machine-readable answer to "what does each detector detect, where was it
validated, and at what maturity tier" - with tiers COMPUTED from committed
evidence by ``scripts/generate_detector_evidence.py``, never hand-assigned.
This gate makes drift impossible: change the manifest, the ontology, the
evidence seed, or the tier rules without regenerating the ledger and this
test names the mismatch. The same discipline as the README scale block.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "generate_detector_evidence.py"
_LEDGER = _REPO_ROOT / "artifacts" / "detector_evidence" / "ledger.json"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_detector_evidence", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("generate_detector_evidence", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_generator()


class TestLedgerSync:
    def test_committed_ledger_is_regenerable_byte_identical(self, generator: ModuleType) -> None:
        assert _LEDGER.exists(), "ledger missing: run scripts/generate_detector_evidence.py"
        expected = generator._serialise(generator.build_ledger())
        assert _LEDGER.read_text() == expected, (
            "detector evidence ledger is out of sync with its committed sources; "
            "regenerate with scripts/generate_detector_evidence.py"
        )

    def test_every_manifest_detector_has_a_card(self, generator: ModuleType) -> None:
        from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

        ledger = json.loads(_LEDGER.read_text())
        assert set(ledger["detectors"]) == {e.name for e in DETECTOR_MANIFEST}

    def test_t5_is_unreachable_without_production_evidence(self, generator: ModuleType) -> None:
        ledger = json.loads(_LEDGER.read_text())
        assert (
            "T5" not in ledger["tier_counts"]
        ), "T5 requires live deployment evidence; no committed source provides it yet"
