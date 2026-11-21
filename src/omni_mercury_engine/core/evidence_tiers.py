# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Each detector's and model's evidence tier, as the committed ledger computed it.

The Detector Evidence Ledger (``artifacts/detector_evidence/ledger.json``,
written by ``scripts/generate_detector_evidence.py``) grades every manifest
component from committed evidence alone: T0 is measured broken on its
declared modality, T1 runs ungraded or at about chance, T2 and above are
verified on real data, never hand-assigned. The ledger is an artifact, not
package data; this module reads the small copy of its tiers that ships
inside the package (``evidence_tiers.json``, written by the same generator
and held to the ledger by ``scripts/run_ci_gates.sh`` and
``tests/core/test_evidence_ledger_in_sync.py``), so that what runs can ask
what is proven.

The engine uses it for one thing: a component the ledger measures broken
(T0) is constructed and reachable exactly as before — no released API is
removed — but contributes no feature and no score on the default path
(:meth:`~omni_mercury_engine.engine.OmniMercuryEngine._models_on_the_default_path`).

If the file cannot be read, nothing is claimed either way: every tier is
``None`` and nothing is excluded, which is the behaviour before tiers
existed.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["measured_broken", "tier", "tiers"]

_PATH = Path(__file__).with_name("evidence_tiers.json")


@functools.lru_cache(maxsize=1)
def tiers() -> dict[str, dict[str, str]]:
    """Every component's ``{"tier": ..., "basis": ...}``, or ``{}`` if unreadable."""
    try:
        document: Any = json.loads(_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("the evidence tiers could not be read (%s); none is claimed", exc)
        return {}
    held = document.get("tiers")
    return held if isinstance(held, dict) else {}


def tier(name: str) -> str | None:
    """The evidence tier of one component, or ``None`` where the ledger has no card."""
    card = tiers().get(name)
    return str(card["tier"]) if card else None


def measured_broken() -> frozenset[str]:
    """The components the ledger measures broken on their declared modality (T0)."""
    return frozenset(name for name, card in tiers().items() if card.get("tier") == "T0")
