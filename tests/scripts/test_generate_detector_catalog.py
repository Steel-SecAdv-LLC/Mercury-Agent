# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The detector catalog is generated, complete, and drift-gated."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "generate_detector_catalog.py"
CATALOG = REPO / "docs" / "DETECTORS.md"


def test_committed_catalog_is_fresh() -> None:
    """`--check` passes against the committed docs/DETECTORS.md."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_catalog_covers_every_manifest_entry() -> None:
    from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

    text = CATALOG.read_text(encoding="utf-8")
    missing = [e.name for e in DETECTOR_MANIFEST if f"| `{e.name}` |" not in text]
    assert not missing, f"catalog rows missing for: {missing}"


def test_check_fails_on_stale_catalog(tmp_path: Path) -> None:
    """Drift is refused: a modified catalog fails `--check` (fails open = theater)."""
    original = CATALOG.read_text(encoding="utf-8")
    try:
        CATALOG.write_text(original + "\nstale line\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 1
        assert "DRIFT" in proc.stderr
    finally:
        CATALOG.write_text(original, encoding="utf-8")
