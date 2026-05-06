"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Collection-time guard for tests that require the real AMA Cryptography
native library.  By default, ``test_pqc_gate_real_ama.py`` is excluded
from collection so that a plain ``pytest`` invocation on a developer
machine (which typically lacks the AMA native build) does not fail at
import time.

CI workflows that build AMA Cryptography from source set
``MERCURY_PQC_REAL_AMA=1`` in their environment, which allows
collection to proceed normally.  The tests themselves still perform
a hard-fail import (no ``importorskip``, no ``skipif``) — the only
gating is whether pytest *collects* the file at all.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Files that require the real AMA native library to be importable.
_AMA_REQUIRED_FILES = {"test_pqc_gate_real_ama.py"}


def pytest_ignore_collect(collection_path: Path) -> bool:
    """Prevent collection of AMA-native tests unless the env flag is set.

    Returns True (ignore) when the file requires AMA and the
    ``MERCURY_PQC_REAL_AMA`` env var is not set to a truthy value.
    """
    if collection_path.name in _AMA_REQUIRED_FILES:
        flag = os.environ.get("MERCURY_PQC_REAL_AMA", "").strip()
        if flag not in ("1", "true", "yes"):
            return True
    return False
