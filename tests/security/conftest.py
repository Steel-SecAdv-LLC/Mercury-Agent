"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Collection-time contract for tests that require the real AMA Cryptography
native library.  Mercury now treats AMA/PQC as mandatory, so
``test_pqc_gate_real_ama.py`` is always collected and missing AMA fails
at import time.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Security tests always collect AMA-native coverage."""
    del config
