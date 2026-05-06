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

**Safeguard (pytest_configure):**  If we detect a CI environment
(``CI=true`` or ``GITHUB_ACTIONS=true``) *and* the
``MERCURY_PQC_REAL_AMA`` env var is missing, the session fails
immediately with a clear error.  This prevents the PQC gate from
being silently disabled if the env var is ever removed or renamed
in the workflow YAML without updating this file.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# Files that require the real AMA native library to be importable.
_AMA_REQUIRED_FILES = {"test_pqc_gate_real_ama.py"}

# CI environment indicators (GitHub Actions, GitLab CI, generic CI).
_CI_ENV_VARS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI")


def _is_ci() -> bool:
    """Return True if running inside a CI environment."""
    return any(
        os.environ.get(var, "").strip().lower() in ("true", "1", "yes")
        for var in _CI_ENV_VARS
    )


def pytest_configure(config: pytest.Config) -> None:
    """Fail-fast safeguard: CI must always set MERCURY_PQC_REAL_AMA.

    If we are running inside CI (detected via CI/GITHUB_ACTIONS env vars)
    and the ``MERCURY_PQC_REAL_AMA`` env var is not set to a truthy value,
    fail the session immediately.  This ensures the PQC gate tests cannot
    be silently skipped due to a workflow YAML change that drops the var.
    """
    if not _is_ci():
        return
    flag = os.environ.get("MERCURY_PQC_REAL_AMA", "").strip()
    if flag in ("1", "true", "yes"):
        return
    pytest.fail(
        "MERCURY_PQC_REAL_AMA is not set in this CI environment.\n"
        "The PQC gate tests (test_pqc_gate_real_ama.py) require this env var "
        "to be exported in the workflow YAML (ci.yml ml-tests job and "
        "pqc-production-check.yml verify-real-pqc job).\n"
        "If this var was intentionally removed, update "
        "tests/security/conftest.py accordingly.\n"
        "This safeguard exists to prevent the PQC production gate from "
        "being silently disabled.",
        pytrace=False,
    )


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
