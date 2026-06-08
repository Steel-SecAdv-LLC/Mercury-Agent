# Copyright (C) 2025 Steel Security Advisors LLC
"""Collection-time contract for tests that require the real AMA Cryptography."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Security tests always collect AMA-native coverage."""
    del config
