"""
Mercury Agent - Validation test conftest with reproducibility fixtures.

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+

Provides session-scoped fixtures for reproducibility and caching.
"""

from __future__ import annotations

import random

import numpy as np
import pytest


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@pytest.fixture(scope="session", autouse=True)
def setup_reproducibility() -> None:
    """Set seeds once per test session for deterministic results."""
    set_seeds(42)
