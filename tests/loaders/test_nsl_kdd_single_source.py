# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The NSL-KDD column schema has one source of truth across packages.

``loaders/network_security_loader`` bridges to ``datasets/security`` and used to
carry a byte-identical *copy* of the 43-column NSL-KDD schema (plus an unused
duplicate of the categorical list).  These tests pin the reconciliation: the
loader now *derives* its columns from ``datasets.security.NSLKDDLoader`` so the
two definitions can never drift.
"""

from __future__ import annotations

from omni_mercury_engine.datasets.security import NSLKDDLoader
from omni_mercury_engine.loaders import network_security_loader as nsl


def test_nsl_kdd_columns_are_single_sourced_from_datasets_security() -> None:
    """The loader's column list *is* the canonical ``NSLKDDLoader.COLUMN_NAMES``."""
    assert nsl._NSLKDD_COLUMNS is NSLKDDLoader.COLUMN_NAMES


def test_nsl_kdd_schema_shape_is_pinned() -> None:
    """The shared schema is 41 features + label + difficulty, with 3 categoricals."""
    assert len(NSLKDDLoader.COLUMN_NAMES) == 43
    assert NSLKDDLoader.COLUMN_NAMES[-2:] == ["label", "difficulty"]
    assert NSLKDDLoader.CATEGORICAL_COLS == ["protocol_type", "service", "flag"]


def test_dead_categorical_duplicate_is_removed() -> None:
    """The unused ``_NSLKDD_CATEGORICAL`` copy no longer exists in the loader."""
    assert not hasattr(nsl, "_NSLKDD_CATEGORICAL")
