# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test federated robust."""

from __future__ import annotations


def test_federated_robust_canonical_import() -> None:
    from omni_mercury_engine.federated_learning.federated_robust import (
        FederatedAnomalyDetection,
    )

    assert FederatedAnomalyDetection is not None


def test_federated_robust_package_export() -> None:
    from omni_mercury_engine.federated_learning import FederatedAnomalyDetection

    assert FederatedAnomalyDetection is not None
