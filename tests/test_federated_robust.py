"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations


def test_federated_robust_canonical_import() -> None:
    from omni_mercury_engine.federated_learning.federated_robust import (
        FederatedAnomalyDetection,
    )

    assert FederatedAnomalyDetection is not None


def test_federated_robust_compat_import() -> None:
    from omni_mercury_engine.federated.federated_robust import (
        FederatedAnomalyDetection,
    )

    assert FederatedAnomalyDetection is not None


def test_federated_robust_package_export_canonical() -> None:
    from omni_mercury_engine.federated_learning import FederatedAnomalyDetection

    assert FederatedAnomalyDetection is not None


def test_federated_robust_package_export_compat() -> None:
    from omni_mercury_engine.federated import FederatedAnomalyDetection

    assert FederatedAnomalyDetection is not None
