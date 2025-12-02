"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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

"""Verification script to test imports and scalar counts."""

from omni_anomaly_engine.models import multiverse as _multiverse  # noqa: F401
from omni_anomaly_engine.models import astrophysical as _astrophysical  # noqa: F401
from omni_anomaly_engine.models import neurosymbolic as _neurosymbolic  # noqa: F401
from omni_anomaly_engine.core import ethical_config as _ethical_config  # noqa: F401
from omni_anomaly_engine.core.ethical_config import EthicalScalars

print("✅ All imports successful")

s = EthicalScalars()
scalar_count = len(s.to_dict())
print(f"✅ Total scalars: {scalar_count}")

if scalar_count >= 100:
    print(f"✅ PASS: Scalar count ({scalar_count}) meets requirement (>=100)")
else:
    print(f"❌ FAIL: Scalar count ({scalar_count}) below requirement (>=100)")

print("\nSample omni- scalars:")
sample_scalars = [
    "omnibenevolent",
    "omni_justitia",
    "omni_prescience",
    "omni_perspicacious",
    "omni_sagacious",
    "omni_ineffable_transcendence",
]
for scalar in sample_scalars:
    value = getattr(s, scalar, None)
    if value is not None:
        print(f"  - {scalar}: {value}")
    else:
        print(f"  - {scalar}: NOT FOUND")
