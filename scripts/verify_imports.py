# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verification script to test imports and scalar counts."""

from omni_mercury_engine.core import ethical_config
from omni_mercury_engine.core.ethical_config import EthicalScalars
from omni_mercury_engine.models import (
    astrophysical,
    multiverse,
    neurosymbolic,
)

# Verify modules are accessible (satisfies static analysis)
assert ethical_config is not None, "ethical_config module not loaded"
assert astrophysical is not None, "astrophysical module not loaded"
assert multiverse is not None, "multiverse module not loaded"
assert neurosymbolic is not None, "neurosymbolic module not loaded"

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
    "omni_aspirational_excellence",
]
for scalar in sample_scalars:
    value = getattr(s, scalar, None)
    if value is not None:
        print(f"  - {scalar}: {value}")
    else:
        print(f"  - {scalar}: NOT FOUND")
