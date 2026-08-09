# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pillar tests: one module per pillar Mercury claims to stand on.

Every module here answers the same question for its pillar: *what would have to
be true in the code for this claim to be honest, and does a test observe it?*
A pillar with no passing test in this package is an aspiration, and
``tests/pillars/test_candor.py`` fails CI if it is written up as anything else.

Measured capability numbers do **not** live here. They live in
``CAPABILITY_MATRIX.md`` with a repro command per row, because a pillar is a
property that either holds or does not, while a capability is a measurement
that drifts and must be re-measured rather than asserted.
"""
