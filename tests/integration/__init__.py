# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end integration tests for Mercury Agent.

This package holds cross-subsystem tests that exercise real component
boundaries — the detection pipeline, the REST API surface, external
data-source ingestion (network mocked), and post-quantum signing of
detection artifacts — rather than a single unit in isolation.

The CI ``integration-tests`` lane runs ``pytest tests/integration/`` and the
``ml-tests`` lane collects this directory as part of ``pytest tests/``. Both
lanes build the native AMA Cryptography backend first, so the import-time PQC
gate in ``omni_mercury_engine`` is satisfied and these tests run against the
real cryptographic primitives — never a stub. There is intentionally no
``skipif`` escape hatch on the package import: if AMA is missing the import
fails loudly, which is the correct signal for a mandatory dependency (matching
``tests/security/test_pqc_gate_real_ama.py``).
"""
