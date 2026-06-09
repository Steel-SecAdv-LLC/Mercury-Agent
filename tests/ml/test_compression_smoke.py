# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test compression smoke."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_compression_importable() -> None:
    from omni_mercury_engine.ml.compression import CompressionMethod, ModelCompressor

    assert ModelCompressor is not None
    assert CompressionMethod is not None
