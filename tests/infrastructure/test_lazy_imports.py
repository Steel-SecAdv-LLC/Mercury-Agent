# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``omni_mercury_engine.infrastructure``: names that load on first use."""

from __future__ import annotations

import subprocess
import sys

from omni_mercury_engine import infrastructure


def test_every_exported_name_resolves_and_is_listed() -> None:
    for name in infrastructure.__all__:
        assert getattr(infrastructure, name) is not None, name
    assert {"SpaceExplorationAnalyzer", "StreamingAnomalyPipeline", "streaming"} <= set(
        dir(infrastructure)
    )


def test_humanitarian_import_leaves_torch_and_streaming_unloaded() -> None:
    """The lazy names pull in torch and the broker clients; GDACS needs neither."""
    heavy = ("torch", "omni_mercury_engine.infrastructure.streaming")
    code = (
        "import sys\n"
        "import omni_mercury_engine.infrastructure.humanitarian.crisis_monitoring.gdacs\n"
        f"print([m for m in {heavy!r} if m in sys.modules])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == "[]"
