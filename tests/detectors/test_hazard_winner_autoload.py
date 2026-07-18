# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The seven ratified hazard winners serve by default -- one honesty contract.

Each of these detectors ships a merit-gated checkpoint that beat its physics
fallback on real held-out data (see the hook registry and each checkpoint's
provenance sidecar). Mirroring the EarthquakeDetector/seismic_stead template,
a default-constructed detector auto-loads its winner, ``load_shipped_weights``
False pins the disclosed physics configuration, an absent checkpoint falls open
to physics, and a present-but-unreadable checkpoint fails loud rather than
degrading silently. This module pins that uniform contract across all seven so
none can regress to shipping a dormant winner.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")

if TYPE_CHECKING:
    from pathlib import Path

# (test id, dotted module, class name, shipped checkpoint basename)
WINNERS: list[tuple[str, str, str, str]] = [
    (
        "hurricane",
        "omni_mercury_engine.detectors.geological.hurricane_detector",
        "HurricaneDetector",
        "hurricane_era5",
    ),
    (
        "landslide",
        "omni_mercury_engine.detectors.geological.landslide",
        "LandslideDetector",
        "landslide_coolr",
    ),
    (
        "tornado",
        "omni_mercury_engine.detectors.geological.tornado_detector",
        "TornadoDetector",
        "tornado_nexrad",
    ),
    (
        "volcanic",
        "omni_mercury_engine.detectors.geological.volcanic",
        "VolcanicEruptionDetector",
        "volcanic_avo_seismic",
    ),
    (
        "wildfire",
        "omni_mercury_engine.detectors.geological.wildfire",
        "WildfireDetector",
        "wildfire_firms",
    ),
    (
        "solar",
        "omni_mercury_engine.space.solar_storm_detector",
        "SolarStormDetector",
        "solar_storm_geomag",
    ),
    (
        "reg_deviation",
        "omni_mercury_engine.models.parapsychology",
        "ParapsychologyDetector",
        "reg_deviation_gcp",
    ),
]

_IDS = [w[0] for w in WINNERS]


def _cls(module: str, name: str) -> Any:
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize(("_id", "module", "name", "checkpoint"), WINNERS, ids=_IDS)
class TestHazardWinnerAutoload:
    """Uniform serve-the-winner-by-default contract for the seven winners."""

    def test_default_construction_serves_the_winner(
        self, _id: str, module: str, name: str, checkpoint: str
    ) -> None:
        detector = _cls(module, name)()
        assert (
            detector._neural_trained is True
        ), f"{name}: default construction did not auto-load '{checkpoint}'"

    def test_physics_configuration_is_untrained(
        self, _id: str, module: str, name: str, checkpoint: str
    ) -> None:
        detector = _cls(module, name)(load_shipped_weights=False)
        assert detector._neural_trained is False

    def test_absent_checkpoint_falls_open_to_physics(
        self,
        _id: str,
        module: str,
        name: str,
        checkpoint: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stripped install (no shipped .pt) degrades to physics, not a crash."""
        import omni_mercury_engine.models.checkpoint_paths as cp

        monkeypatch.setattr(cp, "checkpoints_dir", lambda: tmp_path)
        detector = _cls(module, name)()
        assert detector._neural_trained is False

    def test_unreadable_checkpoint_fails_loud(
        self,
        _id: str,
        module: str,
        name: str,
        checkpoint: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A present-but-corrupt checkpoint must raise at construction, never load."""
        import omni_mercury_engine.models.checkpoint_paths as cp

        (tmp_path / f"{checkpoint}.pt").write_bytes(b"not a valid checkpoint")
        monkeypatch.setattr(cp, "checkpoints_dir", lambda: tmp_path)
        with pytest.raises(RuntimeError):
            _cls(module, name)()
