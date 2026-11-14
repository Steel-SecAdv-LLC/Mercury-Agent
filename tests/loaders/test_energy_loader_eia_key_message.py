# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``texas_2021`` refusal names the ``EIA_API_KEY`` variable the user must set.

``texas_2021`` is the one catalog event backed by a real keyed source (EIA
grid data). Without the key the loader refused with the same reason it gives
the pre-SWPC storms -- "predates the SWPC real-time feeds and no keyed grid
source is configured for it" -- which is false for a 2021 event and never
names the variable that would fix it. No network is touched: the refusal is
raised before any request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.loaders.energy_loader import EnergyLoader

if TYPE_CHECKING:
    from pathlib import Path


def test_texas_2021_without_a_key_names_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: a refusal that never mentioned ``EIA_API_KEY``."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.delenv("EIA_API_KEY", raising=False)

    with pytest.raises(DataSourceUnavailableError) as excinfo:
        EnergyLoader(cache_dir=tmp_path).fetch_historical("texas_2021")

    message = str(excinfo.value)
    assert "EIA_API_KEY" in message
    assert "texas_2021" in message
    assert "predates" not in message


def test_pre_swpc_storm_keeps_its_own_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A storm with no keyed source must not send the user after an EIA key."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.delenv("EIA_API_KEY", raising=False)

    with pytest.raises(DataSourceUnavailableError) as excinfo:
        EnergyLoader(cache_dir=tmp_path).fetch_historical("quebec_1989")

    message = str(excinfo.value)
    assert "predates the SWPC real-time feeds" in message
    assert "EIA_API_KEY" not in message
