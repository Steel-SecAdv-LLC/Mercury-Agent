# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Live-endpoint smoke tests for :class:`HailLoader` (``@pytest.mark.network``).

``tests/loaders/test_hail_loader.py`` exercises the loader offline against
recorded SPC bytes. These tests exercise the two live code paths those
recordings stand in for, so an upstream format change or URL move surfaces
as a failed weekly ``network-tests`` run rather than as silent bitrot:

* ``fetch_realtime`` against the SPC filtered daily-reports feed -- the hail
  section must still be present and sizes must still be hundredths of an
  inch (normalised to inches here).
* ``fetch_historical`` against the zipped SPC 1955-2023 hail archive -- the
  Vivian 2010 window must still reproduce the counts the offline fixture was
  recorded from, so the fixture cannot drift from the source it claims to
  mirror.

Skipped unless ``MERCURY_NETWORK_TESTS=1`` (see ``tests/conftest.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from omni_mercury_engine.loaders.hail_loader import HailLoader

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.network

#: Largest hailstone on US record is 8.0 in (Vivian SD, 2010); a daily size
#: above this means the hundredths-of-an-inch scaling broke.
_MAX_PLAUSIBLE_HAIL_IN = 8.0


def test_realtime_feed_still_has_a_parseable_hail_section(tmp_path: Path) -> None:
    df = HailLoader(cache_dir=tmp_path).fetch_realtime()

    assert {"time", "mag", "slat", "slon"} <= set(df.columns)
    sizes = pd.to_numeric(df["mag"], errors="coerce").dropna()
    # The feed may legitimately carry zero hail reports on a quiet day; when
    # it carries any, they must be positive and on the inch scale.
    assert (sizes > 0).all(), sizes.tolist()
    assert (sizes <= _MAX_PLAUSIBLE_HAIL_IN).all(), sizes.tolist()


def test_live_archive_reproduces_the_recorded_vivian_2010_window(tmp_path: Path) -> None:
    loader = HailLoader(cache_dir=tmp_path)

    df = loader.fetch_historical("vivian_2010")
    labels = loader.get_ground_truth("vivian_2010")

    # The same (n_rows, n_significant, max_size) the offline fixture pins.
    assert len(df) == 69
    assert int(labels.sum()) == 4
    assert pd.to_numeric(df["mag"]).max() == pytest.approx(8.0)
