# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The watch: Mercury reading its live feeds on its own and saying what matters.

The loaders can each read a live source and the detectors can each score a
batch, but nothing ran them: Mercury acted only when called, and a batch
scored against itself flags about one row in twenty whatever it holds. This
package is the part in between.

* :mod:`~omni_mercury_engine.watch.store` remembers every finding, so one
  already reported is not reported again, and what a later record said of it.
* :mod:`~omni_mercury_engine.watch.feeds` judges. Each feed turns a pull into
  at most a few findings by the scale its field already uses (USGS magnitude
  classes, the NOAA G scale, a DART buoy's own event mode, the NWS flood
  category), and later checks each finding against the record that settles
  it. Most pulls yield none.
* :mod:`~omni_mercury_engine.watch.voice` speaks: one plain account per
  finding, with its source and what it cannot tell.
* :mod:`~omni_mercury_engine.watch.watcher` runs them on a schedule, from
  ``mercury-agent watch`` or inside the server
  (:mod:`~omni_mercury_engine.watch.service`).
"""

from omni_mercury_engine.watch.feeds import FEEDS, Feed
from omni_mercury_engine.watch.model import FeedStatus, Finding, Severity
from omni_mercury_engine.watch.store import WatchStore, default_state_dir
from omni_mercury_engine.watch.watcher import Watcher

__all__ = [
    "FEEDS",
    "Feed",
    "FeedStatus",
    "Finding",
    "Severity",
    "WatchStore",
    "Watcher",
    "default_state_dir",
]
