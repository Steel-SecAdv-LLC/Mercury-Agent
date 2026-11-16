# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run the watch inside a serving process.

Off unless ``MERCURY_WATCH_ENABLED=true``: a server that reads third-party
feeds on a timer is a choice for the operator, like the browser front end.
The watch runs on one daemon thread; a feed that fails is recorded as down and
never reaches the request path.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

__all__ = ["start_background_watch", "stop_background_watch", "watch_enabled", "watching"]

_thread: threading.Thread | None = None
_stop = threading.Event()
_guard = threading.Lock()


def watch_enabled() -> bool:
    """Whether the serving process should run the watch (default: off)."""
    return os.getenv("MERCURY_WATCH_ENABLED", "false").strip().lower() == "true"


def watching() -> bool:
    """Whether this process is running the watch now."""
    return _thread is not None and _thread.is_alive()


def start_background_watch() -> bool:
    """Start the watch on a daemon thread. Returns whether it is running.

    Idempotent, and failure-isolated: if the watch cannot start, the reason
    is logged and the server serves without it.
    """
    global _thread
    with _guard:
        if watching():
            return True
        try:
            from omni_mercury_engine.watch.watcher import Watcher

            watcher = Watcher()
        except Exception:
            logger.exception("watch: could not start; serving without it")
            return False
        _stop.clear()
        _thread = threading.Thread(
            target=watcher.run_forever, kwargs={"stop": _stop}, name="mercury-watch", daemon=True
        )
        _thread.start()
        logger.info("watch: reading %d feeds in the background", len(watcher.feeds))
        return True


def stop_background_watch(timeout: float = 5.0) -> None:
    """Ask the watch to stop after the read in progress."""
    global _thread
    with _guard:
        _stop.set()
        if _thread is not None:
            _thread.join(timeout)
        _thread = None
