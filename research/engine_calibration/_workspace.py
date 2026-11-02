# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the calibration scripts keep their fetched-once cache.

These scripts hardcoded ``/home/user/eqlab/cache`` -- a path on one
developer's machine. ``research/engine_calibration/results.json`` is measured
from that cache and backs the README's per-domain hazard table, so the table
was not reproducible in a clean checkout or in CI by anyone but its author:
the scripts would create an empty directory, find no ``*.npz``, and measure
nothing.

Resolution order, so a clean checkout works and CI can point it anywhere:

1. ``MERCURY_CALIBRATION_CACHE`` if set;
2. ``$XDG_CACHE_HOME/mercury-agent/engine_calibration``;
3. ``~/.cache/mercury-agent/engine_calibration``.

Deliberately outside the repository: the cache is tens of megabytes of
fetched third-party data, it is a build intermediate rather than a committed
artifact, and nothing in ``.gitignore`` should have to chase it.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "MERCURY_CALIBRATION_CACHE"
_SUBDIR = Path("mercury-agent") / "engine_calibration"


def workspace() -> Path:
    """Return the calibration workspace directory, creating it if needed."""
    override = os.environ.get(_ENV_VAR)
    if override:
        root = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        root = base / _SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_dir() -> str:
    """The per-event ``{domain}__{event}.npz`` cache directory, as a string.

    A string because the call sites build paths with f-strings and ``glob``.
    """
    path = workspace() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


#: Suffix of the de-leak measurement's archives (``X_full``/``X_scored``/``y``),
#: which share the cache directory with the per-event ``X``/``y`` archives.
DELEAK_SUFFIX = ".deleak.npz"


#: Suffix of the marker ``quarantine_event`` leaves beside a removed archive.
QUARANTINE_SUFFIX = ".quarantined.json"


def quarantine_event(domain: str, event: str, reason: str) -> None:
    """Remove ``{domain}__{event}.npz`` and leave a marker saying why.

    A rebuild whose loader refuses a de-leaked matrix (a synthetic frame) must
    not leave an older archive of the event in place for ``cache_archives()``
    to hand to the evaluation scripts; the archive is deleted, and the marker
    keeps the event out even if a stale archive is copied back in.
    """
    import json
    import time

    root = Path(cache_dir())
    archive = root / f"{domain}__{event}.npz"
    if archive.exists():
        archive.unlink()
    (root / f"{domain}__{event}{QUARANTINE_SUFFIX}").write_text(
        json.dumps({"domain": domain, "event": event, "reason": reason, "at": time.time()}),
        encoding="utf-8",
    )


def cache_archives() -> list[str]:
    """The per-event ``{domain}__{event}.npz`` archives, sorted, as strings.

    The de-leak measurement (``deleak_impact.py``) writes ``*.deleak.npz``
    into the same directory with a different schema (``X_full``, ``X_scored``,
    ``y`` rather than ``X``, ``y``). A bare ``*.npz`` glob picked those up and
    the evaluation scripts failed at ``d["X"]`` after a documented
    ``deleak_impact.py --fetch``; every script that reads the per-event
    archives goes through this filter instead. An event with a quarantine
    marker (see :func:`quarantine_event`) is left out as well.
    """
    root = Path(cache_dir())
    quarantined = {p.name[: -len(QUARANTINE_SUFFIX)] for p in root.glob("*" + QUARANTINE_SUFFIX)}
    return sorted(
        str(p)
        for p in root.glob("*.npz")
        if not p.name.endswith(DELEAK_SUFFIX) and p.name[: -len(".npz")] not in quarantined
    )


def workspace_file(name: str) -> str:
    """Path to a named intermediate (``scores.pkl``, ``real_results.json``)."""
    return str(workspace() / name)
