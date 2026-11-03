# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The calibration scripts read only the per-event archives, never the de-leak ones.

``deleak_impact.py`` writes ``{domain}__{event}.deleak.npz`` (``X_full``,
``X_scored``, ``y``) into the same cache directory as the per-event
``{domain}__{event}.npz`` (``X``, ``y``). A bare ``*.npz`` glob in
``eval_offline.py``, ``measure.py`` and ``method_bakeoff.py`` picked both up
and failed at ``d["X"]``; they now share one filtered listing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "research" / "engine_calibration"


def _workspace_module(cache_root: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MERCURY_CALIBRATION_CACHE", str(cache_root))
    monkeypatch.syspath_prepend(str(_SCRIPTS))
    sys.modules.pop("_workspace", None)
    return importlib.import_module("_workspace")


def test_cache_archives_exclude_the_deleak_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _workspace_module(tmp_path, monkeypatch)
    cache = Path(ws.cache_dir())
    np.savez_compressed(
        cache / "earthquake__ev1.npz", X=np.zeros((4, 2)), y=np.zeros(4), schema=np.array(2)
    )
    np.savez_compressed(
        cache / "flood__ev2.npz", X=np.zeros((4, 2)), y=np.zeros(4), schema=np.array(2)
    )
    np.savez_compressed(
        cache / "earthquake__ev1.deleak.npz",
        X_full=np.zeros((4, 3)),
        X_scored=np.zeros((4, 2)),
        y=np.zeros(4),
    )
    names = [Path(p).name for p in ws.cache_archives()]
    assert names == ["earthquake__ev1.npz", "flood__ev2.npz"]
    for path in ws.cache_archives():
        assert "X" in np.load(path).files


def test_every_reader_goes_through_the_filtered_listing() -> None:
    for script in ("eval_offline.py", "measure.py", "method_bakeoff.py"):
        text = (_SCRIPTS / script).read_text(encoding="utf-8")
        assert 'glob("*.npz")' not in text, script
        assert "cache_archives()" in text, script


def test_quarantined_events_are_removed_and_stay_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected rebuild deletes the stale archive and a marker keeps the event out."""
    ws = _workspace_module(tmp_path, monkeypatch)
    cache = Path(ws.cache_dir())
    np.savez_compressed(
        cache / "energy__quebec_1989.npz", X=np.zeros((4, 2)), y=np.zeros(4), schema=np.array(2)
    )
    np.savez_compressed(
        cache / "flood__ev2.npz", X=np.zeros((4, 2)), y=np.zeros(4), schema=np.array(2)
    )
    ws.quarantine_event("energy", "quebec_1989", "synthetic frame")
    assert not (cache / "energy__quebec_1989.npz").exists()
    assert (cache / "energy__quebec_1989.quarantined.json").exists()
    assert [Path(p).name for p in ws.cache_archives()] == ["flood__ev2.npz"]
    # even a stale archive copied back in stays out while the marker exists
    np.savez_compressed(
        cache / "energy__quebec_1989.npz", X=np.zeros((4, 2)), y=np.zeros(4), schema=np.array(2)
    )
    assert [Path(p).name for p in ws.cache_archives()] == ["flood__ev2.npz"]


def test_build_cache_quarantines_a_refused_synthetic_frame() -> None:
    text = (_SCRIPTS / "build_cache.py").read_text(encoding="utf-8")
    assert "except SyntheticDataNotDeleakable" in text
    assert "quarantine_event(dom, eid" in text


def test_a_recovered_event_leaves_quarantine_once_its_archive_is_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real data arriving later lifts the marker; a failed rebuild does not."""
    ws = _workspace_module(tmp_path, monkeypatch)
    cache = Path(ws.cache_dir())
    ws.quarantine_event("energy", "quebec_1989", "synthetic frame")
    assert ws.cache_archives() == []
    # the rebuild path writes the validated archive, then lifts the marker
    np.savez_compressed(
        cache / "energy__quebec_1989.npz", X=np.zeros((4, 2)), y=np.zeros(4), schema=np.array(2)
    )
    assert ws.clear_quarantine("energy", "quebec_1989") is True
    assert not (cache / "energy__quebec_1989.quarantined.json").exists()
    assert [Path(p).name for p in ws.cache_archives()] == ["energy__quebec_1989.npz"]
    assert ws.clear_quarantine("energy", "quebec_1989") is False
    text = (_SCRIPTS / "build_cache.py").read_text(encoding="utf-8")
    assert text.index("np.savez_compressed(\n                fn,") < text.index(
        "clear_quarantine(dom, eid)"
    )


def test_archives_written_before_the_exclusion_are_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-exclusion archive still holds the label-derived columns.

    ``build_cache.py`` once wrote ``X`` from ``engineer_features`` (every
    column, the labelled ones included). Such an archive carries no ``schema``
    key; reading it would score a circular matrix, so the listing leaves it
    out until the event is rebuilt with the de-leaked matrix and the schema.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    ws = _workspace_module(tmp_path, monkeypatch)
    np.savez_compressed(cache / "earthquake__old.npz", X=np.zeros((4, 3)), y=np.zeros(4))
    np.savez_compressed(
        cache / "earthquake__v1.npz", X=np.zeros((4, 3)), y=np.zeros(4), schema=np.array(1)
    )
    np.savez_compressed(
        cache / "earthquake__new.npz",
        X=np.zeros((4, 2)),
        y=np.zeros(4),
        schema=np.array(ws.ARCHIVE_SCHEMA),
        scored_columns=np.array(["depth", "rate"], dtype=str),
        label_derived_excluded=np.array(["magnitude"], dtype=str),
    )
    assert [Path(p).name for p in ws.cache_archives()] == ["earthquake__new.npz"]
    assert ws.archive_schema(cache / "earthquake__old.npz") == 1
    text = (_SCRIPTS / "build_cache.py").read_text(encoding="utf-8")
    assert "schema=np.array(ARCHIVE_SCHEMA)" in text
    assert "scored_columns=np.array(list(L.scored_feature_columns())" in text
