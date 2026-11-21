# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An AUROC that does not exist is said to be undefined, in strict JSON, and kept out of means.

``research/engine_calibration/results.json`` backs the README's hazard table.
It held a bare ``NaN`` for the cholera event (not JSON), and
``deleak_impact.py`` scored the same event's constant score as 0.5 and
averaged it in (pandemic read 0.7741 where its three scorable events give
0.8654). ``scores.pkl``, which ``measure_cached.py`` reads, was written by a
script that was not in the repository.

Offline: the committed artifacts are read, and the scripts are run on small
arrays made here.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "research" / "engine_calibration"


def _strict(path: Path) -> Any:
    """Parse ``path`` as strict JSON: ``NaN`` and ``Infinity`` are errors."""

    def refuse(token: str) -> Any:
        raise AssertionError(f"{path.name} holds {token}, which is not JSON")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def _script(name: str, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("MERCURY_CALIBRATION_CACHE", str(cache_root))
    monkeypatch.delenv("MERCURY_CALIBRATION_PROXY_FALLBACK_HOSTS", raising=False)
    monkeypatch.syspath_prepend(str(_SCRIPTS))
    for stale in ("_workspace", "_summary", "_proxy_fallback", name):
        sys.modules.pop(stale, None)
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# The committed artifacts
# ---------------------------------------------------------------------------


def test_results_json_is_strict_json() -> None:
    _strict(_SCRIPTS / "results.json")


def test_every_event_says_whether_its_auroc_exists() -> None:
    results = _strict(_SCRIPTS / "results.json")
    undefined = {(row["dom"], row["event"]) for row in results["auroc_undefined"]}
    for row in results["per_event"]:
        if row["auroc"] is None:
            assert row["auroc_status"].startswith("undefined: "), row
            assert (row["dom"], row["event"]) in undefined
        else:
            assert row["auroc_status"] == "measured", row
            assert 0.0 <= row["auroc"] <= 1.0
    assert results["events"] == len(results["per_event"])
    assert results["events_auroc_defined"] == results["events"] - len(undefined)


def test_the_means_leave_the_undefined_events_out() -> None:
    results = _strict(_SCRIPTS / "results.json")
    scored = [row for row in results["per_event"] if row["auroc"] is not None]
    assert results["overall"]["auroc"] == round(float(np.mean([r["auroc"] for r in scored])), 4)
    for domain, told in results["per_domain"].items():
        mine = [row["auroc"] for row in scored if row["dom"] == domain]
        assert told["n_auroc_defined"] == len(mine)
        assert told["n_events"] == sum(row["dom"] == domain for row in results["per_event"])
        assert told["auroc"] == (round(float(np.mean(mine)), 4) if mine else None)


def test_the_cholera_event_is_undefined_in_both_artifacts() -> None:
    """Fetched, both classes present, and nothing left to rank it by."""
    results = _strict(_SCRIPTS / "results.json")
    row = next(r for r in results["per_event"] if r["event"] == "cholera_yemen_2017")
    assert row["auroc"] is None
    assert row["auroc_status"] == "undefined: constant score"
    assert row["pos"] > 0

    impact = _strict(_SCRIPTS / "deleak_impact.json")
    cholera = next(r for r in impact["per_event"] if r["event"] == "cholera_yemen_2017")
    assert cholera["auroc_deleaked"] is None
    assert cholera["delta"] is None
    assert cholera["undefined"] == {"deleaked": "undefined: constant score"}
    pandemic = impact["per_domain"]["pandemic"]
    assert pandemic["undefined_events"] == ["cholera_yemen_2017"]
    assert pandemic["n_events_scored"] == pandemic["n_events"] - 1


def test_the_deleak_means_are_over_the_events_scored_in_both_views() -> None:
    impact = _strict(_SCRIPTS / "deleak_impact.json")
    for domain, told in impact["per_domain"].items():
        rows = [r for r in impact["per_event"] if r["domain"] == domain]
        scored = [r for r in rows if r["delta"] is not None]
        assert told["n_events"] == len(rows)
        assert told["n_events_scored"] == len(scored)
        circular = round(float(np.mean([r["auroc_circular"] for r in scored])), 4)
        deleaked = round(float(np.mean([r["auroc_deleaked"] for r in scored])), 4)
        assert told["auroc_circular"] == circular, domain
        assert told["auroc_deleaked"] == deleaked, domain
    # No row passes an all-tied score off as a rank of one half.
    assert not [r for r in impact["per_event"] if "undefined" in r and r["delta"] is not None]


def test_the_two_artifacts_agree_on_the_deleaked_view() -> None:
    """Same events, same matrix, same detector: the figures are the same where both hold them.

    ``deleak_impact.py`` scores every row of an event; ``results.json`` reads
    an event over 20,000 rows at 20,000 of them, so those are left out here.
    """
    results = _strict(_SCRIPTS / "results.json")
    impact = _strict(_SCRIPTS / "deleak_impact.json")
    table = {(r["dom"], r["event"]): r for r in results["per_event"]}
    compared = 0
    for row in impact["per_event"]:
        mine = table.get((row["domain"], row["event"]))
        if mine is None or mine["subsampled"]:
            continue
        compared += 1
        if row["auroc_deleaked"] is None:
            assert mine["auroc"] is None
        else:
            assert abs(row["auroc_deleaked"] - mine["auroc"]) < 5e-5, row
    assert compared >= 40


# ---------------------------------------------------------------------------
# The scripts
# ---------------------------------------------------------------------------


def test_summarise_writes_null_for_an_undefined_auroc_and_leaves_it_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_module = _script("_summary", tmp_path, monkeypatch)
    rows = [
        {"dom": "a", "event": "one", "auroc": 0.8, "auprc": 0.5},
        {"dom": "a", "event": "two", "auroc": float("nan"), "auprc": 0.1},
        {"dom": "b", "event": "three", "auroc": None, "auroc_status": "undefined: constant score"},
    ]
    summary = summary_module.summarise(rows)
    assert summary["events"] == 3
    assert summary["events_auroc_defined"] == 1
    assert summary["overall"]["auroc"] == 0.8
    assert summary["per_domain"]["a"]["auroc"] == 0.8
    assert summary["per_domain"]["a"]["n_auroc_defined"] == 1
    assert summary["per_domain"]["b"]["auroc"] is None
    assert [u["event"] for u in summary["auroc_undefined"]] == ["two", "three"]

    out = tmp_path / "out.json"
    summary_module.write_json(out, summary)
    assert "NaN" not in out.read_text(encoding="utf-8")
    assert _strict(out)["per_event"][1]["auroc"] is None


def test_auroc_status_names_the_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary_module = _script("_summary", tmp_path, monkeypatch)
    labels = np.array([0, 0, 1, 1])
    assert summary_module.auroc_status(labels, [0.1, 0.2, 0.8, 0.9]) == "measured"
    assert summary_module.auroc_status(labels, [0.3, 0.3, 0.3, 0.3]) == "undefined: constant score"
    assert summary_module.auroc_status([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]) == (
        "undefined: one class only"
    )


def test_deleak_impact_does_not_score_a_constant_score_as_a_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One event with a real ranking and one whose de-leaked matrix is all zeros."""
    impact = _script("deleak_impact", tmp_path, monkeypatch)
    cache = Path(impact.cache_dir())
    monkeypatch.setattr(impact, "CACHE", cache)
    monkeypatch.setattr(impact, "ARTIFACT", tmp_path / "deleak_impact.json")

    rng = np.random.default_rng(7)
    labels = np.zeros(80, dtype=int)
    labels[60:70] = 1
    informative = rng.normal(size=(80, 3))
    informative[60:70] += 6.0
    np.savez_compressed(
        cache / "pandemic__ranked.deleak.npz", X_full=informative, X_scored=informative, y=labels
    )
    np.savez_compressed(
        cache / "pandemic__nothing_left.deleak.npz",
        X_full=informative,
        X_scored=np.zeros((80, 3)),
        y=labels,
    )

    artifact = impact.measure()
    rows = {row["event"]: row for row in artifact["per_event"]}
    assert rows["nothing_left"]["auroc_deleaked"] is None
    assert rows["nothing_left"]["delta"] is None
    assert rows["nothing_left"]["undefined"] == {"deleaked": "undefined: constant score"}
    assert rows["ranked"]["delta"] == 0.0

    pandemic = artifact["per_domain"]["pandemic"]
    assert pandemic["n_events"] == 2
    assert pandemic["n_events_scored"] == 1
    assert pandemic["undefined_events"] == ["nothing_left"]
    assert pandemic["auroc_deleaked"] == rows["ranked"]["auroc_deleaked"]
    assert pandemic["delta"] == 0.0
    json.dumps(artifact, allow_nan=False)


def test_score_events_reads_a_long_event_at_evenly_spaced_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = _script("score_events", tmp_path, monkeypatch)
    assert producer.ROW_LIMIT == 20_000
    assert producer.evenly_spaced(12).tolist() == list(range(12))
    kept = producer.evenly_spaced(148_517)
    assert len(kept) == 20_000
    assert (kept[0], kept[-1]) == (0, 148_516)
    assert np.array_equal(kept, np.linspace(0, 148_516, 20_000).astype(int))
    assert np.all(np.diff(kept) >= 7)


def test_score_events_writes_what_measure_cached_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The producer's ``scores.pkl`` is the file, and the keys, the measuring step opens."""
    import pickle

    producer = _script("score_events", tmp_path, monkeypatch)
    workspace = importlib.import_module("_workspace")
    cache = Path(workspace.cache_dir())
    rng = np.random.default_rng(3)
    rows = rng.normal(size=(90, 2))
    labels = np.zeros(90, dtype=int)
    labels[40:48] = 1
    rows[40:48] += 5.0
    np.savez_compressed(cache / "demo__two_class.npz", X=rows, y=labels, schema=np.array(2))
    np.savez_compressed(
        cache / "demo__one_class.npz", X=rows, y=np.zeros(90, dtype=int), schema=np.array(2)
    )

    assert producer.main() == 0
    with open(workspace.workspace_file("scores.pkl"), "rb") as handle:
        scored = pickle.load(handle)
    assert list(scored) == [("demo", "two_class")]
    held = scored[("demo", "two_class")]
    assert held["scores"].shape == (90,)
    assert held["y"].tolist() == labels.tolist()
    assert held["subsampled"] is False
    assert held["rows"] == 90

    measuring = (_SCRIPTS / "measure_cached.py").read_text(encoding="utf-8")
    assert 'workspace_file("scores.pkl")' in measuring
    for key in ('r["scores"]', 'r["y"]', 'r.get("subsampled"'):
        assert key in measuring, key


def test_the_table_can_be_built_from_the_repository() -> None:
    """Each step names the next, and the cache holds every row the table prints."""
    producer = (_SCRIPTS / "score_events.py").read_text(encoding="utf-8")
    for step in ("build_cache.py", "score_events.py", "measure_cached.py"):
        assert f"python research/engine_calibration/{step}" in producer, step
    assert "cache_archives()" in producer
    assert 'glob("*.npz")' not in producer

    building = (_SCRIPTS / "build_cache.py").read_text(encoding="utf-8")
    results = _strict(_SCRIPTS / "results.json")
    for domain in results["domains"]:
        assert f'"{domain}": (' in building, f"{domain} is in the table but not in the cache build"


def test_the_proxy_fallback_is_off_unless_asked_for_and_is_not_in_the_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = _script("_proxy_fallback", tmp_path, monkeypatch)
    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    before = SafeHTTPClient.__dict__["get_bytes"]
    assert fallback.install_from_env() == frozenset()
    assert SafeHTTPClient.__dict__["get_bytes"] is before

    shipped = _REPO / "src" / "omni_mercury_engine"
    for source in shipped.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "MERCURY_CALIBRATION_PROXY_FALLBACK_HOSTS" not in text, source
        assert "_proxy_fallback" not in text, source
