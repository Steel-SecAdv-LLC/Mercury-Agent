# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hand-maintained README numbers must match the committed measurement artifacts.

``scripts/update_readme_benchmarks.py`` regenerates two marker blocks
(``BENCHMARK`` and ``BENCHMARK-DOMAINS``) from
``benchmarks/mercury_benchmark_results.json``, and
``scripts/measure_codebase_scale.py --check`` gates the ``SCALE`` block.  Every
other number quoted in the README is typed by hand, and an audit of the v3.0.0
tree found that is exactly where the drift was:

* ``Std AUC | 0.1460`` against a run whose ``summary.std_auc`` is ``0.14379``;
* "the 9 failures were unreachable external sources" against a run where 2 of
  the 9 reached their source and failed on degenerate labels;
* "of 20 concrete loaders" against a provenance audit reporting 21.

None of those is load-bearing for a benchmark claim, which is the point: a
number nobody re-derives is a number nobody notices going stale.  This gate
re-derives each of them from the artifact that owns it, so the next run that
moves a figure fails here instead of shipping a stale README.

Offline, stdlib + numpy only; reads committed JSON, imports no engine module.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_README = _REPO / "README.md"
_GENUINE_LABELS = ("ground_truth", "expert_annotated")

#: README figures are quoted to 4 decimal places, so a half-ulp of that is the
#: tightest tolerance a correct value can be held to.
_TOL = 5e-5


def _text() -> str:
    return _README.read_text(encoding="utf-8")


def _artifact(relative: str) -> Any:
    path = _REPO / relative
    assert path.is_file(), f"{relative} is cited by the README but not committed"
    return json.loads(path.read_text(encoding="utf-8"))


def _genuine_rows() -> list[dict[str, Any]]:
    """The successful, externally-labeled rows the headline is averaged over."""
    rows = _artifact("benchmarks/mercury_benchmark_results.json")["per_dataset"]
    return [r for r in rows if not r.get("error") and r.get("label_source") in _GENUINE_LABELS]


def _match(pattern: str, text: str | None = None) -> re.Match[str]:
    """The single match of ``pattern``, failing loudly on 0 or 2+ hits.

    README prose is the input, so "matched twice" is as much a drift signal as
    "did not match": a duplicated figure means two places now have to be kept
    in step, and the gate should say so rather than silently read the first.
    """
    hits = list(re.finditer(pattern, text if text is not None else _text()))
    assert len(hits) == 1, f"expected exactly one README match for {pattern!r}, got {len(hits)}"
    return hits[0]


def _one(pattern: str, *, text: str | None = None) -> str:
    """The single capture group of a single match."""
    return _match(pattern, text).group(1)


def _groups(pattern: str, *, text: str | None = None) -> tuple[str, ...]:
    """Every capture group of a single match, in order."""
    return _match(pattern, text).groups()


# ---------------------------------------------------------------------------
# Aggregate table (hand-maintained: outside every generated marker block)
# ---------------------------------------------------------------------------


def test_aggregate_std_auc_matches_the_run() -> None:
    """``Std AUC`` is the committed run's own ``summary.std_auc``.

    It read 0.1460 against a run measuring 0.14379 -- neither the population
    (0.1438) nor the sample (0.1451) spread of the genuine-label AUCs, so it
    was a figure from an earlier run that no recomputation reproduced.
    """
    summary = _artifact("benchmarks/mercury_benchmark_results.json")["summary"]
    claimed = float(_one(r"\|\s*Std AUC\s*\|\s*([0-9.]+)\s*\|"))

    assert claimed == pytest.approx(summary["std_auc"], abs=_TOL)


def test_summary_std_auc_is_the_population_spread_of_the_genuine_aucs() -> None:
    """The artifact's own ``std_auc`` is re-derivable from its per-dataset rows."""
    summary = _artifact("benchmarks/mercury_benchmark_results.json")["summary"]
    aucs = [r["ensemble_auc"] for r in _genuine_rows() if r.get("ensemble_auc") is not None]

    assert statistics.pstdev(aucs) == pytest.approx(summary["std_auc"], abs=1e-9)


@pytest.mark.parametrize(
    ("label", "key"),
    [
        ("Mean AUC / Median AUC", "mean_auc"),
        ("Mean / Median Oracle F1", "mean_oracle_f1"),
    ],
)
def test_aggregate_mean_median_pairs_match_the_run(label: str, key: str) -> None:
    """The ``mean / median`` rows quote the committed summary, in that order."""
    summary = _artifact("benchmarks/mercury_benchmark_results.json")["summary"]
    row = _one(rf"\|\s*{re.escape(label)}\s*\|\s*([0-9.]+\s*/\s*[0-9.]+)\s*\|")
    mean_claim, median_claim = (float(part) for part in row.split("/"))

    assert mean_claim == pytest.approx(summary[key], abs=_TOL)
    assert median_claim == pytest.approx(summary[key.replace("mean", "median")], abs=_TOL)


def test_aggregate_row_header_states_the_dataset_counts_it_averages() -> None:
    """``N genuine-label datasets of M successful`` matches the run's own counts."""
    summary = _artifact("benchmarks/mercury_benchmark_results.json")["summary"]
    genuine, successful = (
        int(n)
        for n in _groups(r"\|\s*Aggregate \((\d+) genuine-label datasets of (\d+) successful\)")
    )

    assert genuine == summary["n_genuine_labeled"] == len(_genuine_rows())
    assert successful == summary["successful"]


@pytest.mark.parametrize(
    ("component", "column"),
    [
        ("ResonanceScore", "resonance_auc"),
        ("KinematicScore", "kinematic_auc"),
        ("InfoGeometryScore", "info_geometry_auc"),
    ],
)
def test_component_table_means_and_medians_are_recomputable(component: str, column: str) -> None:
    """Each component row is the mean/median of that column over the genuine rows."""
    row = _groups(rf"\|\s*{component}\s*\|[^|]*\|[^|]*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|")
    values = [r[column] for r in _genuine_rows() if r.get(column) is not None]

    assert float(row[0]) == pytest.approx(statistics.fmean(values), abs=_TOL)
    assert float(row[1]) == pytest.approx(statistics.median(values), abs=_TOL)


# ---------------------------------------------------------------------------
# Failure attribution
# ---------------------------------------------------------------------------


def _failure_split() -> tuple[int, int]:
    """(unreachable, other) over the run's failed rows, by their own messages."""
    rows = _artifact("benchmarks/mercury_benchmark_results.json")["per_dataset"]
    failures = [str(r["error"]) for r in rows if r.get("error")]
    unreachable = [e for e in failures if "unavailable" in e or "download" in e.lower()]
    return len(unreachable), len(failures) - len(unreachable)


def test_readme_attributes_each_failure_to_its_actual_cause() -> None:
    """The failure split is stated, not collapsed into "unreachable".

    All 9 failures were described as unreachable external sources while 2 --
    ``UCR`` and ``USGS_Geochemistry`` -- reached their source and failed on a
    single-valued label column, which is a pipeline result, not a network one.
    """
    unreachable, other = _failure_split()
    total = unreachable + other
    text = _text()

    assert f"Of the {total} failures, {unreachable} were unreachable external" in text
    assert f"Of the {total} failures, {unreachable} are unreachable sources" in text
    assert other == 2, "the degenerate-label failures changed; update the README prose"
    for name in ("UCR", "USGS_Geochemistry"):
        assert name in text, f"{name} fails on degenerate labels but is not named"


def test_degenerate_label_failures_are_what_the_readme_says_they_are() -> None:
    """The 2 non-reachability failures really are single-label-value failures."""
    rows = _artifact("benchmarks/mercury_benchmark_results.json")["per_dataset"]
    degenerate = {
        r["name"]
        for r in rows
        if r.get("error") and "fewer than 2 distinct values" in str(r["error"])
    }

    assert degenerate == {"UCR", "USGS_Geochemistry"}


# ---------------------------------------------------------------------------
# Loader provenance
# ---------------------------------------------------------------------------


def test_loader_counts_match_the_provenance_registry() -> None:
    """``of N concrete loaders, only M`` matches the registry the CI audit reads."""
    from omni_mercury_engine.loaders.label_provenance import (
        GENUINE_LABEL_SOURCES,
        LABEL_PROVENANCE_REGISTRY,
    )

    concrete, genuine = (int(n) for n in _groups(r"of (\d+) concrete\s+loaders, only (\d+) \("))
    actual_genuine = sum(
        1
        for entry in LABEL_PROVENANCE_REGISTRY.values()
        if _label_source(entry) in GENUINE_LABEL_SOURCES
    )

    assert concrete == len(LABEL_PROVENANCE_REGISTRY)
    assert genuine == actual_genuine


def _label_source(entry: Any) -> str:
    """The declared label source, whether the registry stores tuples or records."""
    if isinstance(entry, tuple):
        return str(entry[0])
    return str(getattr(entry, "label_source", entry))


# ---------------------------------------------------------------------------
# Domain hazard table
# ---------------------------------------------------------------------------


def test_hazard_domain_table_matches_the_engine_calibration_run() -> None:
    """Every AUROC in the per-domain table is the committed run's own figure."""
    results = _artifact("research/engine_calibration/results.json")
    claimed = {
        "Earthquake": "earthquake",
        "Pandemic": "pandemic",
        "Energy": "energy",
        "Tsunami": "tsunami",
        "Tornado": "tornado",
        "Network Security": "network_security",
        "Hurricane": "hurricane",
        "FEMA": "fema",
        "Marine": "marine",
    }
    text = _text()
    for label, key in claimed.items():
        value = float(_one(rf"\|\s*{re.escape(label)}\s*\|\s*([0-9.]+)\s*\|", text=text))
        assert value == pytest.approx(results["per_domain"][key]["auroc"], abs=_TOL), label

    overall = float(_one(r"\|\s*Overall\s*\|\s*([0-9.]+)\s*\|", text=text))
    assert overall == pytest.approx(results["overall"]["auroc"], abs=_TOL)


# ---------------------------------------------------------------------------
# Calibration validation
# ---------------------------------------------------------------------------


def test_calibration_lift_row_matches_the_calibration_run() -> None:
    """The MD-011 row's counts, percentage and mean delta all come from the run."""
    md011 = _artifact("benchmarks/calibration_validation_results.json")["summary"]["md_011"]
    improved, total, pct, delta = _groups(
        r"(\d+)/(\d+) datasets improved \((\d+\.\d)% improved-or-same\), " r"mean ΔF1 \+(0\.\d+)"
    )

    assert int(improved) == md011["calibration_improved"]
    assert int(total) == md011["calibration_improved"] + md011["calibration_same"] + (
        md011["calibration_degraded"]
    )
    assert float(pct) == pytest.approx(md011["pct_improved_or_same"], abs=0.05)
    # Quoted to 3 decimals ("+0.220"), so held to half an ulp of that.
    assert float(delta) == pytest.approx(md011["mean_delta_f1"], abs=5e-4)


def test_adaptive_weight_means_match_the_calibration_run() -> None:
    """``mean weights R / K / IG`` are the md_003 weight distribution's means."""
    md003 = _artifact("benchmarks/calibration_validation_results.json")["summary"]["md_003"]
    r, k, ig = (float(v) for v in _groups(r"mean weights R (0\.\d+) / K (0\.\d+) / IG (0\.\d+)"))
    dist = md003["weight_distribution"]

    assert r == pytest.approx(dist["resonance"]["mean"], abs=5e-3)
    assert k == pytest.approx(dist["kinematic"]["mean"], abs=5e-3)
    assert ig == pytest.approx(dist["infogeo"]["mean"], abs=5e-3)


# ---------------------------------------------------------------------------
# Competitive positioning
# ---------------------------------------------------------------------------


def test_competitive_ranks_match_the_competitive_run() -> None:
    """Mercury's placing, and every baseline's mean rank, come from the run."""
    per_method = _artifact("benchmarks/competitive_results.json")["summary"]["per_method"]
    ordered = sorted(per_method.items(), key=lambda kv: kv[1]["mean_rank"])
    place = 1 + next(i for i, (name, _) in enumerate(ordered) if name == "mercury_tier")
    text = _text()

    claimed_place, claimed_of = (
        int(n) for n in _groups(r"places \*\*(\d+)(?:st|nd|rd|th) of (\d+) by mean rank", text=text)
    )
    assert claimed_place == place
    assert claimed_of == len(per_method)

    tier_rank = float(_one(r"by mean rank \((\d+\.\d+)\)", text=text))
    assert tier_rank == pytest.approx(per_method["mercury_tier"]["mean_rank"], abs=5e-3)

    for method, label in [
        ("knn", "knn"),
        ("local_outlier_factor", "local_outlier_factor"),
        ("isolation_forest", "isolation_forest"),
        ("hbos", "hbos"),
        ("copod", "copod"),
        ("ecod", "ecod"),
    ]:
        quoted = float(_one(rf"`{re.escape(label)}` \((\d+\.\d+)\)", text=text))
        assert quoted == pytest.approx(per_method[method]["mean_rank"], abs=5e-3), label


def test_highest_mean_auc_claim_holds_in_the_competitive_run() -> None:
    """ "highest mean ROC-AUC of all eight methods" is true of the committed run."""
    per_method = _artifact("benchmarks/competitive_results.json")["summary"]["per_method"]
    best = max(per_method.items(), key=lambda kv: kv[1]["mean_auc"])[0]
    tier, knn = (
        float(v)
        for v in _groups(r"ROC-AUC of all eight methods\*\* \((0\.\d+) vs `knn`'s (0\.\d+)\)")
    )

    assert best == "mercury_tier"
    assert tier == pytest.approx(per_method["mercury_tier"]["mean_auc"], abs=_TOL)
    assert knn == pytest.approx(per_method["knn"]["mean_auc"], abs=_TOL)


# ---------------------------------------------------------------------------
# Event-protocol benchmarks
# ---------------------------------------------------------------------------


def test_nab_streaming_numbers_match_the_artifact() -> None:
    """Per-stream AUROCs and the mean are the committed NAB evaluation's."""
    nab = _artifact("artifacts/nab_streaming.eval.json")
    streams = [v["auroc"] for v in nab["per_stream"].values()]
    quoted = [
        float(v) for v in _one(r"per-stream AUROC ([0-9. /]+) \(\*\*mean (0\.\d+)\*\*\)").split("/")
    ]
    mean = float(_one(r"per-stream AUROC [0-9. /]+ \(\*\*mean (0\.\d+)\*\*\)"))

    assert quoted == [pytest.approx(s, abs=5e-4) for s in streams]
    assert mean == pytest.approx(nab["mean_auroc"], abs=5e-4)


def test_alfa_uav_numbers_match_the_artifact() -> None:
    """Detections, per-tick false-alarm rate and median delay come from the run."""
    alfa = _artifact("artifacts/alfa_uav.eval.json")
    detected, total = (int(n) for n in _groups(r"\*\*(\d+)/(\d+) fault flights at"))
    far = float(_one(r"fault flights at ([0-9.]+)%\s+per-tick false-alarm rate"))
    delay = float(_one(r"with ([0-9.]+) s median delay"))

    assert detected == alfa["detected_flights"]
    assert total == alfa["detected_flights"] + len(alfa["missed_flights"])
    assert far == pytest.approx(100.0 * alfa["joint_normal_fpr"], abs=5e-3)
    assert delay == pytest.approx(alfa["delay_median_s"], abs=5e-2)


def test_robot_failure_numbers_match_the_artifact() -> None:
    """The four distinct problems and the pooled AUROC are the run's own."""
    robot = _artifact("artifacts/robot_failures.eval.json")
    pooled = float(_one(r"pooled ([0-9.]+)\*\*\n"))
    quoted = [float(v) for v in _one(r"\*\*AUROC ([0-9. /]+) across the four").split("/")]

    # lp3 relabels lp2's executions, so it is excluded from the pooled figure
    # and from the four distinct problems the README quotes.
    distinct = [v["auroc_mean"] for k, v in robot["per_problem"].items() if k != "lp3"]

    assert pooled == pytest.approx(robot["pooled_auroc"], abs=5e-4)
    assert quoted == [pytest.approx(v, abs=5e-4) for v in distinct]


# ---------------------------------------------------------------------------
# Fusion-hardening ledger staleness
# ---------------------------------------------------------------------------


def _last_commit_touching(relative: str) -> str | None:
    """The last commit that changed ``relative``, or None without git."""
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        return None
    try:
        done = subprocess.run(
            [git, "log", "-1", "--format=%H", "--", relative],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return None
    return done.stdout.strip() or None


def test_fusion_hardening_ledger_records_the_detector_it_measured() -> None:
    """The committed ADBench run says which detector revision produced it.

    It did not, which is why a stale number survived: the run was recorded at
    ``bdcdf6a`` and the detector moved seven commits, so the README's
    ``0.7397 -> 0.7634`` ledger described a detector that no longer existed
    and there was nothing in the file to reveal it.
    """
    for relative in (
        "research/omni_equation/adbench_results.json",
        "research/omni_equation/adbench_base_e118e1f.json",
    ):
        summary = _artifact(relative)["summary"]
        assert summary.get("detector_commit"), f"{relative} records no detector_commit"


def test_readme_flags_the_ledger_when_the_detector_has_moved_past_it() -> None:
    """A ledger older than the detector it is quoted for must say so in the README.

    The guard is the pairing, not the number: while the committed run predates
    the live ``detectors/statistical.py``, the README has to carry the caveat;
    once a refresh makes them agree, the caveat is no longer required.
    """
    recorded = _artifact("research/omni_equation/adbench_results.json")["summary"][
        "detector_commit"
    ]
    live = _last_commit_touching("src/omni_mercury_engine/detectors/statistical.py")
    if live is None:
        pytest.skip("no git available to date the detector")
    assert live is not None  # narrow for the checker: pytest.skip never returns

    text = _text()
    caveat = "This ledger is a historical measurement, not the current tree."
    if recorded == live:
        assert caveat not in text, (
            "the committed ADBench run is current again -- drop the staleness caveat "
            "and quote the refreshed ledger"
        )
    else:
        assert caveat in text, (
            f"the committed ADBench run is at {str(recorded)[:7]} but the detector is at "
            f"{live[:7]}; the README quotes the ledger without saying it is historical"
        )
