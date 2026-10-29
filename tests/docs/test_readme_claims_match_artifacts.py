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
    # ``ground_truth_loader_keys`` is the module's own exported answer to
    # "which loaders carry labels independent of a scored feature" -- the same
    # one the CI audit prints -- so the gate reads that rather than
    # re-implementing the predicate over a non-exported constant.
    from omni_mercury_engine.loaders.label_provenance import (
        LABEL_PROVENANCE_REGISTRY,
        ground_truth_loader_keys,
    )

    concrete, genuine = (int(n) for n in _groups(r"of (\d+) concrete\s+loaders, only (\d+) \("))
    actual_genuine = len(ground_truth_loader_keys())

    assert concrete == len(LABEL_PROVENANCE_REGISTRY)
    assert genuine == actual_genuine


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

    overall = float(_one(r"\|\s*\*Overall\*\s*\|\s*\*([0-9.]+)\*\s*\|", text=text))
    assert overall == pytest.approx(results["overall"]["auroc"], abs=_TOL)


def test_each_hazard_row_states_its_loaders_declared_provenance() -> None:
    """The table's Labels column is the loader's own ``LABEL_SOURCE``.

    The table used to print nine AUROCs with no provenance at all, while eight
    of the nine loaders declared ``statistical`` at source -- the label is a
    threshold on a column that is also scored. Pinning the column to the
    declaration is what stops the two drifting apart again.
    """
    import importlib

    from omni_mercury_engine.loaders.label_provenance import (
        GENUINE_LABEL_SOURCES,
        LABEL_PROVENANCE_REGISTRY,
    )

    rows = re.findall(
        r"\|\s*(Network Security|Earthquake|Pandemic|Energy|Tsunami|Tornado|Hurricane|FEMA|"
        r"Marine)\s*\|\s*[0-9.]+\s*\|\s*([^|]+?)\s*\|",
        _text(),
    )
    assert len(rows) == 9, f"expected 9 hazard rows, matched {len(rows)}"

    for label, stated in rows:
        key = next(
            k for k in LABEL_PROVENANCE_REGISTRY if k.split("_loader")[0] == _loader_stem(label)
        )
        mod, cname = key.split(".")
        cls = getattr(importlib.import_module(f"omni_mercury_engine.loaders.{mod}"), cname)
        genuine = cls.LABEL_SOURCE in GENUINE_LABEL_SOURCES
        says_independent = "independent" in stated.lower()
        assert says_independent == genuine, (
            f"{label}: README says {stated!r} but the loader declares "
            f"LABEL_SOURCE={cls.LABEL_SOURCE!r}"
        )


def _loader_stem(readme_label: str) -> str:
    """Map a README domain label onto its loader module stem."""
    return readme_label.lower().replace(" ", "_")


def test_deleak_impact_figures_match_the_evidence_artifact() -> None:
    """The quoted de-leak deltas are the measured ones, including the control.

    The claim that matters is comparative -- earthquake's fall against the
    drift of the domains that excluded nothing -- so both halves are pinned.
    """
    impact = _artifact("research/engine_calibration/deleak_impact.json")
    eq = impact["per_domain"]["earthquake"]
    text = _text()

    before, after, delta = _groups(
        r"Earthquake falls\s*\n?\*\*(0\.\d+) → (0\.\d+) \(−(0\.\d+)\)\*\*", text=text
    )
    assert float(before) == pytest.approx(eq["auroc_circular"], abs=_TOL)
    assert float(after) == pytest.approx(eq["auroc_deleaked"], abs=_TOL)
    assert float(delta) == pytest.approx(abs(eq["delta"]), abs=_TOL)

    band = float(_one(r"moved by up to\s*\n?(0\.\d+) on the same events", text=text))
    # Quoted to 3 decimals ("0.119"), so held to half an ulp of that.
    assert band == pytest.approx(impact["no_exclusion_control_band"]["max_abs_delta"], abs=5e-4)
    assert abs(eq["delta"]) > band, "the headline claim requires the delta to exceed the band"


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


def test_readme_ledger_is_not_quoted_past_the_detector_that_produced_it() -> None:
    """A ledger older than the detector it is quoted for must be flagged.

    The guard is the pairing, not the number. While the committed run matches
    the live ``detectors/statistical.py``, the README quotes it as current and
    no staleness caveat belongs there; the moment the detector moves ahead, the
    caveat becomes mandatory until someone refreshes both sides.
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
            "the committed ADBench run matches the live detector -- drop the staleness "
            "caveat and quote the refreshed ledger"
        )
    else:
        assert caveat in text, (
            f"the committed ADBench run is at {str(recorded)[:7]} but the detector is at "
            f"{live[:7]}; the README quotes the ledger without saying it is historical"
        )


def test_readme_ledger_numbers_match_the_committed_comparison() -> None:
    """The quoted mean, delta and win/tie/loss counts are the ledger's own.

    The previous ledger drifted because nothing re-derived it: the figures sat
    in prose while the comparison file moved underneath them.
    """
    ledger = _artifact("research/omni_equation/adbench_base_vs_current.json")["summary"]
    base, current, delta, wins, ties, losses = _groups(
        r"Mean AUROC \*\*(0\.\d+) \u2192 (0\.\d+) \(\+(\d+\.\d+) pts, "
        r"(\d+) W / (\d+) tie / (\d+) L\)\*\*"
    )

    assert float(base) == pytest.approx(ledger["mean_baseline"], abs=5e-5)
    assert float(current) == pytest.approx(ledger["mean_current"], abs=5e-5)
    # The prose quotes the delta in points, the ledger stores it as a fraction.
    assert float(delta) == pytest.approx(100.0 * ledger["mean_delta"], abs=5e-3)
    assert int(wins) == ledger["wins"]
    assert int(ties) == ledger["ties"]
    assert int(losses) == ledger["losses"]


def test_the_base_side_of_the_ledger_is_the_commit_the_readme_names() -> None:
    """The ledger's baseline really is the `e118e1f` the README cites."""
    ledger = _artifact("research/omni_equation/adbench_base_vs_current.json")["summary"]
    named = _one(r"from base `([0-9a-f]{7})` to the detector at `([0-9a-f]{7})`")
    recorded = _artifact("research/omni_equation/adbench_results.json")["summary"][
        "detector_commit"
    ]

    assert str(ledger["baseline_commit"]).startswith(named)
    assert str(recorded).startswith(
        _groups(r"from base `([0-9a-f]{7})` to the detector at `([0-9a-f]{7})`")[1]
    )


# ---------------------------------------------------------------------------
# Claims the first sweep of this gate did not cover
# ---------------------------------------------------------------------------


def test_benchmark_coverage_row_splits_both_denominators() -> None:
    """The capability row separates the successful split from the attempted one.

    It read "67 reproducible datasets (of 76 attempted; 47 ADBench + 29
    domain)", where 47 + 29 totals the *attempted* 76 while the sentence reads
    as the composition of the reproducible 67. The successful split is
    47 ADBench + 20 domain.
    """
    rows = _artifact("benchmarks/mercury_benchmark_results.json")["per_dataset"]
    successful = [r for r in rows if not r.get("error")]
    adbench_ok = [r for r in successful if r["category"] == "adbench"]
    adbench_all = [r for r in rows if r["category"] == "adbench"]

    ok_total, ok_adb, ok_dom, att_total, att_adb, att_dom = (
        int(n)
        for n in _groups(
            r"(\d+) reproducible datasets \((\d+) ADBench \+ (\d+) domain\) of "
            r"(\d+) attempted \((\d+) ADBench \+ (\d+) domain\)"
        )
    )

    assert (ok_adb, ok_dom) == (len(adbench_ok), len(successful) - len(adbench_ok))
    assert ok_adb + ok_dom == ok_total == len(successful)
    assert (att_adb, att_dom) == (len(adbench_all), len(rows) - len(adbench_all))
    assert att_adb + att_dom == att_total == len(rows)


def test_stated_caveats_are_the_measured_ones() -> None:
    """The "when to use Mercury" caveats quote the run, not an impression.

    Both are checkable: the KinematicScore mean, and how many genuine-label
    datasets the ensemble inverts below chance.
    """
    gen = _genuine_rows()
    kinematic = float(_one(r"near-random on shuffled tabular data \(mean AUC (0\.\d+)\)"))
    inverted = int(_one(r"(\d+) datasets fall below 0\.50\s*\nAUC"))
    measured_inverted = [
        r["name"] for r in gen if r.get("ensemble_auc") is not None and r["ensemble_auc"] < 0.50
    ]

    assert kinematic == pytest.approx(statistics.fmean([r["kinematic_auc"] for r in gen]), abs=5e-3)
    assert inverted == len(measured_inverted), f"below-chance sets are {measured_inverted}"


def test_nsl_kdd_figure_matches_its_row() -> None:
    """The validation-status row's NSL-KDD AUC is the run's own figure."""
    gen = _genuine_rows()
    nsl = next(r for r in gen if r["name"] == "NSL-KDD")
    quoted = float(_one(r"NSL-KDD benchmarked \(AUC (0\.\d+), table above\)"))

    assert quoted == pytest.approx(nsl["ensemble_auc"], abs=5e-4)


def test_regression_floors_match_the_workflow_that_enforces_them() -> None:
    """The quoted coarse floors are the ones ``benchmark.yml`` actually gates on."""
    workflow = (_REPO / ".github" / "workflows" / "benchmark.yml").read_text(encoding="utf-8")
    auc_floor, f1_floor = (
        float(n) for n in _groups(r"ROC-AUC must stay ≥ (\d\.\d+) and Mean Oracle F1 ≥ (\d\.\d+)")
    )

    assert f'MIN_ROC_AUC: "{auc_floor}"' in workflow, "quoted AUC floor is not the gated one"
    assert f'MIN_F1: "{f1_floor}"' in workflow, "quoted F1 floor is not the gated one"


def test_sklearn_spot_check_table_matches_its_committed_run() -> None:
    """Every cell of the baseline spot check is the committed run's own figure.

    The table cited a command and a date but not a file, so nothing re-derived
    it -- the only README table left unguarded after the first sweep. The run
    is deterministic at ``--seed 42``: re-running it reproduced all 20 cells.
    """
    rows = _artifact("results/empirical_benchmark_results.json")["results"]
    measured = {(r["detector_name"], r["dataset_name"]): r["roc_auc"] for r in rows}
    datasets = ("breast_cancer", "covtype", "digits_8", "kddcup99")
    detectors = {
        "Mercury-Agent": r"\*\*Mercury-Agent\*\*",
        "IsolationForest": "Isolation Forest",
        "OneClassSVM": "One-Class SVM",
        "LocalOutlierFactor": "LOF",
        "EllipticEnvelope": "Elliptic Envelope",
    }
    cell = r"\s*\|\s*\**([0-9.]+)\**"

    for key, label in detectors.items():
        quoted = _groups(rf"\|\s*{label}{cell}{cell}{cell}{cell}\s*\|")
        for dataset, value in zip(datasets, quoted, strict=True):
            assert (key, dataset) in measured, f"{key}/{dataset} is not in the committed run"
            # The table quotes 3 decimals, so it is held to half an ulp of that.
            assert float(value) == pytest.approx(
                measured[(key, dataset)], abs=5e-4
            ), f"{key}/{dataset}"


def test_spot_check_standing_matches_the_committed_run() -> None:
    """ "#4 of 5 by mean AUC, 0.736 vs the 0.829 best baseline" is the run's own."""
    summary = _artifact("results/empirical_benchmark_results.json")["summary"][
        "omni_mercury_comparison"
    ]
    rank, of, mercury, best = _groups(
        r"#(\d+) of (\d+) by mean AUC, (0\.\d+) vs the\s*\n?(0\.\d+) best baseline"
    )

    assert int(rank) == summary["rank_by_roc_auc"]
    assert float(mercury) == pytest.approx(summary["omni_mercury_roc_auc"], abs=5e-4)
    assert float(best) == pytest.approx(summary["best_baseline_roc_auc"], abs=5e-4)
    assert int(of) == len(
        _artifact("results/empirical_benchmark_results.json")["summary"]["per_detector"]
    )


def test_head_to_head_and_loss_margin_match_the_competitive_run() -> None:
    """The LOF record and the median margin of the losses to knn are the run's.

    Verified by hand during the v3 audit and guarded here so the next
    competitive run cannot leave them behind.
    """
    doc = _artifact("benchmarks/competitive_results.json")
    head = doc["summary"]["head_to_head"]["mercury_tier"]["local_outlier_factor"]
    rows = doc["per_dataset"]
    text = _text()

    wins, losses = (int(n) for n in _groups(r"LOF\s*\nhead-to-head reads (\d+)W/(\d+)L", text=text))
    assert (wins, losses) == (head["wins"], head["losses"])

    def _auc(row: dict[str, Any], method: str) -> float | None:
        cell = row.get("methods", row).get(method)
        return cell.get("roc_auc") if isinstance(cell, dict) else None

    margins = sorted(
        knn - mercury
        for row in rows
        if (mercury := _auc(row, "mercury_tier")) is not None
        and (knn := _auc(row, "knn")) is not None
        and knn > mercury
    )
    n_losses, quoted_margin = _groups(
        r"the (\d+) losses to `knn` carry a median margin\s*\nof (0\.\d+) AUC"
    )

    assert int(n_losses) == len(margins)
    # The prose quotes 3 decimals, so it is held to half an ulp of that.
    assert float(quoted_margin) == pytest.approx(statistics.median(margins), abs=5e-4)


def test_quickstart_transcript_matches_a_real_run() -> None:
    """The default quickstart figures are the committed benchmark's own.

    The harness row the transcript prints is the ``DTU_AIS_Abnormal`` row of
    ``mercury_benchmark_results.json``, which the quickstart's own closing line
    promises. Re-running the quickstart during the v3 audit reproduced the
    whole transcript, interval bounds included.
    """
    gen = _genuine_rows()
    dtu = next(r for r in gen if r["name"] == "DTU_AIS_Abnormal")
    harness = float(_one(r"all 25 abnormal: ROC-AUC (0\.\d+) \[95% CI"))

    assert harness == pytest.approx(dtu["ensemble_auc"], abs=5e-4)


def test_quoted_interval_widths_are_the_widths_of_the_quoted_intervals() -> None:
    """ "the intervals span about X of AUC" is arithmetic on the printed bounds.

    It read 0.16 against bounds of 0.821-0.955 (0.134) and 0.809-0.938
    (0.129) -- a figure no subtraction in the section produces. The sentence
    is about the two published-protocol intervals, so those two are the ones
    checked, located by the point estimates the same paragraph names.
    """
    claimed = float(_one(r"the intervals span about (0\.\d+) of AUC"))
    text = _text()
    widths = []
    for point in _groups(r"\*\*Published protocols \((0\.\d+), (0\.\d+)\)\*\*"):
        low, high = _groups(rf"ROC-AUC {point} \[95% CI (0\.\d+)-(0\.\d+)\]", text=text)
        widths.append(float(high) - float(low))

    assert len(widths) == 2, "the paragraph no longer names two published protocols"
    for width in widths:
        assert width == pytest.approx(claimed, abs=0.01), (
            f"claimed span {claimed} is not the width of a printed interval "
            f"({[round(w, 3) for w in widths]})"
        )
