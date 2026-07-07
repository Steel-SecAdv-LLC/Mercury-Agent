# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Reproducible real-NAB run + before/after analysis for the detector tier.

This is the committed reproduction harness for the detector-tier hardening PR. It
runs :func:`benchmarks.detection_tier_benchmark.run_realdata_benchmark` on the
**real** Numenta Anomaly Benchmark (NAB) and records, side by side:

* the **best single detector** vs the **calibrated ensemble** (the acceptance
  criterion: calibrated ensemble mean ROC-AUC beats the best single detector's
  mean ROC-AUC by > 0.003), and
* **before vs after** the per-detector ensemble calibration -- the same run with
  ``OMNI_ENSEMBLE_CALIBRATION=none`` (raw score averaging, the old behaviour) and
  with ``OMNI_ENSEMBLE_CALIBRATION=rank`` (the empirical-CDF calibration this PR
  makes the default).

Data source
-----------
It first tries the repo's canonical loader
(:func:`benchmarks.detection_tier_benchmark.load_nab_series`, which uses
:class:`omni_mercury_engine.datasets.timeseries.NABLoader`). When that import
chain is unavailable (e.g. a sandbox without the native crypto backend the
``omni_mercury_engine.datasets`` package transitively imports), it falls back to
downloading the identical canonical NAB files directly from the upstream mirror
and reproducing ``NABLoader``'s exact per-point label semantics (a point is
anomalous iff its timestamp lies inside a documented ``combined_windows.json``
window). Both paths feed ``run_realdata_benchmark(datasets=...)``, which scores
and aggregates identically -- so the numbers are the real-NAB numbers either way.

Determinism
-----------
Every ensemble is seeded (``--seed``, default 0); the crop, warm-up, and scoring
are deterministic. Re-running with the same seed and cache reproduces the JSON
byte-for-byte (modulo the ``generated_at`` stamp, which the caller may omit).

Usage::

    python -m benchmarks.reproduce_detection_tier_nab \
        --out benchmarks/detection_tier_nab_results.json \
        --analysis benchmarks/detection_tier_nab_analysis.md
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Canonical NAB real categories + files, mirrored from
# omni_mercury_engine.datasets.timeseries.NABLoader so this harness downloads the
# exact same series the repo loader would, without importing the datasets stack.
NAB_DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/"
NAB_LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
NAB_FILES: dict[str, list[str]] = {
    "realKnownCause": [
        "ambient_temperature_system_failure.csv",
        "cpu_utilization_asg_misconfiguration.csv",
        "ec2_request_latency_system_failure.csv",
        "machine_temperature_system_failure.csv",
        "nyc_taxi.csv",
        "rogue_agent_key_hold.csv",
        "rogue_agent_key_updown.csv",
    ],
    "realAWSCloudwatch": [
        "ec2_cpu_utilization_24ae8d.csv",
        "ec2_cpu_utilization_53ea38.csv",
        "ec2_cpu_utilization_5f5533.csv",
        "ec2_cpu_utilization_77c1ca.csv",
        "ec2_cpu_utilization_825cc2.csv",
        "ec2_cpu_utilization_ac20cd.csv",
        "ec2_cpu_utilization_c6585a.csv",
        "ec2_cpu_utilization_fe7f93.csv",
        "ec2_disk_write_bytes_1ef3de.csv",
        "ec2_disk_write_bytes_c0d644.csv",
        "ec2_network_in_257a54.csv",
        "ec2_network_in_5abac7.csv",
        "elb_request_count_8c0756.csv",
        "grok_asg_anomaly.csv",
        "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
        "rds_cpu_utilization_cc0c53.csv",
        "rds_cpu_utilization_e47b3b.csv",
    ],
    "realTraffic": [
        "TravelTime_387.csv",
        "TravelTime_451.csv",
        "occupancy_6005.csv",
        "occupancy_t4013.csv",
        "speed_6005.csv",
        "speed_7578.csv",
        "speed_t4013.csv",
    ],
}


def _cache_dir() -> Path:
    root = os.getenv("MERCURY_DATA_DIR", "./data")
    path = Path(root) / "nab"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _http_get(url: str) -> bytes:
    """GET a URL through the environment proxy/CA (requests honours the env vars)."""
    import requests

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_http_get(url))


def _parse_windows(labels_json: dict[str, Any], rel_path: str) -> list[tuple[datetime, datetime]]:
    ranges: list[tuple[datetime, datetime]] = []
    for window in labels_json.get(rel_path, []):
        try:
            start = datetime.fromisoformat(window[0].replace("Z", "+00:00"))
            end = datetime.fromisoformat(window[1].replace("Z", "+00:00"))
            ranges.append((start, end))
        except (ValueError, IndexError):
            continue
    return ranges


def _parse_nab_csv(
    text: str, ranges: list[tuple[datetime, datetime]]
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce NABLoader._parse_nab_file: value channel + per-point window labels."""
    values: list[float] = []
    labels: list[int] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            value = float(row.get("value", 0))
        except (ValueError, TypeError):
            continue
        ts_str = (row.get("timestamp", "") or "").replace("Z", "+00:00")
        is_anom = False
        try:
            ts = datetime.fromisoformat(ts_str)
            for start, end in ranges:
                if start <= ts <= end:
                    is_anom = True
                    break
        except ValueError:
            pass
        values.append(value)
        labels.append(1 if is_anom else 0)
    return np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _download_nab_series(
    categories: tuple[str, ...], max_len: int, max_files: int | None
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Direct-download fallback: canonical NAB series with NABLoader label semantics."""
    from benchmarks.detection_tier_benchmark import _crop_to_anomaly

    cache = _cache_dir()
    labels_path = cache / "labels.json"
    _download_file(NAB_LABELS_URL, labels_path)
    labels_json = json.loads(labels_path.read_text())

    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for category in categories:
        for filename in sorted(NAB_FILES.get(category, [])):
            rel = f"{category}/{filename}"
            dest = cache / category / filename
            _download_file(f"{NAB_DATA_URL}{rel}", dest)
            ranges = _parse_windows(labels_json, rel)
            values, labels = _parse_nab_csv(dest.read_text(), ranges)
            if values.size == 0:
                continue
            series, lab = _crop_to_anomaly(values, labels, max_len)
            if int(lab.sum()) == 0:
                continue
            out.append((rel, series, lab))
            if max_files is not None and len(out) >= max_files:
                return out
    return out


def load_series(
    categories: tuple[str, ...], max_len: int, max_files: int | None
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], str]:
    """Load real NAB series, preferring the repo loader, else the direct download."""
    try:
        from benchmarks.detection_tier_benchmark import load_nab_series

        series = load_nab_series(categories=categories, max_len=max_len, max_files=max_files)
        return series, "omni_mercury_engine.datasets.timeseries.NABLoader"
    except Exception as exc:
        print(
            f"[repro] repo NAB loader unavailable ({type(exc).__name__}: {exc}); "
            f"using direct upstream download",
            file=sys.stderr,
        )
        series = _download_nab_series(categories, max_len, max_files)
        return series, "direct upstream download (raw.githubusercontent.com/numenta/NAB)"


def _best_member(members_agg: dict[str, Any]) -> tuple[str | None, float | None]:
    measurable = {k: v for k, v in members_agg.items() if "mean_auc" in v}
    if not measurable:
        return None, None
    best = max(measurable.items(), key=lambda kv: kv[1]["mean_auc"])
    return best[0], float(best[1]["mean_auc"])


def _run_once(
    datasets: list[tuple[str, np.ndarray, np.ndarray]], seed: int, calibration: str
) -> dict[str, Any]:
    """Run the tier benchmark with a fixed ensemble calibration (via the env knob)."""
    from benchmarks.detection_tier_benchmark import run_realdata_benchmark

    prev = os.environ.get("OMNI_ENSEMBLE_CALIBRATION")
    os.environ["OMNI_ENSEMBLE_CALIBRATION"] = calibration
    try:
        results = run_realdata_benchmark(seed=seed, datasets=datasets)
    finally:
        if prev is None:
            os.environ.pop("OMNI_ENSEMBLE_CALIBRATION", None)
        else:
            os.environ["OMNI_ENSEMBLE_CALIBRATION"] = prev
    return results


def run(
    seed: int = 0,
    max_len: int = 6000,
    max_files: int | None = None,
    categories: tuple[str, ...] = ("realKnownCause", "realAWSCloudwatch", "realTraffic"),
) -> dict[str, Any]:
    """Load real NAB, run before/after ensemble calibration, and summarise the deltas."""
    datasets, source = load_series(categories, max_len, max_files)
    if not datasets:
        raise RuntimeError("no NAB series with a labelled anomaly were loaded")

    before = _run_once(datasets, seed, "none")  # raw score averaging (old behaviour)
    after = _run_once(datasets, seed, "rank")  # empirical-CDF calibration (new default)

    def summarise(results: dict[str, Any], ensemble_key: str) -> dict[str, Any]:
        members = results["aggregate"]["members"]
        best_name, best_auc = _best_member(members)
        ens = results["aggregate"][ensemble_key]
        ens_auc = ens.get("mean_auc")
        delta = (ens_auc - best_auc) if (ens_auc is not None and best_auc is not None) else None
        return {
            "best_member": best_name,
            "best_member_mean_auc": best_auc,
            "ensemble_mean_auc": ens_auc,
            "ensemble_minus_best_member": delta,
            "ensemble_mean_f1": ens.get("mean_f1"),
            "n_datasets": results["config"]["n_datasets_measured"],
        }

    # "before" = the pre-PR behaviour: raw-score averaging (no calibration).
    # "after"  = the new default: rank/ECDF calibration + robust consensus combiner.
    before_sum = summarise(before, "ensemble_average")
    after_sum = summarise(after, "ensemble_consensus")
    return {
        "data_source": source,
        "seed": seed,
        "max_len": max_len,
        "n_datasets": len(datasets),
        "before_calibration": {"calibration": "none", "combiner": "average", **before_sum},
        "after_calibration": {"calibration": "rank", "combiner": "consensus", **after_sum},
        "improvement_ensemble_vs_best_member": after_sum["ensemble_minus_best_member"],
        "acceptance_threshold": 0.003,
        "meets_acceptance": bool(
            after_sum["ensemble_minus_best_member"] is not None
            and after_sum["ensemble_minus_best_member"] > 0.003
        ),
        "raw_after": after,
        "raw_before": before,
    }


def _fmt_metric(value: Any, placeholder: str = "n/a") -> str:
    """Format a metric as ``.4f``, degrading to ``placeholder`` when it is ``None``.

    ``run()`` sets the AUC / improvement fields to ``None`` on the error path where
    the ensemble or the best member has no measurable score (see ``summarise``'s
    ``delta`` and ``meets_acceptance``'s explicit ``is not None`` guard). Formatting
    such a value with a bare ``:.4f`` raises ``TypeError`` and crashes the report
    writer, so every human-facing metric is routed through this guard.
    """
    return f"{value:.4f}" if isinstance(value, (int, float)) else placeholder


def _analysis_markdown(summary: dict[str, Any]) -> str:
    b = summary["before_calibration"]
    a = summary["after_calibration"]
    lines = [
        "# Detector tier -- real NAB before/after analysis",
        "",
        f"- **Data source:** {summary['data_source']}",
        f"- **Series measured:** {summary['n_datasets']}  |  **seed:** {summary['seed']}  "
        f"|  **max_len:** {summary['max_len']}",
        "",
        "## Ensemble vs best single detector (mean ROC-AUC over all series)",
        "",
        "| pipeline | combiner | calibration | best single detector | best-single AUC | "
        "ensemble AUC | ensemble - best | ensemble F1 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    def _row(tag: str, s: dict[str, Any]) -> str:
        def fmt(x: Any) -> str:
            return _fmt_metric(x, placeholder="-")

        return (
            f"| {tag} | `{s['combiner']}` | `{s['calibration']}` | {s['best_member']} | "
            f"{fmt(s['best_member_mean_auc'])} | {fmt(s['ensemble_mean_auc'])} | "
            f"{fmt(s['ensemble_minus_best_member'])} | {fmt(s['ensemble_mean_f1'])} |"
        )

    lines.append(_row("before (pre-PR)", b))
    lines.append(_row("after (this PR)", a))
    lines += [
        "",
        f"**Calibrated consensus ensemble beats the best single detector by "
        f"{_fmt_metric(summary['improvement_ensemble_vs_best_member'])} ROC-AUC** "
        f"(acceptance threshold > {summary['acceptance_threshold']}): "
        f"{'PASS' if summary['meets_acceptance'] else 'FAIL'}.",
        "",
        "The `before` row is the pre-PR pipeline: raw-score averaging with no "
        "per-detector calibration (`OMNI_ENSEMBLE_CALIBRATION=none`, combiner "
        "`average`). The `after` row is this PR's default pipeline: per-detector "
        "empirical-CDF calibration (`rank`) plus the robust high-quantile "
        "`consensus` combiner. Both rows are the *same detectors on the same real "
        "NAB series*; only the ensemble's calibration + combination differ. A plain "
        "mean of anomaly scores is known to be dominated by robust rank aggregation "
        "for outlier ensembles (Aggarwal & Sathe, 2017); the consensus combiner is "
        "not dragged toward 0.5 by the uninformative members a mean averages in.",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point: run the real-NAB before/after benchmark and write artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=6000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--out", type=str, default="benchmarks/detection_tier_nab_results.json")
    parser.add_argument("--analysis", type=str, default="benchmarks/detection_tier_nab_analysis.md")
    args = parser.parse_args()

    summary = run(seed=args.seed, max_len=args.max_len, max_files=args.max_files)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    Path(args.analysis).write_text(_analysis_markdown(summary))

    a = summary["after_calibration"]
    print(f"data source: {summary['data_source']}")
    print(
        f"series: {summary['n_datasets']}  best single: {a['best_member']} "
        f"AUC={_fmt_metric(a['best_member_mean_auc'])}  "
        f"consensus ensemble AUC={_fmt_metric(a['ensemble_mean_auc'])}"
    )
    print(
        f"ensemble - best single = {_fmt_metric(summary['improvement_ensemble_vs_best_member'])} "
        f"(threshold > 0.003): {'PASS' if summary['meets_acceptance'] else 'FAIL'}"
    )
    print(f"wrote {out_path} and {args.analysis}")


if __name__ == "__main__":
    main()
