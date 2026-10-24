# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
r"""Regenerate the README's benchmark blocks from ``benchmarks/mercury_benchmark_results.json``.

Two marker blocks are rewritten: ``<!-- BENCHMARK:START -->`` … ``END``
(headline, current vs previous run) and ``<!-- BENCHMARK-DOMAINS:START -->``
… ``END`` (per-domain means over the genuinely labelled datasets).

This script is invoked by ``.github/workflows/benchmark.yml`` after every
benchmark run on ``main`` so the live-data results are auto-committed and
permanently visible at the top of the README — never stuck in 90-day-expiring
CI artifacts.

The "previous" snapshot is read by asking ``git`` for the last committed copy
of the result file.  If there is no previous snapshot (first run) the diff
columns render as ``—``.

Usage:
    python scripts/update_readme_benchmarks.py \\
        --results benchmarks/mercury_benchmark_results.json \\
        --readme README.md \\
        [--commit-sha <sha>]

Exit codes:
    0  README updated (or already up-to-date).
    2  Results file missing or malformed.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

START_MARKER = "<!-- BENCHMARK:START -->"
END_MARKER = "<!-- BENCHMARK:END -->"
DOMAINS_START_MARKER = "<!-- BENCHMARK-DOMAINS:START -->"
DOMAINS_END_MARKER = "<!-- BENCHMARK-DOMAINS:END -->"

#: Label sources that count as independent of the scored signal; the same
#: policy as ``summary.headline_label_policy``. ``statistical`` labels are
#: manufactured by thresholding a scored feature and are excluded.
GENUINE_LABEL_SOURCES = frozenset({"ground_truth", "expert_annotated"})

_DOMAIN_LABELS = {"adbench": "ADBench", "timeseries": "Time series"}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            # ``json.load`` returns ``Any``; the result file is a JSON object
            # (``main`` verifies the ``summary`` block before use).
            return cast("dict[str, Any]", json.load(fh))
    except (OSError, json.JSONDecodeError):
        return None


def _previous_snapshot(path: Path) -> dict[str, Any] | None:
    """Return the previously committed copy of *path* via ``git show``.

    Returns ``None`` if the file is not in HEAD or git is not available
    (e.g. shallow clone with no history for the file).
    """
    try:
        completed = subprocess.run(  # noqa: S603
            ["git", "show", f"HEAD:{path.as_posix()}"],  # noqa: S607
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    try:
        # ``json.loads`` returns ``Any``; the committed snapshot has the same
        # JSON-object shape as the current results file.
        return cast("dict[str, Any]", json.loads(completed.stdout.decode("utf-8")))
    except json.JSONDecodeError:
        return None


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary") or {}
    # ``benchmarks/mercury_benchmark.py`` writes provenance under
    # ``data["metadata"]`` (git_commit, timestamp, python_version, …).
    # Older fixtures and externally-produced result files used to put
    # the same fields at the top level (``data["commit"]``,
    # ``data["timestamp"]``, ``data["run_timestamp"]``, …).  The
    # canonical nested path takes precedence; the flat fallbacks are
    # kept so the script still renders historical runs cleanly.
    metadata = data.get("metadata") or {}
    return {
        "mean_auc": summary.get("mean_auc"),
        "median_auc": summary.get("median_auc"),
        "mean_oracle_f1": summary.get("mean_oracle_f1"),
        "successful": summary.get("successful"),
        # ``benchmarks/mercury_benchmark.py`` writes the attempted-dataset count
        # under ``summary["total_datasets"]`` (alongside ``successful`` /
        # ``failed``).  Older/external fixtures used ``total`` or ``n_datasets``.
        # ``total_datasets`` must be in this chain or the ratio silently collapses
        # to ``successful / successful`` (e.g. ``65 / 65`` instead of ``65 / 75``).
        "total": (
            summary.get("total")
            or summary.get("total_datasets")
            or summary.get("n_datasets")
            or summary.get("successful")
        ),
        "timestamp": (
            metadata.get("timestamp") or data.get("timestamp") or data.get("run_timestamp")
        ),
        "commit": (metadata.get("git_commit") or data.get("commit") or data.get("git_commit")),
    }


def _fmt(value: Any, *, kind: str = "float") -> str:
    if value is None:
        return "—"
    if kind == "float":
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _delta(current: Any, previous: Any) -> str:
    if current is None or previous is None:
        return "—"
    try:
        diff = float(current) - float(previous)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.4f}"


def render_block(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    commit_sha: str | None,
) -> str:
    """Render the README block between START_MARKER and END_MARKER."""
    cur = _summary(current)
    prev = _summary(previous) if previous else {}

    if commit_sha and not cur["commit"]:
        cur["commit"] = commit_sha

    rows = [
        ("Mean ROC-AUC", "mean_auc", "float"),
        ("Median ROC-AUC", "median_auc", "float"),
        ("Mean Oracle F1", "mean_oracle_f1", "float"),
    ]
    table_rows: list[str] = []
    for label, key, kind in rows:
        c = cur.get(key)
        p = prev.get(key)
        table_rows.append(
            f"| {label} | {_fmt(c, kind=kind)} | {_fmt(p, kind=kind)} | {_delta(c, p)} |"
        )

    successful_current = cur.get("successful")
    total_current = cur.get("total")
    successful_previous = prev.get("successful")
    total_previous = prev.get("total")
    table_rows.append(
        "| Datasets (successful / total) | "
        f"{_fmt(successful_current, kind='int')} / {_fmt(total_current, kind='int')} | "
        f"{_fmt(successful_previous, kind='int')} / {_fmt(total_previous, kind='int')} | "
        f"{_delta(successful_current, successful_previous)} |"
    )

    table_rows.append(
        f"| Run timestamp (UTC) | {_fmt(cur.get('timestamp'), kind='int')} | "
        f"{_fmt(prev.get('timestamp'), kind='int')} | — |"
    )

    cur_commit = cur.get("commit") or "—"
    prev_commit = prev.get("commit") or "—"
    cur_commit_short = cur_commit[:7] if isinstance(cur_commit, str) and cur_commit != "—" else "—"
    prev_commit_short = (
        prev_commit[:7] if isinstance(prev_commit, str) and prev_commit != "—" else "—"
    )
    table_rows.append(f"| Commit | `{cur_commit_short}` | `{prev_commit_short}` | — |")

    table = "\n".join(table_rows)

    return f"""{START_MARKER}
## Latest Benchmark Results

> *This block is regenerated by `.github/workflows/benchmark.yml` on every
> push to `main` and committed back to the repo, so the most recent live-data
> run is always front-and-center — never lost to expiring CI artifacts.*

The full result file lives at [`benchmarks/mercury_benchmark_results.json`](benchmarks/mercury_benchmark_results.json).
Methodology is documented in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
A multi-panel visual summary appears in the [Current Benchmarks and Visual Proof](#current-benchmarks-and-visual-proof) section below.

| Metric | Current | Previous | Δ |
|---|---|---|---|
{table}

Regression gates: ROC-AUC must stay ≥ 0.75 and Mean Oracle F1 ≥ 0.55 (the coarse floors set in `.github/workflows/benchmark.yml`, held a deliberate margin below the current headline AUC/F1 so they do not flap on normal load-availability variance). Fine-grained, tight regression protection is the deterministic per-dataset guard (`anomaly_regression_guard.py`), not these coarse floors. CI fails the workflow if either drops below threshold.
{END_MARKER}"""


def render_domain_block(current: dict[str, Any]) -> str:
    """Render per-domain means over successful, genuinely labelled datasets.

    Domains are the records' ``category``, largest first; small domains list
    their datasets by name. Mean oracle F1 is taken over the records that
    carry one.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in current.get("per_dataset") or []:
        if (
            record.get("error") in (None, "", False)
            and record.get("label_source") in GENUINE_LABEL_SOURCES
            and isinstance(record.get("ensemble_auc"), (int, float))
        ):
            groups.setdefault(str(record.get("category") or "uncategorised"), []).append(record)

    rows: list[str] = []
    for category in sorted(groups, key=lambda c: (-len(groups[c]), c)):
        records = groups[category]
        label = _DOMAIN_LABELS.get(category, category.replace("_", " ").capitalize())
        if len(records) <= 3:
            label += " (" + ", ".join(str(r.get("name")) for r in records) + ")"
        f1 = [r["oracle_f1"] for r in records if isinstance(r.get("oracle_f1"), (int, float))]
        rows.append(
            f"| {label} | {len(records)} | "
            f"{statistics.mean(r['ensemble_auc'] for r in records):.4f} | "
            f"{_fmt(statistics.mean(f1) if f1 else None)} |"
        )
    n_datasets = sum(len(records) for records in groups.values())
    n_adbench = len(groups.get("adbench", []))
    comparable = (
        f"; the comparable figure is the\n{n_adbench}-dataset ADBench row" if n_adbench else ""
    )
    table = "\n".join(rows)
    return f"""{DOMAINS_START_MARKER}
| Domain | Datasets | Mean AUC | Mean oracle F1 |
|--------|----------|----------|----------------|
{table}

*Generated from `benchmarks/mercury_benchmark_results.json` by
`scripts/update_readme_benchmarks.py`: the {n_datasets} successful datasets whose
labels are `ground_truth` or `expert_annotated`. Threshold-derived
(`statistical`) labels are excluded as circular. Domains of one to three
datasets are easy to separate at that size{comparable}.*
{DOMAINS_END_MARKER}"""


def update_readme(
    readme_path: Path,
    new_block: str,
    start_marker: str = START_MARKER,
    end_marker: str = END_MARKER,
) -> bool:
    """Replace one marker block in *readme_path*.  Returns ``True`` if changed."""
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        msg = (
            f"could not find '{start_marker}' … '{end_marker}' markers in "
            f"{readme_path}; aborting"
        )
        raise SystemExit(msg)
    # A callable replacement: the block is literal text, never a regex template.
    new_text = pattern.sub(lambda _match: new_block, text)
    if new_text == text:
        return False
    readme_path.write_text(new_text, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmarks/mercury_benchmark_results.json"),
        help="Path to current results JSON.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to README.md.",
    )
    parser.add_argument(
        "--commit-sha",
        default=None,
        help="Optional commit SHA to record (defaults to whatever is in the results JSON).",
    )
    args = parser.parse_args(argv)

    current = _load_json(args.results)
    if current is None:
        print(f"ERROR: cannot read {args.results}", file=sys.stderr)
        return 2
    if "summary" not in current:
        print(f"ERROR: {args.results} has no 'summary' block", file=sys.stderr)
        return 2

    previous = _previous_snapshot(args.results)

    block = render_block(current, previous, commit_sha=args.commit_sha)
    changed = update_readme(args.readme, block)
    changed |= update_readme(
        args.readme, render_domain_block(current), DOMAINS_START_MARKER, DOMAINS_END_MARKER
    )
    print("README updated." if changed else "README already up-to-date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
