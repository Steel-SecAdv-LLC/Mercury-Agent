"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""Regenerate the ``<!-- BENCHMARK:START -->`` … ``<!-- BENCHMARK:END -->``
block in ``README.md`` from the current and previous
``benchmarks/mercury_benchmark_results.json``.

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

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

START_MARKER = "<!-- BENCHMARK:START -->"
END_MARKER = "<!-- BENCHMARK:END -->"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
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
        return json.loads(completed.stdout.decode("utf-8"))
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
        "total": summary.get("total") or summary.get("n_datasets") or summary.get("successful"),
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

Regression gates: ROC-AUC must stay ≥ 0.68 and Mean Oracle F1 ≥ 0.50 (set 15% below the 2026-02-15 measured baseline of AUC 0.803 / F1 0.589). CI fails the workflow if either drops below threshold.
{END_MARKER}"""


def update_readme(readme_path: Path, new_block: str) -> bool:
    """Replace the marker block in *readme_path*.  Returns ``True`` if changed."""
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        msg = (
            f"could not find '{START_MARKER}' … '{END_MARKER}' markers in "
            f"{readme_path}; aborting"
        )
        raise SystemExit(msg)
    new_text = pattern.sub(new_block, text)
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
    print("README updated." if changed else "README already up-to-date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
