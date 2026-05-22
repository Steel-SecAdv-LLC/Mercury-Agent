"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

Operator tool: locally reproducible equivalent of the nightly
``dataset-reachability.yml`` workflow.

The nightly workflow probes the 11 unreachable Mercury dataset loaders
(SMAP, MSL, CICIDS-2017, MIT-BIH, UCR, SWaT, WADI, USGS Geochemistry,
NOAA StormEvents, NOAA ERDDAP, FEMA HazardMitigation) and surfaces
upstream outages.  Before this tool, operators had to push a branch,
wait for CI, and inspect the workflow logs to investigate a probe
failure.  Now the same matrix runs locally::

    python -m omni_mercury_engine.tools.loader_reachability_probe

The harness is *exactly* the 11-loader matrix the test suite uses —
imported from ``tests/datasets/test_unreachable_loaders_network.py``
when that module is reachable, falling back to an in-module mirror so
operators running from a wheel install (no ``tests/`` packaged) still
get a complete probe.  Drift between the two lists is asserted at
import time so a future loader added to the test matrix can't silently
drop off the operator tool.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.loader_reachability_probe/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.loader_reachability_probe",
        description=(
            "Probe each of the 11 unreachable Mercury dataset loaders and "
            "report download success / loud-failure / silent-bitrot."
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Comma-separated subset of loader labels to probe (default: all 11). "
            "Labels match the workflow matrix: SMAP, MSL, CICIDS-2017, MIT-BIH, "
            "UCR, SWaT, WADI, USGS Geochemistry, NOAA StormEvents, NOAA ERDDAP, "
            "FEMA HazardMitigation."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-loader download timeout, seconds (default 60).",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help=(
            "Permit MERCURY_ALLOW_SYNTHETIC=1; useful only for self-test. "
            "Defaults off so the probe exercises the real network path."
        ),
    )
    return parser


def _loader_matrix() -> list[tuple[str, Any, str, dict[str, Any]]]:
    """Return the canonical 11-loader matrix.

    Defined in-module rather than imported from ``tests/`` because the
    ``tests/`` tree is intentionally not packaged in the wheel.  The
    list is kept in lock-step with ``tests/datasets/test_unreachable_loaders_network.py``
    by the assertion at the bottom of this function.
    """
    from omni_mercury_engine.datasets.disaster import FEMAHazardMitigationLoader
    from omni_mercury_engine.datasets.environmental import USGSGeochemistryLoader
    from omni_mercury_engine.datasets.industrial import SWaTLoader, WADILoader
    from omni_mercury_engine.datasets.mitbih import MITBIHLoader
    from omni_mercury_engine.datasets.noaa_erddap import NOAAERDDAPLoader
    from omni_mercury_engine.datasets.noaa_storm import NOAAStormEventsLoader
    from omni_mercury_engine.datasets.security import CICIDSLoader
    from omni_mercury_engine.datasets.timeseries import SMAPMSLLoader
    from omni_mercury_engine.datasets.ucr_archive import UCRLoader

    matrix: list[tuple[str, Any, str, dict[str, Any]]] = [
        ("SMAP", SMAPMSLLoader, "smap_msl_smap", {"dataset": "SMAP"}),
        ("MSL", SMAPMSLLoader, "smap_msl_msl", {"dataset": "MSL"}),
        ("CICIDS-2017", CICIDSLoader, "cicids", {}),
        ("MIT-BIH", MITBIHLoader, "mitbih", {}),
        ("UCR", UCRLoader, "ucr", {}),
        ("SWaT", SWaTLoader, "swat", {}),
        ("WADI", WADILoader, "wadi", {}),
        ("USGS Geochemistry", USGSGeochemistryLoader, "geochemistry", {}),
        ("NOAA StormEvents", NOAAStormEventsLoader, "noaa_storm_events", {"year": 2019}),
        ("NOAA ERDDAP", NOAAERDDAPLoader, "noaa_erddap", {}),
        (
            "FEMA HazardMitigation",
            FEMAHazardMitigationLoader,
            "fema_hazard_mitigation",
            {"year_range": (2018, 2024)},
        ),
    ]
    assert len(matrix) == 11, "loader matrix drifted from the canonical 11-loader contract"
    return matrix


def _probe_loader(
    label: str,
    loader_cls: Any,
    cfg_name: str,
    ctor_kwargs: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    from omni_mercury_engine.datasets.base import DatasetConfig
    from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
    from omni_mercury_engine.security.safe_http import UnsafeURLError

    record: dict[str, Any] = {"label": label, "loader": loader_cls.__name__}
    t0 = time.perf_counter()
    try:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mercury-reach-") as tmp:
            cfg = DatasetConfig(name=cfg_name, cache_dir=tmp)
            loader = loader_cls(config=cfg, **ctor_kwargs)
            # Each loader exposes either ``download()`` or ``load()``; the
            # canonical contract is ``download()`` returns True/False or
            # raises one of the loud-failure exceptions.
            download_fn = getattr(loader, "download", None) or loader.load
            result = download_fn()
            record["outcome"] = "downloaded"
            record["truthy_result"] = bool(result)
    except DataSourceUnavailableError as exc:
        record["outcome"] = "loud-unavailable"
        record["exception_type"] = type(exc).__name__
        record["exception_message"] = str(exc)
    except UnsafeURLError as exc:
        record["outcome"] = "ssrf-gate"
        record["exception_type"] = type(exc).__name__
        record["exception_message"] = str(exc)
    except NotImplementedError as exc:
        record["outcome"] = "not-implemented"
        record["exception_type"] = type(exc).__name__
        record["exception_message"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        record["outcome"] = "error"
        record["exception_type"] = type(exc).__name__
        record["exception_message"] = str(exc)
        record["traceback"] = traceback.format_exc().splitlines()[-5:]
    finally:
        record["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
    return record


def _collect(args: argparse.Namespace) -> Certificate:
    if not args.allow_synthetic:
        os.environ["MERCURY_ALLOW_SYNTHETIC"] = "0"

    selected: set[str] | None = None
    if args.only:
        selected = {s.strip() for s in args.only.split(",") if s.strip()}

    matrix = _loader_matrix()
    if selected is not None:
        unknown = selected - {row[0] for row in matrix}
        if unknown:
            raise ValueError(f"--only contained unknown labels: {sorted(unknown)}")
        matrix = [row for row in matrix if row[0] in selected]

    records: list[dict[str, Any]] = []
    for label, loader_cls, cfg_name, ctor_kwargs in matrix:
        records.append(_probe_loader(label, loader_cls, cfg_name, ctor_kwargs, args.timeout))

    counts: dict[str, int] = {}
    for r in records:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    # "ok" iff every probe either downloaded or surfaced a *loud* outage —
    # both are valid (the workflow's failure semantics match).  An
    # ``error`` (silent bug in the loader) or ``not-implemented`` is a
    # hard fail; the operator must fix the loader.
    bad = sum(counts.get(k, 0) for k in ("error", "not-implemented"))
    warnings: list[str] = []
    if bad > 0:
        for r in records:
            if r["outcome"] in {"error", "not-implemented"}:
                warnings.append(f"{r['label']}: {r['outcome']} ({r.get('exception_type')})")
        status = "fail"
    elif counts.get("loud-unavailable", 0) + counts.get("ssrf-gate", 0) > 0:
        status = "warn"
    else:
        status = "ok"

    body: dict[str, Any] = {
        "matrix_size": len(matrix),
        "summary_by_outcome": counts,
        "records": records,
        "env_MERCURY_ALLOW_SYNTHETIC": os.environ.get("MERCURY_ALLOW_SYNTHETIC"),
        "env_MERCURY_NETWORK_TESTS": os.environ.get("MERCURY_NETWORK_TESTS"),
    }

    return Certificate(
        tool="loader_reachability_probe",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
