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

Operator tool: emit a CycloneDX 1.5 SBOM for the installed environment.

Walks ``importlib.metadata`` for every installed distribution and
emits a CycloneDX 1.5 JSON document with name, version, PURL, and (when
available) license + SHA-256 hashes from the wheel record.  Output is
deterministic (sorted components) so the SBOM diff between two builds
is reviewable.

We deliberately *do not* shell out to ``cyclonedx-bom`` or any other
external CLI — the SBOM is constructed in-process from the Python
runtime's own metadata view, keeping it dependency-free.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.metadata as _md
import json
import os
import uuid
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

_SCHEMA = "mercury.tools.sbom_emitter/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.sbom_emitter",
        description="Emit a CycloneDX 1.5 SBOM for the active Python environment.",
    )
    parser.add_argument(
        "--sbom-path",
        default=None,
        help="Write the SBOM JSON to this path in addition to the certificate body.",
    )
    parser.add_argument(
        "--root-name",
        default="omni-mercury-engine",
        help="Distribution name to treat as the SBOM root (default: omni-mercury-engine).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the SBOM but do NOT write --sbom-path.",
    )
    parser.add_argument(
        "--serial-number",
        default=None,
        help=(
            "Pin the CycloneDX serialNumber (urn:uuid:...).  Defaults to a "
            "deterministic UUID derived from the component digests so two "
            "runs with identical inputs produce byte-identical SBOMs."
        ),
    )
    return parser


def _component_for(dist: _md.Distribution) -> dict[str, Any]:
    meta = dist.metadata
    name = (meta.get("Name") or dist.name or "").strip()
    version = (meta.get("Version") or dist.version or "").strip()
    license_id = meta.get("License") or ""
    purl = f"pkg:pypi/{name.lower()}@{version}"
    component: dict[str, Any] = {
        "type": "library",
        "bom-ref": purl,
        "name": name,
        "version": version,
        "purl": purl,
    }
    if license_id:
        component["licenses"] = [{"license": {"name": license_id}}]
    # Optional: per-file hashes via the dist record entries.  We hash the
    # WHEEL's recorded file digests if present (avoids re-hashing every
    # file on disk).
    try:
        record = dist.read_text("RECORD") or ""
        digests = []
        for line in record.splitlines():
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].startswith("sha256="):
                digests.append(parts[1].removeprefix("sha256="))
        if digests:
            combined = hashlib.sha256("\n".join(sorted(digests)).encode("utf-8")).hexdigest()
            component["hashes"] = [{"alg": "SHA-256", "content": combined}]
    except Exception:
        pass
    return component


def _build_sbom(root_name: str, serial_number: str | None = None) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dist in _md.distributions():
        try:
            name = (dist.metadata.get("Name") or "").lower()
        except Exception:
            continue
        if not name or name in seen:
            continue
        seen.add(name)
        components.append(_component_for(dist))
    components.sort(key=lambda c: c["name"].lower())

    root_purl = None
    for c in components:
        if c["name"].lower() == root_name.lower():
            root_purl = c["bom-ref"]
            break

    # Deterministic serialNumber: UUIDv5 over the sorted component
    # digests so two runs with the same environment produce byte-
    # identical SBOMs (modulo timestamp).  Operators can override via
    # ``--serial-number`` when re-using a previously published SBOM.
    if serial_number is None:
        agg = hashlib.sha256(
            "\n".join(c.get("bom-ref", c["name"]) for c in components).encode("utf-8")
        ).hexdigest()
        serial_number = f"urn:uuid:{uuid.UUID(agg[:32])}"

    # Deterministic timestamp under SOURCE_DATE_EPOCH (in seconds since
    # the Unix epoch) so a reproducible-build environment can pin the
    # SBOM's wall-clock too.
    sde = os.environ.get("SOURCE_DATE_EPOCH")
    if sde and sde.isdigit():
        ts_dt = _dt.datetime.fromtimestamp(int(sde), tz=_dt.UTC)
    else:
        ts_dt = _dt.datetime.now(_dt.UTC)
    timestamp = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": serial_number,
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [
                {
                    "vendor": "Steel Security Advisors",
                    "name": "mercury.sbom_emitter",
                    "version": "1",
                }
            ],
            "component": (
                {"type": "application", "bom-ref": root_purl, "name": root_name}
                if root_purl
                else {"type": "application", "name": root_name}
            ),
        },
        "components": components,
    }


def _collect(args: argparse.Namespace) -> Certificate:
    sbom = _build_sbom(args.root_name, args.serial_number)
    if args.sbom_path and not args.dry_run:
        atomic_write_text(Path(args.sbom_path), json.dumps(sbom, indent=2, sort_keys=True) + "\n")
    return Certificate(
        tool="sbom_emitter",
        schema=_SCHEMA,
        status="ok",
        body={
            "component_count": len(sbom["components"]),
            "root_name": args.root_name,
            "sbom_path": args.sbom_path,
            "dry_run": bool(args.dry_run),
            "sbom": sbom,
        },
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
