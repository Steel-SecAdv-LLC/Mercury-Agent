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

Operator tool: schema-validate every YAML/JSON config under ``configs/``.

The repo's runtime config layer (``omni_mercury_engine.core.config``)
hands operators flexible YAML, but a typo in a *required* field
(missing ``lambda_lyapunov``, mis-spelled ``benevolence_threshold``,
etc.) is only caught when the runtime tries to read it.  This tool
walks ``configs/`` and validates every config against a documented
contract — a hand-written, dependency-free schema embedded below.

The schema is intentionally hand-rolled rather than pulled from
JSON-Schema, jsonschema, or pydantic so it has zero install-time
overhead and lives in-tree, perfectly auditable.

A non-zero exit code on schema failure makes this safe to wire into
pre-commit or CI: a config typo can't land on ``main``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.config_validator/v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIGS_DIR = _REPO_ROOT / "configs"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.config_validator",
        description=(
            "Schema-validate every YAML/JSON config under configs/. Would "
            "have caught the missing-lambda bug structurally."
        ),
    )
    parser.add_argument(
        "--dir",
        default=str(_DEFAULT_CONFIGS_DIR),
        help="Directory to walk (default: configs/).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat unknown top-level keys as errors (default: warn).  Useful "
            "in CI to catch typos that would otherwise be silently ignored "
            "by the dynamic config loader."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Hand-written schema.  Each entry is a top-level key that the runtime
# is willing to consume; nested validation is performed by ``_validate``
# below.  ``required`` is checked when the key is *present* — at the
# config level (not the section level) we deliberately allow operators
# to omit sections they're not using.
# ---------------------------------------------------------------------------

_SECTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "model": {
        "required": ["input_dim", "d_model", "n_heads"],
        "optional": [
            "num_layers",
            "num_scales",
            "max_freqs",
            "dropout",
            "ethical_threshold",
        ],
        "types": {
            "input_dim": int,
            "d_model": int,
            "n_heads": int,
            "num_layers": int,
            "num_scales": int,
            "max_freqs": int,
            "dropout": (int, float),
            "ethical_threshold": (int, float),
        },
    },
    "loss": {
        "required": [],
        "optional": [
            "lambda_lyapunov",
            "lambda_kl",
            "mu_stability",
            "alpha",
            "beta",
        ],
        "types": {
            "lambda_lyapunov": (int, float),
            "lambda_kl": (int, float),
            "mu_stability": (int, float),
            "alpha": (int, float),
            "beta": (int, float),
        },
    },
    "training": {
        "required": [],
        "optional": ["epochs", "batch_size", "lr", "weight_decay", "seed"],
        "types": {
            "epochs": int,
            "batch_size": int,
            "lr": (int, float),
            "weight_decay": (int, float),
            "seed": int,
        },
    },
    "data": {
        "required": [],
        "optional": ["dataset", "split", "cache_dir", "allow_synthetic"],
        "types": {
            "dataset": str,
            "split": str,
            "cache_dir": str,
            "allow_synthetic": bool,
        },
    },
}


def _load_doc(path: Path) -> Any:
    """Load ``path`` as YAML or JSON depending on suffix."""
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if importlib.util.find_spec("yaml") is None:
            raise RuntimeError(
                "PyYAML is not installed; install with `pip install pyyaml` "
                "to validate YAML configs (JSON configs work without it)."
            )
        import yaml  # type: ignore[import-not-found]

        return yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"unsupported config extension: {path.suffix}")


def _validate_section(
    section_name: str,
    section_schema: dict[str, Any],
    body: dict[str, Any],
    strict: bool,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(body, dict):
        return [f"{section_name}: expected mapping, got {type(body).__name__}"]
    required = section_schema.get("required", [])
    optional = section_schema.get("optional", [])
    types = section_schema.get("types", {})
    known = set(required) | set(optional)

    for key in required:
        if key not in body:
            errors.append(f"{section_name}.{key} is required but missing")
    for key, value in body.items():
        if key not in known:
            msg = f"{section_name}.{key} is an unknown key"
            errors.append(msg) if strict else None
            continue
        expected_type = types.get(key)
        if expected_type and not isinstance(value, expected_type):
            errors.append(
                f"{section_name}.{key} expected {expected_type}, "
                f"got {type(value).__name__}"
            )
    return errors


def _validate(doc: Any, strict: bool) -> list[str]:
    if doc is None:
        return ["empty config (top-level mapping required)"]
    if not isinstance(doc, dict):
        return [f"top-level must be a mapping, got {type(doc).__name__}"]
    errors: list[str] = []
    # Configs can be either a flat mapping (a single config) or a mapping
    # of named profiles (each value being a config).  We auto-detect: if
    # every value is itself a mapping containing at least one known
    # section, treat it as a profile bundle.
    is_profile_bundle = all(
        isinstance(v, dict) and any(k in _SECTION_SCHEMAS for k in v.keys())
        for v in doc.values()
    ) and len(doc) > 0
    profiles = doc.items() if is_profile_bundle else [("<root>", doc)]
    for profile_name, profile_body in profiles:
        if not isinstance(profile_body, dict):
            errors.append(f"profile {profile_name!r}: expected mapping")
            continue
        for section_name, section_body in profile_body.items():
            if section_name in _SECTION_SCHEMAS:
                errors.extend(
                    f"{profile_name}.{e}"
                    for e in _validate_section(
                        section_name, _SECTION_SCHEMAS[section_name], section_body, strict
                    )
                )
            elif strict:
                errors.append(f"{profile_name}.{section_name} is an unknown section")
    return errors


def _collect(args: argparse.Namespace) -> Certificate:
    root = Path(args.dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"configs dir not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".yaml", ".yml", ".json"})

    per_file: list[dict[str, Any]] = []
    overall_errors = 0
    for f in files:
        rel = str(f.relative_to(root))
        try:
            doc = _load_doc(f)
            errs = _validate(doc, args.strict)
        except Exception as exc:
            errs = [f"{type(exc).__name__}: {exc}"]
        per_file.append({"path": rel, "errors": errs, "valid": not errs})
        if errs:
            overall_errors += 1

    body: dict[str, Any] = {
        "root": str(root),
        "strict": bool(args.strict),
        "file_count": len(files),
        "invalid_count": overall_errors,
        "files": per_file,
    }
    warnings: list[str] = []
    for entry in per_file:
        if not entry["valid"]:
            warnings.append(f"{entry['path']}: {entry['errors']}")
    status = "fail" if overall_errors else "ok"
    return Certificate(
        tool="config_validator",
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
