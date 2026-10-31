# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decoupling switches: fragment the three kin systems without estranging them.

Mercury Agent, AMA Cryptography and FINDΩYOU™ are separate applications that
are designed as kin. An operator must be able to pull them apart cleanly --
turn an integration off, detach Mercury while keeping every byte of work,
destroy that work, or remove Mercury and all of its code from the machine --
and each of those has to be a real action with a visible result, not a flag
that is read nowhere.

What this module controls, and what it refuses to pretend to control:

* ``findoyou`` -- the FINDΩYOU™ evidence bridge
  (``integrations/findoyou_bridge.py``). Off means importing the bridge
  raises :class:`MercuryDecoupledError` with the re-enable command, and the
  self-model reports the capability as disabled by the operator.
* ``ama_optional`` -- the AMA-backed surfaces Mercury can run *without*:
  the auth layer's HD key-derivation/rotation manager, and the bridge above.
  Off means those surfaces refuse with a named reason.
* The AMA **core** gate is not a switch. ``_pqc_gate`` makes AMA mandatory
  for the engine to import at all, and ``tests/test_pqc_startup_gate.py``
  pins that "unset/false compatibility env vars must not create an AMA-less
  mode". This module does not add one. The only total decoupling from AMA
  is :func:`uninstall`, and :func:`status` says so in plain words.

Switch state lives in ``decouple.json`` beside the existing
``~/.config/omni_mercury/config.yaml`` (``XDG_CONFIG_HOME`` honoured). An
environment variable ``MERCURY_DECOUPLE_<NAME>=0|1`` overrides the file for
one process, so a deployment can pin a switch without touching the file.

Destructive operations are dry-run unless given the exact confirmation
phrase, and they print what they will remove, with sizes, before they do.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CONFIRM_PURGE",
    "CONFIRM_UNINSTALL",
    "DISTRIBUTION",
    "SWITCHES",
    "InventoryItem",
    "MercuryDecoupledError",
    "config_path",
    "is_enabled",
    "purge_work",
    "read_switches",
    "require_enabled",
    "set_switch",
    "snapshot",
    "state_inventory",
    "status",
    "uninstall",
    "uninstall_plan",
]

DISTRIBUTION = "mercury-agent-global"
CONFIRM_PURGE = "PURGE WORK"
CONFIRM_UNINSTALL = "DELETE MERCURY AGENT"

#: Switch name -> (default, what turning it off does).
SWITCHES: dict[str, tuple[bool, str]] = {
    "findoyou": (
        True,
        "FINDΩYOU™ evidence bridge (integrations/findoyou_bridge.py) refuses to import; "
        "self-model reports the capability as disabled by the operator",
    ),
    "ama_optional": (
        True,
        "AMA-backed optional surfaces refuse: auth HD key-derivation/rotation "
        "(api/auth.py AuthKeyManager) and the FINDΩYOU™ bridge. The AMA core "
        "gate is unaffected and cannot be switched off while Mercury runs",
    ),
}

#: Browser-side state the console keeps. A CLI cannot clear another
#: program's localStorage; these are listed so the operator can.
CONSOLE_LOCAL_STORAGE_KEYS: tuple[str, ...] = (
    "mercury_cli_state_v1",
    "mercury_cli_capture_v1",
    "mercury_cli_conn_v1",
    "mercury_cli_mcp_order",
)


class MercuryDecoupledError(RuntimeError):
    """Raised when a surface is reached that the operator has switched off."""


# ---------------------------------------------------------------------------
# switches
# ---------------------------------------------------------------------------


def config_path() -> Path:
    """Where switch state is stored (beside the engine's own config.yaml)."""
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "omni_mercury" / "decouple.json"


def read_switches() -> dict[str, bool]:
    """Effective switch state: file, then ``MERCURY_DECOUPLE_<NAME>`` overrides."""
    state = {name: default for name, (default, _) in SWITCHES.items()}
    path = config_path()
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            for name in SWITCHES:
                if isinstance(stored.get(name), bool):
                    state[name] = stored[name]
        except (OSError, ValueError) as exc:
            logger.warning("decouple.json unreadable (%s); using defaults", exc)
    for name in SWITCHES:
        raw = os.environ.get(f"MERCURY_DECOUPLE_{name.upper()}", "").strip().lower()
        if raw in {"0", "false", "off", "no"}:
            state[name] = False
        elif raw in {"1", "true", "on", "yes"}:
            state[name] = True
    return state


def is_enabled(name: str) -> bool:
    """Whether ``name`` is on. Unknown names raise, so a typo cannot read as on."""
    if name not in SWITCHES:
        raise KeyError(f"unknown decoupling switch {name!r}; known: {sorted(SWITCHES)}")
    return read_switches()[name]


def set_switch(name: str, enabled: bool) -> Path:
    """Persist ``name`` = ``enabled``; returns the file written."""
    if name not in SWITCHES:
        raise KeyError(f"unknown decoupling switch {name!r}; known: {sorted(SWITCHES)}")
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    stored: dict[str, Any] = {}
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = {}
    stored[name] = bool(enabled)
    stored["_updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def require_enabled(name: str, surface: str) -> None:
    """Refuse ``surface`` when ``name`` is off, naming the switch and the remedy."""
    if not is_enabled(name):
        raise MercuryDecoupledError(
            f"{surface} is switched off by the operator (decoupling switch "
            f"{name!r}). Re-enable with: mercury-agent decouple {name.replace('_', '-')} on "
            f"-- or unset MERCURY_DECOUPLE_{name.upper()} if it is pinned in the environment."
        )


# ---------------------------------------------------------------------------
# state inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InventoryItem:
    """One place Mercury keeps work on this machine."""

    label: str
    path: Path
    source: str  # "default" or the env var that pointed here
    exists: bool
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total
    return 0


def state_inventory() -> list[InventoryItem]:
    """Every location Mercury writes work to, default and env-pointed.

    The defaults are the ones the code actually uses (``engine.py``,
    ``loaders/base.py``, ``self_model.py``, ``cognitive/gate_audit.py``,
    ``validation/data_loaders.py``, ``core/config.py``,
    ``research/engine_calibration/_workspace.py``), not a guess.
    """
    home = Path.home()
    xdg_state = os.environ.get("XDG_STATE_HOME", "").strip()
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    xdg_cfg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    items: list[tuple[str, Path, str]] = [
        ("models, cache, self-model, adjudication", home / ".mercury", "default"),
        (
            "engine config + decoupling switches",
            (Path(xdg_cfg) if xdg_cfg else home / ".config") / "omni_mercury",
            "default",
        ),
        (
            "ethical-gate audit (XDG state)",
            (Path(xdg_state) if xdg_state else home / ".local" / "state") / "mercury-agent",
            "default",
        ),
        ("validation dataset cache", home / ".omni_mercury", "default"),
        (
            "engine calibration cache",
            (Path(xdg_cache) if xdg_cache else home / ".cache") / "mercury-agent",
            "default",
        ),
    ]
    for var, label in (
        ("MERCURY_KEYSTORE_PATH", "account/API-key store"),
        ("MERCURY_AUDIT_LOG_DIR", "audit log"),
        ("MERCURY_SECURE_AUDIT_DIR", "tamper-evident audit"),
        ("MERCURY_DATA_DIR", "data directory"),
        ("MERCURY_MODEL_DIR", "model directory"),
        ("MERCURY_CACHE_DIR", "cache directory"),
        ("MERCURY_SELF_MODEL_PATH", "self-model"),
        ("MERCURY_ADJUDICATION_PATH", "adjudication ledger"),
        ("MERCURY_CALIBRATION_CACHE", "calibration cache"),
    ):
        raw = os.environ.get(var, "").strip()
        if raw:
            items.append((label, Path(raw).expanduser(), var))
    seen: set[Path] = set()
    out: list[InventoryItem] = []
    for label, path, source in items:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(InventoryItem(label, resolved, source, resolved.exists(), _size(resolved)))
    return out


# ---------------------------------------------------------------------------
# detach (save), purge, uninstall
# ---------------------------------------------------------------------------


def snapshot(dest_dir: Path) -> Path:
    """Save every existing state location into one archive with a manifest.

    Returns the archive path. The manifest records each file's SHA-256 so the
    archive can be verified later, and the switch state so a restore can
    reproduce it.
    """
    dest_dir = Path(dest_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = dest_dir / f"mercury-agent-state-{stamp}.tar.gz"
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "switches": read_switches(),
        "distribution": _distribution_info(),
        "items": [],
        "files": {},
    }
    with tarfile.open(archive, "w:gz") as tar:
        for item in state_inventory():
            if not item.exists:
                continue
            arcroot = item.path.name if item.path.is_dir() else item.path.name
            arcroot = f"{item.source}/{arcroot}" if item.source != "default" else arcroot
            tar.add(item.path, arcname=arcroot)
            manifest["items"].append({**item.to_dict(), "arcname": arcroot})
            paths = (
                [item.path]
                if item.path.is_file()
                else [p for p in item.path.rglob("*") if p.is_file()]
            )
            for p in paths:
                try:
                    manifest["files"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
                except OSError:
                    continue
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        import io

        tar.addfile(info, io.BytesIO(manifest_bytes))
    (dest_dir / (archive.stem.replace(".tar", "") + ".manifest.json")).write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    )
    return archive


def purge_work(confirm: str | None = None, *, dry_run: bool = True) -> list[InventoryItem]:
    """Delete every existing state location. Dry-run unless confirmed.

    ``confirm`` must equal :data:`CONFIRM_PURGE` and ``dry_run`` must be
    ``False`` for anything to be removed; otherwise the list of what *would*
    be removed is returned unchanged. Code is never touched here.
    """
    targets = [item for item in state_inventory() if item.exists]
    if dry_run or confirm != CONFIRM_PURGE:
        return targets
    for item in targets:
        if item.path.is_dir():
            shutil.rmtree(item.path, ignore_errors=False)
        else:
            item.path.unlink()
    return targets


def _distribution_info() -> dict[str, Any]:
    """Installed distribution facts, including an editable source checkout if any."""
    info: dict[str, Any] = {"name": DISTRIBUTION, "installed": False}
    try:
        dist = metadata.distribution(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return info
    info["installed"] = True
    info["version"] = dist.version
    info["location"] = str(dist.locate_file(""))
    direct = dist.read_text("direct_url.json")
    if direct:
        try:
            data = json.loads(direct)
            url = str(data.get("url", ""))
            if data.get("dir_info", {}).get("editable") and url.startswith("file://"):
                info["editable_source"] = url[len("file://") :]
        except ValueError:
            pass
    return info


def uninstall_plan(*, remove_source: bool = False) -> dict[str, Any]:
    """What :func:`uninstall` would do, as data."""
    dist = _distribution_info()
    plan: dict[str, Any] = {
        "distribution": dist,
        "pip_command": [sys.executable, "-m", "pip", "uninstall", "-y", DISTRIBUTION],
        "state_removed": [i.to_dict() for i in state_inventory() if i.exists],
        "source_removed": dist.get("editable_source") if remove_source else None,
        "console_local_storage_keys": list(CONSOLE_LOCAL_STORAGE_KEYS),
    }
    return plan


def uninstall(
    confirm: str | None = None,
    *,
    remove_source: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Remove Mercury Agent: its work, its package, and optionally its source.

    Dry-run unless ``confirm`` equals :data:`CONFIRM_UNINSTALL` and
    ``dry_run`` is ``False``. The source checkout is removed only when
    ``remove_source`` is also set, and only if the install is editable and
    points at a directory that is a git checkout -- never an arbitrary path.
    """
    plan = uninstall_plan(remove_source=remove_source)
    plan["executed"] = False
    if dry_run or confirm != CONFIRM_UNINSTALL:
        return plan
    purge_work(CONFIRM_PURGE, dry_run=False)
    # Fixed argv built above: the interpreter running this code, pip, and the
    # distribution-name constant. Nothing caller-supplied reaches the shell.
    result = subprocess.run(  # noqa: S603
        plan["pip_command"], capture_output=True, text=True, check=False
    )
    plan["pip_returncode"] = result.returncode
    plan["pip_output"] = (result.stdout + result.stderr)[-2000:]
    src = plan["source_removed"]
    if src:
        src_path = Path(src)
        if (src_path / ".git").is_dir() and (src_path / "pyproject.toml").is_file():
            shutil.rmtree(src_path)
        else:
            plan["source_removed"] = None
            plan["source_note"] = f"{src} is not a git checkout with a pyproject; left in place"
    plan["executed"] = True
    return plan


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _ama_facts() -> dict[str, Any]:
    """What AMA actually is on this machine, read from the same flags the gate reads."""
    facts: dict[str, Any] = {
        "core_gate": "mandatory -- AMA/PQC is required for the engine to import; "
        "not a switch (pinned by tests/test_pqc_startup_gate.py). Total decoupling "
        "from AMA is `mercury-agent decouple uninstall`.",
        "required_version": "5.0.0",
    }
    try:
        from ama_cryptography import __version__ as ama_version

        facts["installed_version"] = str(ama_version)
    except Exception:  # pragma: no cover - depends on the AMA build present
        facts["installed_version"] = None
    try:
        import ama_cryptography.pqc_backends as backends

        facts["native_primitives"] = {
            "ML-DSA-65 (Dilithium)": bool(getattr(backends, "DILITHIUM_AVAILABLE", False)),
            "Kyber-1024": bool(getattr(backends, "KYBER_AVAILABLE", False)),
            "SPHINCS+": bool(getattr(backends, "SPHINCS_AVAILABLE", False)),
            "Ed25519 (native)": bool(getattr(backends, "_ED25519_NATIVE_AVAILABLE", False)),
        }
        facts["per_primitive_switches"] = (
            "none -- availability is reported by the AMA build; Mercury does not "
            "select primitives individually"
        )
    except Exception as exc:  # pragma: no cover - depends on the AMA build present
        facts["native_primitives"] = None
        facts["import_error"] = str(exc)
    return facts


def status() -> dict[str, Any]:
    """Everything an operator needs to see before flipping anything."""
    switches = read_switches()
    return {
        "switches": {
            name: {
                "enabled": switches[name],
                "default": SWITCHES[name][0],
                "off_means": SWITCHES[name][1],
                "env_override": os.environ.get(f"MERCURY_DECOUPLE_{name.upper()}") or None,
            }
            for name in SWITCHES
        },
        "config_file": str(config_path()),
        "config_file_exists": config_path().is_file(),
        "ama": _ama_facts(),
        "state": [i.to_dict() for i in state_inventory()],
        "distribution": _distribution_info(),
        "console_local_storage_keys": list(CONSOLE_LOCAL_STORAGE_KEYS),
    }
