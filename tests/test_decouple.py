# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The decoupling switches are real: read where they are written, refused where they say.

Each switch has three facts to pin -- it persists, the environment can pin it
for one process, and the surface it names actually refuses when it is off --
plus the two destructive operations must do nothing without their exact
confirmation phrase and must do what they list when given it.

Everything here runs against a temporary HOME/XDG tree, so no test can read or
touch the operator's real state.
"""

from __future__ import annotations

import importlib
import json
import sys
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from omni_mercury_engine import decouple as dc
from omni_mercury_engine.cli import main


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every state location Mercury knows at a scratch tree."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / ".local" / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    for var in (
        "MERCURY_KEYSTORE_PATH",
        "MERCURY_AUDIT_LOG_DIR",
        "MERCURY_SECURE_AUDIT_DIR",
        "MERCURY_DATA_DIR",
        "MERCURY_MODEL_DIR",
        "MERCURY_CACHE_DIR",
        "MERCURY_SELF_MODEL_PATH",
        "MERCURY_ADJUDICATION_PATH",
        "MERCURY_CALIBRATION_CACHE",
        "MERCURY_DECOUPLE_FINDOYOU",
        "MERCURY_DECOUPLE_AMA_OPTIONAL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


class TestSwitches:
    def test_defaults_are_on_and_config_file_is_beside_engine_config(
        self, isolated_home: Path
    ) -> None:
        assert dc.read_switches() == {"findoyou": True, "ama_optional": True}
        assert dc.config_path() == isolated_home / ".config" / "omni_mercury" / "decouple.json"
        assert not dc.config_path().exists()

    def test_set_persists_and_is_read_back(self, isolated_home: Path) -> None:
        path = dc.set_switch("findoyou", False)
        assert path.is_file()
        assert json.loads(path.read_text())["findoyou"] is False
        assert dc.is_enabled("findoyou") is False
        assert dc.is_enabled("ama_optional") is True
        dc.set_switch("findoyou", True)
        assert dc.is_enabled("findoyou") is True

    def test_environment_pins_a_switch_for_one_process(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dc.set_switch("ama_optional", True)
        monkeypatch.setenv("MERCURY_DECOUPLE_AMA_OPTIONAL", "0")
        assert dc.is_enabled("ama_optional") is False
        monkeypatch.setenv("MERCURY_DECOUPLE_AMA_OPTIONAL", "on")
        assert dc.is_enabled("ama_optional") is True

    def test_unknown_switch_cannot_read_as_on(self) -> None:
        with pytest.raises(KeyError):
            dc.is_enabled("mercury_core")

    def test_require_enabled_names_switch_and_remedy(self, isolated_home: Path) -> None:
        dc.set_switch("findoyou", False)
        with pytest.raises(dc.MercuryDecoupledError) as excinfo:
            dc.require_enabled("findoyou", "the FINDΩYOU™ evidence bridge")
        text = str(excinfo.value)
        assert "FINDΩYOU" in text
        assert "mercury-agent decouple findoyou on" in text
        assert "MERCURY_DECOUPLE_FINDOYOU" in text


class TestSurfacesRefuse:
    def test_bridge_refuses_to_import_when_findoyou_is_off(self, isolated_home: Path) -> None:
        dc.set_switch("findoyou", False)
        sys.modules.pop("omni_mercury_engine.integrations.findoyou_bridge", None)
        with pytest.raises(dc.MercuryDecoupledError):
            importlib.import_module("omni_mercury_engine.integrations.findoyou_bridge")
        sys.modules.pop("omni_mercury_engine.integrations.findoyou_bridge", None)

    def test_self_model_reports_operator_disabled(self, isolated_home: Path) -> None:
        from omni_mercury_engine.self_model import CAPABILITIES, _probe_capability

        spec = next(c for c in CAPABILITIES if c.name == "findoyou_evidence_bridge")
        assert spec.switch == "findoyou"
        dc.set_switch("findoyou", False)
        probe = _probe_capability(spec)
        assert probe["state"] == "unavailable"
        assert "disabled by the operator" in probe["reason"]
        assert "decouple findoyou on" in probe["reason"]

    def test_auth_key_manager_refuses_when_ama_optional_is_off(self, isolated_home: Path) -> None:
        from omni_mercury_engine.api.auth import AuthKeyManager

        dc.set_switch("ama_optional", False)
        with pytest.raises(dc.MercuryDecoupledError):
            AuthKeyManager()


class TestInventoryAndDetach:
    def test_inventory_lists_defaults_and_env_pointed_paths(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keystore = isolated_home / "elsewhere" / "keys.db"
        monkeypatch.setenv("MERCURY_KEYSTORE_PATH", str(keystore))
        items = {i.path: i for i in dc.state_inventory()}
        assert isolated_home / ".mercury" in items
        assert items[keystore].source == "MERCURY_KEYSTORE_PATH"
        assert items[keystore].exists is False

    def test_detach_archives_existing_state_with_manifest(self, isolated_home: Path) -> None:
        work = isolated_home / ".mercury" / "models"
        work.mkdir(parents=True)
        (work / "weights.bin").write_bytes(b"\x00" * 64)
        dc.set_switch("findoyou", False)

        archive = dc.snapshot(isolated_home / "out")
        assert archive.is_file()
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            manifest = json.loads(tar.extractfile("MANIFEST.json").read())  # type: ignore[union-attr]
        assert any(n.endswith("models/weights.bin") for n in names)
        assert manifest["switches"]["findoyou"] is False
        assert any(k.endswith("weights.bin") for k in manifest["files"])
        # detach saves; it never removes
        assert (work / "weights.bin").exists()


class TestDestructiveOpsAreGated:
    def test_purge_is_dry_run_without_the_phrase(self, isolated_home: Path) -> None:
        marker = isolated_home / ".mercury" / "cache" / "x.npz"
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b"1")
        targets = dc.purge_work(None)
        assert any(t.path == isolated_home / ".mercury" for t in targets)
        assert marker.exists()
        targets = dc.purge_work("purge work", dry_run=False)  # wrong case
        assert marker.exists()
        dc.purge_work(dc.CONFIRM_PURGE, dry_run=False)
        assert not marker.exists()

    def test_uninstall_is_a_plan_without_the_phrase(self, isolated_home: Path) -> None:
        plan = dc.uninstall(None, remove_source=True)
        assert plan["executed"] is False
        assert plan["pip_command"][-1] == dc.DISTRIBUTION
        assert plan["distribution"]["name"] == dc.DISTRIBUTION


class TestCli:
    def test_status_runs_and_names_the_core_gate_honestly(self, isolated_home: Path) -> None:
        result = CliRunner().invoke(main, ["decouple", "status"])
        assert result.exit_code == 0, result.output
        assert "mandatory" in result.output
        assert "findoyou" in result.output and "ama_optional" in result.output

    def test_switch_round_trip_via_cli(self, isolated_home: Path) -> None:
        runner = CliRunner()
        off = runner.invoke(main, ["decouple", "findoyou", "off"])
        assert off.exit_code == 0, off.output
        assert dc.is_enabled("findoyou") is False
        st = runner.invoke(main, ["decouple", "findoyou", "status", "--json"])
        assert json.loads(st.output)["enabled"] is False
        on = runner.invoke(main, ["decouple", "ama-optional", "on"])
        assert on.exit_code == 0

    def test_purge_work_cli_is_dry_run_by_default(self, isolated_home: Path) -> None:
        marker = isolated_home / ".mercury" / "self_model.jsonl"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}\n")
        result = CliRunner().invoke(main, ["decouple", "purge-work"])
        assert result.exit_code == 0, result.output
        assert "would remove" in result.output and "dry run" in result.output
        assert marker.exists()

    def test_uninstall_cli_is_dry_run_by_default(self, isolated_home: Path) -> None:
        result = CliRunner().invoke(main, ["decouple", "uninstall", "--remove-source"])
        assert result.exit_code == 0, result.output
        assert "would do" in result.output and "dry run" in result.output
