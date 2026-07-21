# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""In-process behavioral tests for ``tools/migrate_pkl.py``.

The end-to-end operator workflow (real hardened subprocess, scrubbed env,
black-box exit codes) is exercised by ``tests/security/test_migrate_pkl.py``.
This module complements it with in-process tests of the same contract so the
individual branches are directly attributable and coverage-measurable:

Covers:
- ``_build_parser``: required arguments, ``--max-bytes`` default
- ``_banner``: operator-facing hardening statement
- ``_ALLOWED_GLOBALS``: numpy reconstruction surface present, RCE-adjacent
  globals (``os.system`` / ``posix.system`` / ``subprocess.Popen`` /
  ``builtins.eval`` / ``exec`` / ``__import__`` / ``getattr`` /
  ``codecs.encode``) absent
- ``_make_restricted_unpickler``: ``find_class`` allow / refuse behaviour,
  refusal exception is a distinguishable ``pickle.UnpicklingError`` subclass
- ``_do_migration`` exit codes: 0 (happy path, signed and unsigned),
  2 (missing input / oversize / would-overwrite), 3 (corrupt pickle,
  non-dict root, missing keys, non-string feature key, object dtype,
  reserved ``allow_pickle`` key), 4 (bad or short signing key),
  5 (restricted-unpickler refusal of malicious reduce payloads --
  the payloads must NOT execute)
- ``main``: hardened-sentinel dispatch, relaunch argv forwarding, and the
  scrubbed child environment (sentinel + ``PYTHONNOUSERSITE`` set,
  ``LD_PRELOAD`` / ``PYTHONPATH`` / ``PYTHONSTARTUP`` dropped)
"""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest

from omni_mercury_engine.security.safe_load import verify_npz_signature
from omni_mercury_engine.tools.migrate_pkl import (
    _ALLOWED_GLOBALS,
    _EXIT_RESTRICTED_UNPICKLER_REFUSAL,
    _HARDENED_SENTINEL,
    _banner,
    _build_parser,
    _do_migration,
    _make_restricted_unpickler,
    main,
)

if TYPE_CHECKING:
    import argparse
    import io
    from collections.abc import Mapping, Sequence
    from pathlib import Path

# ``omni_mercury_engine.security``'s ``__init__`` re-exports the *function*
# ``safe_exec``, shadowing the submodule of the same name; fetch the module
# object explicitly so ``python_module`` can be monkeypatched on it.
safe_exec = importlib.import_module("omni_mercury_engine.security.safe_exec")

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_SEED = 42


def _write_pickle(path: Path, payload: object) -> Path:
    """Serialise *payload* to *path* with stdlib pickle (fixture creation only)."""
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path


def _legacy_payload() -> dict[str, Any]:
    """A legitimate legacy training payload in the documented schema."""
    rng = np.random.default_rng(_SEED)
    return {
        "features": {
            "modality_a": rng.random((8, 16)).astype(np.float32),
            "modality_b": rng.random((8, 4)).astype(np.float32),
        },
        "labels": np.array([0, 1] * 4, dtype=np.int64),
    }


def _args(input_path: Path, output_path: Path, *extra: str) -> argparse.Namespace:
    """Parse CLI arguments exactly as the tool's entry point would."""
    return _build_parser().parse_args(
        ["--input", str(input_path), "--output", str(output_path), *extra]
    )


@pytest.fixture
def legacy_pkl(tmp_path: Path) -> Path:
    return _write_pickle(tmp_path / "legacy.pkl", _legacy_payload())


# ---------------------------------------------------------------------------
# Parser and banner
# ---------------------------------------------------------------------------


class TestParser:
    def test_input_and_output_are_required(self) -> None:
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

    def test_defaults(self, tmp_path: Path) -> None:
        args = _args(tmp_path / "in.pkl", tmp_path / "out.npz")
        assert args.sign_key_hex is None
        assert args.max_bytes == 256 * 1024 * 1024

    def test_max_bytes_is_parsed_as_int(self, tmp_path: Path) -> None:
        args = _args(tmp_path / "in.pkl", tmp_path / "out.npz", "--max-bytes", "1024")
        assert args.max_bytes == 1024

    def test_banner_states_the_hardening_contract(self) -> None:
        banner = _banner()
        assert "hardened subprocess" in banner
        assert "never" in banner


# ---------------------------------------------------------------------------
# Restricted unpickler allow-list
# ---------------------------------------------------------------------------


class TestAllowedGlobals:
    def test_numpy_reconstruction_surface_is_allowed(self) -> None:
        for entry in [
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
        ]:
            assert entry in _ALLOWED_GLOBALS

    def test_rce_adjacent_globals_are_not_allowed(self) -> None:
        for entry in [
            ("os", "system"),
            ("posix", "system"),
            ("nt", "system"),
            ("subprocess", "Popen"),
            ("builtins", "eval"),
            ("builtins", "exec"),
            ("builtins", "__import__"),
            ("builtins", "getattr"),
            ("builtins", "compile"),
            ("builtins", "open"),
            ("codecs", "encode"),
        ]:
            assert entry not in _ALLOWED_GLOBALS


class TestRestrictedUnpickler:
    def _unpickler(self, stream: io.BytesIO) -> tuple[Any, type[Exception]]:
        unpickler_cls, refusal_exc = _make_restricted_unpickler()
        return unpickler_cls(stream), refusal_exc

    def test_refusal_exception_is_unpickling_error_subclass(self) -> None:
        _, refusal_exc = _make_restricted_unpickler()
        assert issubclass(refusal_exc, pickle.UnpicklingError)
        assert refusal_exc is not pickle.UnpicklingError

    def test_find_class_resolves_allowed_globals(self) -> None:
        import io

        unpickler, _ = self._unpickler(io.BytesIO(b""))
        assert unpickler.find_class("numpy", "ndarray") is np.ndarray
        assert unpickler.find_class("builtins", "dict") is dict

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("os", "system"),
            ("posix", "system"),
            ("subprocess", "Popen"),
            ("builtins", "eval"),
            ("builtins", "__import__"),
            ("numpy", "load"),
        ],
    )
    def test_find_class_refuses_out_of_list_globals(self, module: str, name: str) -> None:
        import io

        unpickler, refusal_exc = self._unpickler(io.BytesIO(b""))
        with pytest.raises(refusal_exc, match="refusing global"):
            unpickler.find_class(module, name)


# ---------------------------------------------------------------------------
# _do_migration: happy paths
# ---------------------------------------------------------------------------


class TestMigrationHappyPath:
    def test_round_trip_produces_pickle_free_npz(
        self, tmp_path: Path, legacy_pkl: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_path = tmp_path / "out.npz"
        rc = _do_migration(_args(legacy_pkl, out_path))
        assert rc == 0
        assert out_path.exists()
        assert f"wrote: {out_path}" in capsys.readouterr().err
        # The archive must load WITHOUT pickle support -- that is the whole
        # point of the migration.
        with np.load(out_path, allow_pickle=False) as archive:
            expected = _legacy_payload()
            assert set(archive.files) == {"modality_a", "modality_b", "labels"}
            np.testing.assert_array_equal(archive["labels"], expected["labels"])
            np.testing.assert_array_equal(archive["modality_a"], expected["features"]["modality_a"])

    def test_signing_writes_verifiable_sidecar(self, tmp_path: Path, legacy_pkl: Path) -> None:
        out_path = tmp_path / "signed.npz"
        key_hex = "11" * 32
        rc = _do_migration(_args(legacy_pkl, out_path, "--sign-key-hex", key_hex))
        assert rc == 0
        sig_path = tmp_path / "signed.npz.sig"
        assert sig_path.exists()
        # 64 hex chars, and the signature must verify with the same key.
        assert len(bytes.fromhex(sig_path.read_text().strip())) == 32
        verify_npz_signature(out_path, bytes.fromhex(key_hex))


# ---------------------------------------------------------------------------
# _do_migration: input-file errors (exit code 2)
# ---------------------------------------------------------------------------


class TestInputFileErrors:
    def test_missing_input_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = _do_migration(_args(tmp_path / "nope.pkl", tmp_path / "out.npz"))
        assert rc == 2
        assert "input file not found" in capsys.readouterr().err

    def test_oversize_input_is_rejected_before_unpickling(
        self, tmp_path: Path, legacy_pkl: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = _do_migration(_args(legacy_pkl, tmp_path / "out.npz", "--max-bytes", "16"))
        assert rc == 2
        err = capsys.readouterr().err
        assert "exceeds --max-bytes 16" in err
        assert not (tmp_path / "out.npz").exists()

    def test_refuses_to_overwrite_existing_output(
        self, tmp_path: Path, legacy_pkl: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        existing = tmp_path / "out.npz"
        existing.write_bytes(b"placeholder")
        rc = _do_migration(_args(legacy_pkl, existing))
        assert rc == 2
        assert "refusing to overwrite" in capsys.readouterr().err
        assert existing.read_bytes() == b"placeholder"


# ---------------------------------------------------------------------------
# _do_migration: payload shape / dtype errors (exit code 3)
# ---------------------------------------------------------------------------


class TestPayloadSchemaErrors:
    def test_corrupt_pickle_maps_to_input_error_not_refusal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "corrupt.pkl"
        bad.write_bytes(b"\x80\x04this is not a valid pickle stream")
        rc = _do_migration(_args(bad, tmp_path / "out.npz"))
        # Corrupt input is exit code 3, NOT the refusal code 5: find_class
        # was never reached, so this must not masquerade as a policy refusal.
        assert rc == 3
        assert "malformed" in capsys.readouterr().err

    def test_non_dict_root(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        bad = _write_pickle(tmp_path / "bad.pkl", [1, 2, 3])
        rc = _do_migration(_args(bad, tmp_path / "out.npz"))
        assert rc == 3
        assert "must be a dict" in capsys.readouterr().err

    def test_missing_labels(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        bad = _write_pickle(tmp_path / "bad.pkl", {"features": {}})
        rc = _do_migration(_args(bad, tmp_path / "out.npz"))
        assert rc == 3
        assert "'features': {str: ndarray}" in capsys.readouterr().err

    def test_features_not_a_dict(self, tmp_path: Path) -> None:
        bad = _write_pickle(
            tmp_path / "bad.pkl",
            {"features": [1.0, 2.0], "labels": np.array([0, 1], dtype=np.int64)},
        )
        assert _do_migration(_args(bad, tmp_path / "out.npz")) == 3

    def test_non_string_feature_key(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = _write_pickle(
            tmp_path / "bad.pkl",
            {
                "features": {7: np.array([1.0], dtype=np.float32)},
                "labels": np.array([0], dtype=np.int64),
            },
        )
        rc = _do_migration(_args(bad, tmp_path / "out.npz"))
        assert rc == 3
        assert "feature key 7 is not a string" in capsys.readouterr().err

    def test_object_dtype_feature_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = _write_pickle(
            tmp_path / "bad.pkl",
            {
                "features": {"weird": np.array([{"k": 1}], dtype=object)},
                "labels": np.array([0], dtype=np.int64),
            },
        )
        rc = _do_migration(_args(bad, tmp_path / "out.npz"))
        assert rc == 3
        assert "object dtype" in capsys.readouterr().err
        assert not (tmp_path / "out.npz").exists()

    def test_object_dtype_labels_are_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = _write_pickle(
            tmp_path / "bad.pkl",
            {
                "features": {"x": np.array([1.0], dtype=np.float32)},
                "labels": np.array([{"k": 1}], dtype=object),
            },
        )
        rc = _do_migration(_args(bad, tmp_path / "out.npz"))
        assert rc == 3
        assert "labels contain object dtype" in capsys.readouterr().err

    def test_reserved_allow_pickle_feature_key_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A feature literally named "allow_pickle" would be swallowed by
        # numpy.savez's kw-only parameter (numpy 2.x) instead of stored.
        bad = _write_pickle(
            tmp_path / "bad.pkl",
            {
                "features": {"allow_pickle": np.array([1.0], dtype=np.float32)},
                "labels": np.array([0], dtype=np.int64),
            },
        )
        rc = _do_migration(_args(bad, tmp_path / "out.npz"))
        assert rc == 3
        err = capsys.readouterr().err
        assert "allow_pickle" in err
        assert "reserved" in err
        assert not (tmp_path / "out.npz").exists()


# ---------------------------------------------------------------------------
# _do_migration: signing-key errors (exit code 4)
# ---------------------------------------------------------------------------


class TestSigningKeyErrors:
    def test_non_hex_key(
        self, tmp_path: Path, legacy_pkl: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = _do_migration(_args(legacy_pkl, tmp_path / "out.npz", "--sign-key-hex", "zz" * 32))
        assert rc == 4
        assert "not valid hex" in capsys.readouterr().err
        assert not (tmp_path / "out.npz.sig").exists()

    def test_short_key(
        self, tmp_path: Path, legacy_pkl: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = _do_migration(_args(legacy_pkl, tmp_path / "out.npz", "--sign-key-hex", "11" * 8))
        assert rc == 4
        assert ">= 32 bytes" in capsys.readouterr().err
        assert not (tmp_path / "out.npz.sig").exists()


# ---------------------------------------------------------------------------
# Adversarial payloads: malicious pickles must NOT execute (exit code 5)
# ---------------------------------------------------------------------------


class _OsSystemPayload:
    """Classic reduce -> ``os.system`` RCE payload; command set per test."""

    command: ClassVar[str] = "true"

    def __reduce__(self) -> tuple[Any, tuple[str, ...]]:
        return (os.system, (type(self).command,))


class _EvalPayload:
    """``builtins.eval`` bootstrap payload."""

    def __reduce__(self) -> tuple[Any, tuple[str, ...]]:
        return (eval, ("__import__('os').system('true')",))


class _ImportPayload:
    """``builtins.__import__`` bootstrap payload."""

    def __reduce__(self) -> tuple[Any, tuple[str, ...]]:
        return (__import__, ("os",))


class TestMaliciousPayloadsAreRefused:
    def test_os_system_payload_is_refused_and_never_executes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sentinel = tmp_path / "rce_proof"
        _OsSystemPayload.command = f"touch {sentinel}"
        malicious = _write_pickle(tmp_path / "evil.pkl", _OsSystemPayload())

        rc = _do_migration(_args(malicious, tmp_path / "out.npz"))

        assert rc == _EXIT_RESTRICTED_UNPICKLER_REFUSAL
        err = capsys.readouterr().err
        assert "refusing global" in err
        assert "system'" in err  # resolved as os.system or posix.system
        # The load-bearing assertion: the payload must never have run.
        assert not sentinel.exists(), "RCE payload executed; restricted unpickler failed open!"
        assert not (tmp_path / "out.npz").exists()

    @pytest.mark.parametrize(
        ("payload_cls", "refused_name"),
        [
            (_EvalPayload, "builtins.eval"),
            (_ImportPayload, "builtins.__import__"),
        ],
    )
    def test_builtins_bootstrap_payloads_are_refused(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        payload_cls: type,
        refused_name: str,
    ) -> None:
        malicious = _write_pickle(tmp_path / "evil.pkl", payload_cls())
        rc = _do_migration(_args(malicious, tmp_path / "out.npz"))
        assert rc == _EXIT_RESTRICTED_UNPICKLER_REFUSAL
        assert f"refusing global '{refused_name}'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main(): sentinel dispatch and hardened relaunch contract
# ---------------------------------------------------------------------------


class TestMainDispatch:
    def test_child_mode_runs_migration_in_process(
        self, tmp_path: Path, legacy_pkl: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With the sentinel set, main() must NOT relaunch: it performs the
        # migration directly (this is what the hardened child executes).
        monkeypatch.setenv(_HARDENED_SENTINEL, "1")
        out_path = tmp_path / "out.npz"
        rc = main(["--input", str(legacy_pkl), "--output", str(out_path)])
        assert rc == 0
        assert out_path.exists()

    def test_parent_mode_relaunches_with_scrubbed_env(
        self,
        tmp_path: Path,
        legacy_pkl: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.delenv(_HARDENED_SENTINEL, raising=False)
        # Poison the parent environment with injection vectors that the
        # relaunch MUST NOT forward to the hardened child.
        monkeypatch.setenv("LD_PRELOAD", "/tmp/evil.so")
        monkeypatch.setenv("PYTHONPATH", "/tmp/evil")
        monkeypatch.setenv("PYTHONSTARTUP", "/tmp/evil.py")

        calls: list[tuple[str, list[str], dict[str, str]]] = []

        def fake_python_module(
            module: str,
            args: Sequence[str] = (),
            *,
            env: Mapping[str, str] | None = None,
            check: bool = False,
            capture_output: bool = False,
            timeout: float | None = None,
        ) -> subprocess.CompletedProcess[str]:
            calls.append((module, list(args), dict(env or {})))
            return subprocess.CompletedProcess(args=[module], returncode=7)

        monkeypatch.setattr(safe_exec, "python_module", fake_python_module)

        argv = ["--input", str(legacy_pkl), "--output", str(tmp_path / "out.npz")]
        rc = main(argv)

        # Child exit code is propagated verbatim.
        assert rc == 7
        stderr = capsys.readouterr().err
        assert "hardened subprocess" in stderr
        assert "relaunching" in stderr

        assert len(calls) == 1
        module, forwarded, env = calls[0]
        assert module == "omni_mercury_engine.tools.migrate_pkl"
        assert forwarded == argv
        # Hardening contract: sentinel + no-user-site set, injection vectors
        # dropped, PATH forwarded, LANG always present.
        assert env[_HARDENED_SENTINEL] == "1"
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert "LD_PRELOAD" not in env
        assert "PYTHONPATH" not in env
        assert "PYTHONSTARTUP" not in env
        assert env["PATH"] == os.environ["PATH"]
        assert "LANG" in env

    def test_parent_mode_falls_back_to_sys_argv(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(_HARDENED_SENTINEL, raising=False)
        calls: list[list[str]] = []

        def fake_python_module(
            module: str,
            args: Sequence[str] = (),
            *,
            env: Mapping[str, str] | None = None,
            check: bool = False,
            capture_output: bool = False,
            timeout: float | None = None,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(list(args))
            return subprocess.CompletedProcess(args=[module], returncode=0)

        monkeypatch.setattr(safe_exec, "python_module", fake_python_module)
        cli_args = ["--input", str(tmp_path / "a.pkl"), "--output", str(tmp_path / "b.npz")]
        monkeypatch.setattr(sys, "argv", ["migrate_pkl", *cli_args])

        assert main(None) == 0
        assert calls == [cli_args]
