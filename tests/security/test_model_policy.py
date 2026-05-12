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

Gate tests for :mod:`omni_mercury_engine.security.model_policy`.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.security.model_policy import (
    HFModelPolicy,
    SafeHFLoader,
    UnsafeModelError,
)


class TestIdentifierShape:
    def test_bare_name_rejected(self) -> None:
        with pytest.raises(UnsafeModelError, match="namespace/name"):
            HFModelPolicy.validate("bert", revision="abc123")

    def test_lookalike_id_rejected(self) -> None:
        # Multiple slashes, leading dots, etc. None of these match
        # the strict HuggingFace namespace/name pattern, and none
        # are absolute local paths.
        for bad in ["evil//slash", ".hidden/model", "name with spaces/model", "no-slash"]:
            with pytest.raises(UnsafeModelError):
                HFModelPolicy.validate(bad, revision="abc123")

    def test_valid_namespace_name_accepted(self) -> None:
        HFModelPolicy.validate("Salesforce/blip-image-captioning-base", revision="abc123")

    def test_absolute_local_path_accepted_without_revision(self) -> None:
        HFModelPolicy.validate("/opt/models/local-bert", revision=None)

    def test_relative_path_rejected(self) -> None:
        # ./foo and ../foo are NOT treated as local paths -- they
        # would let resolution depend on the current working
        # directory. Refuse so the operator commits to an absolute
        # path or to a Hub id.
        for bad in ["./local-model", "../models/foo"]:
            with pytest.raises(UnsafeModelError):
                HFModelPolicy.validate(bad, revision=None)


class TestRevisionPinning:
    def test_remote_without_revision_rejected(self) -> None:
        with pytest.raises(UnsafeModelError, match="revision"):
            HFModelPolicy.validate("Salesforce/blip", revision=None)

    def test_remote_with_empty_revision_rejected(self) -> None:
        with pytest.raises(UnsafeModelError, match="revision"):
            HFModelPolicy.validate("Salesforce/blip", revision="   ")

    def test_remote_with_sha_accepted(self) -> None:
        HFModelPolicy.validate(
            "Salesforce/blip",
            revision="a1b2c3d4e5f6",
        )


class TestAllowlist:
    def test_unlisted_rejected(self) -> None:
        with pytest.raises(UnsafeModelError, match="allowlist"):
            HFModelPolicy.validate(
                "evil/model",
                revision="abc123",
                allowlist={"Salesforce/blip"},
            )

    def test_listed_accepted(self) -> None:
        HFModelPolicy.validate(
            "Salesforce/blip",
            revision="abc123",
            allowlist={"Salesforce/blip"},
        )

    def test_trust_remote_code_requires_allowlist(self) -> None:
        with pytest.raises(UnsafeModelError, match="trust_remote_code"):
            HFModelPolicy.validate(
                "openbmb/MiniCPM-V-2_6",
                revision="abc123",
                trust_remote_code=True,
            )


class TestSafeHFLoaderInvokesValidate:
    """SafeHFLoader.load_* must run HFModelPolicy.validate before from_pretrained."""

    def test_load_model_rejects_unpinned(self) -> None:
        class FakeModel:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> object:
                raise AssertionError("must not be called")

        with pytest.raises(UnsafeModelError, match="revision"):
            SafeHFLoader.load_model(FakeModel, "evil/model", revision=None)

    def test_load_dataset_requires_allowlist_membership(self) -> None:
        with pytest.raises(UnsafeModelError, match="allowlist"):
            SafeHFLoader.load_dataset(
                "evil/dataset",
                allowlist={"good/dataset"},
                revision="abc123",
            )


class TestNoFromPretrainedOutsidePolicy:
    """No code outside security/model_policy.py may call from_pretrained directly."""

    def test_no_from_pretrained_calls_outside_policy(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2] / "src" / "omni_mercury_engine"
        for path in root.rglob("*.py"):
            if path.name == "model_policy.py":
                continue
            content = path.read_text(encoding="utf-8")
            # We look for the literal call pattern '.from_pretrained('
            # which is what bandit's B615 matches. Comments and
            # docstrings using the bare word are fine.
            assert ".from_pretrained(" not in content, f"Direct from_pretrained call in {path}"
