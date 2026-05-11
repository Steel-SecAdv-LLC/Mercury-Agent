"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for security/model_policy.py (Bandit B615 architecture).

Each test exercises the policy contract that authorises
``HuggingFace.from_pretrained`` calls in the rest of the codebase.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.security.model_policy import (
    REMOTE_CODE_ALLOWED_NAMESPACES,
    TRUSTED_NAMESPACES,
    HFModelPolicy,
    UnsafeModelError,
)


class TestHFModelPolicyAllowlist:
    """Namespace allowlist behaviour."""

    def test_known_namespace_with_revision_passes(self):
        HFModelPolicy.validate("meta-llama/Llama-3.2-1B", revision="abc1234")

    def test_unknown_namespace_rejected(self):
        with pytest.raises(UnsafeModelError, match="trusted allowlist"):
            HFModelPolicy.validate("attacker/evil", revision="abc1234")

    def test_extra_namespace_env_extends_allowlist(self, monkeypatch):
        monkeypatch.setenv("MERCURY_HF_EXTRA_NAMESPACES", "internal-org")
        HFModelPolicy.validate("internal-org/private-model", revision="abc1234")

    def test_known_namespaces_are_strings(self):
        # Sanity: every entry parses as a single segment.
        for ns in TRUSTED_NAMESPACES:
            assert isinstance(ns, str) and "/" not in ns


class TestHFModelPolicyModelIdShape:
    """Identifier shape and parsing."""

    def test_malformed_id_rejected(self):
        with pytest.raises(UnsafeModelError, match="malformed"):
            HFModelPolicy.validate("not-a-real-id", revision="abc1234")

    def test_dangerous_chars_rejected(self):
        with pytest.raises(UnsafeModelError, match="malformed"):
            HFModelPolicy.validate("meta-llama/bad name", revision="abc1234")

    def test_empty_id_rejected(self):
        with pytest.raises(UnsafeModelError):
            HFModelPolicy.validate("", revision="abc1234")


class TestHFModelPolicyLocalPaths:
    """Local-path bypass — operator-managed weights."""

    @pytest.mark.parametrize(
        "path",
        [
            "/abs/path/to/model",
            "./relative/model",
            "../relative/model",
            "~/models/llama",
        ],
    )
    def test_local_paths_pass_without_revision(self, path):
        HFModelPolicy.validate(path)  # should not raise


class TestHFModelPolicyRevisionPinning:
    """Revision-pinning enforcement."""

    def test_missing_revision_rejected_in_production_mode(self, monkeypatch):
        monkeypatch.setenv("MERCURY_HF_REQUIRE_REVISION", "true")
        with pytest.raises(UnsafeModelError, match="pinned revision"):
            HFModelPolicy.validate("meta-llama/Llama-3.2-1B")

    def test_missing_revision_allowed_when_disabled(self, monkeypatch):
        monkeypatch.setenv("MERCURY_HF_REQUIRE_REVISION", "false")
        HFModelPolicy.validate("meta-llama/Llama-3.2-1B")

    def test_malformed_revision_rejected(self):
        with pytest.raises(UnsafeModelError, match="revision"):
            HFModelPolicy.validate(
                "meta-llama/Llama-3.2-1B", revision="bad revision with spaces"
            )


class TestHFModelPolicyTrustRemoteCode:
    """``trust_remote_code=True`` requires explicit namespace opt-in."""

    def test_remote_code_rejected_for_non_allowlisted_namespace(self):
        with pytest.raises(UnsafeModelError, match="trust_remote_code"):
            HFModelPolicy.validate(
                "meta-llama/Llama-3.2-1B", revision="abc1234", trust_remote_code=True
            )

    def test_remote_code_allowed_for_qwen(self):
        HFModelPolicy.validate(
            "Qwen/Qwen2-VL-2B", revision="abc1234", trust_remote_code=True
        )

    def test_remote_code_allowed_list_is_subset(self):
        # Every namespace permitted for trust_remote_code must also be in
        # the general namespace allowlist (defence-in-depth).
        assert REMOTE_CODE_ALLOWED_NAMESPACES <= TRUSTED_NAMESPACES


class TestValidateVettedDataset:
    """Vetted-dataset gate (used by datasets/security.py)."""

    ALLOWLIST = ("bvk/CICIDS-2017", "Riccorl/CIC-IDS-2017")

    def test_member_passes(self):
        HFModelPolicy.validate_vetted_dataset(
            "bvk/CICIDS-2017", allowlist=self.ALLOWLIST, revision="main"
        )

    def test_non_member_rejected(self):
        with pytest.raises(UnsafeModelError, match="vetted allowlist"):
            HFModelPolicy.validate_vetted_dataset(
                "attacker/Hijacked", allowlist=self.ALLOWLIST, revision="main"
            )

    def test_bad_revision_rejected(self):
        with pytest.raises(UnsafeModelError, match="revision"):
            HFModelPolicy.validate_vetted_dataset(
                "bvk/CICIDS-2017",
                allowlist=self.ALLOWLIST,
                revision="bad revision",
            )
