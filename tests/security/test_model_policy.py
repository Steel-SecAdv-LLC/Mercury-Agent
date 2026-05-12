"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for ``security/model_policy.HFModelPolicy``.

The policy is the only thing standing between Mercury Agent's
``from_pretrained`` callsites and arbitrary code execution from a
compromised Hugging Face repository.  These tests exercise every gate:
identifier shape, namespace allow-list, revision pinning, ``trust_remote_code``
allow-list, dataset namespace allow-list, and the local-path bypass.

No mocks against the Hugging Face Hub are needed — the policy is a
pure function over its arguments.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from omni_mercury_engine.security.model_policy import (
    REMOTE_CODE_ALLOWED_NAMESPACES,
    TRUSTED_DATASET_NAMESPACES,
    TRUSTED_NAMESPACES,
    HFModelPolicy,
    UnsafeModelError,
)

# A revision string used to satisfy revision-pinning gates in tests that
# are not specifically about that gate.  The exact value is irrelevant —
# the policy enforces presence, not a SHA-256 shape.
_PIN = "1234567890abcdef1234567890abcdef12345678"


@pytest.fixture(autouse=True)
def _require_revision_default():
    """Default test environment to require revision pinning (production mode)."""
    with patch.dict(os.environ, {"MERCURY_HF_REQUIRE_REVISION": "true"}, clear=False):
        yield


class TestIdentifierShape:
    """A malformed identifier is rejected before any namespace check."""

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",  # empty string
            "namespace//name",  # double slash
            "namespace/name/extra",  # too many segments
            "a b/name",  # space in namespace
            "a$ns/name",  # $ in namespace
            "ns/" + "x" * 200,  # name way too long
        ],
    )
    def test_malformed_rejected(self, bad_id):
        with pytest.raises(UnsafeModelError):
            HFModelPolicy.validate(bad_id, revision=_PIN)

    def test_non_string_rejected(self):
        with pytest.raises(UnsafeModelError):
            HFModelPolicy.validate(None, revision=_PIN)  # type: ignore[arg-type]


class TestNamespaceAllowList:
    """Namespace must be in TRUSTED_NAMESPACES (unless local path)."""

    def test_trusted_namespace_passes(self):
        # ``Salesforce`` is in TRUSTED_NAMESPACES
        HFModelPolicy.validate("Salesforce/blip-image-captioning-base", revision=_PIN)

    def test_unknown_namespace_rejected(self):
        with pytest.raises(UnsafeModelError) as excinfo:
            HFModelPolicy.validate("attacker/evil-model", revision=_PIN)
        assert "attacker" in str(excinfo.value)

    def test_legacy_single_segment_rejected(self):
        # Synthetic ``"_legacy"`` namespace is never in TRUSTED_NAMESPACES.
        with pytest.raises(UnsafeModelError):
            HFModelPolicy.validate("bert-base-uncased", revision=_PIN)

    def test_every_trusted_namespace_accepts(self):
        for ns in TRUSTED_NAMESPACES:
            HFModelPolicy.validate(f"{ns}/some-model", revision=_PIN)


class TestRevisionPinning:
    """Revision pin is mandatory when MERCURY_HF_REQUIRE_REVISION is on."""

    def test_missing_revision_rejected_in_production(self):
        with pytest.raises(UnsafeModelError) as excinfo:
            HFModelPolicy.validate("Salesforce/blip-image-captioning-base", revision=None)
        assert "revision" in str(excinfo.value).lower()

    def test_blank_revision_rejected(self):
        with pytest.raises(UnsafeModelError):
            HFModelPolicy.validate("Salesforce/foo", revision="   ")

    def test_revision_optional_when_env_off(self):
        with patch.dict(os.environ, {"MERCURY_HF_REQUIRE_REVISION": "false"}):
            # No revision but env says it's not mandatory; namespace check still runs.
            HFModelPolicy.validate("Salesforce/foo", revision=None)
            with pytest.raises(UnsafeModelError):
                HFModelPolicy.validate("attacker/foo", revision=None)


class TestTrustRemoteCode:
    """``trust_remote_code=True`` is only allowed for vetted namespaces."""

    def test_rejected_for_default_namespace(self):
        # ``Salesforce`` is in TRUSTED_NAMESPACES but NOT in
        # REMOTE_CODE_ALLOWED_NAMESPACES.
        with pytest.raises(UnsafeModelError) as excinfo:
            HFModelPolicy.validate(
                "Salesforce/blip-image-captioning-base",
                revision=_PIN,
                trust_remote_code=True,
            )
        assert "trust_remote_code" in str(excinfo.value)

    def test_allowed_for_openbmb(self):
        # openbmb is in REMOTE_CODE_ALLOWED_NAMESPACES.
        HFModelPolicy.validate(
            "openbmb/MiniCPM-V-2_6",
            revision=_PIN,
            trust_remote_code=True,
        )

    def test_default_false_does_not_require_allow_list(self):
        HFModelPolicy.validate(
            "Salesforce/blip-image-captioning-base",
            revision=_PIN,
            trust_remote_code=False,
        )

    def test_every_remote_code_namespace_in_trusted(self):
        assert REMOTE_CODE_ALLOWED_NAMESPACES.issubset(TRUSTED_NAMESPACES)


class TestLocalPathBypass:
    """Absolute / explicit relative paths bypass the network policy."""

    def test_absolute_path_accepted(self, tmp_path):
        p = str(tmp_path / "my-model")
        HFModelPolicy.validate(p, revision=None, trust_remote_code=False)

    def test_relative_path_accepted(self):
        HFModelPolicy.validate("./local-model", revision=None)
        HFModelPolicy.validate("../local-model", revision=None)

    def test_local_path_with_trust_remote_code_logs_but_passes(self, caplog):
        with caplog.at_level("WARNING"):
            HFModelPolicy.validate("./local-model", trust_remote_code=True)
        assert any("trust_remote_code" in rec.message for rec in caplog.records)


class TestDatasetPolicy:
    """``validate_vetted_dataset`` uses TRUSTED_DATASET_NAMESPACES."""

    def test_trusted_dataset_accepted(self):
        HFModelPolicy.validate_vetted_dataset("bvk/CICIDS-2017", revision=_PIN)

    def test_untrusted_dataset_rejected(self):
        with pytest.raises(UnsafeModelError):
            HFModelPolicy.validate_vetted_dataset("attacker/poisoned", revision=_PIN)

    def test_dataset_namespace_set_is_separate(self):
        # Model namespaces should not silently widen the dataset allow-list.
        only_in_dataset = TRUSTED_DATASET_NAMESPACES - TRUSTED_NAMESPACES
        # ``bvk``, ``Riccorl``, ``tcabanski`` etc. are dataset-only.
        assert only_in_dataset, "Dataset namespaces must not be a subset of model namespaces"


class TestPolicySummary:
    """``policy_summary`` is a debug helper; sanity-check its shape."""

    def test_summary_has_expected_keys(self):
        s = HFModelPolicy.policy_summary()
        assert "require_revision" in s
        assert "trusted_namespaces" in s
        assert "trusted_dataset_namespaces" in s
        assert "remote_code_allowed_namespaces" in s
        assert isinstance(s["require_revision"], bool)


class TestAssertNamespaceTrusted:
    """``assert_namespace_trusted`` returns (ns, name) without revision policy."""

    def test_trusted_returns_tuple(self):
        ns, name = HFModelPolicy.assert_namespace_trusted("Salesforce/foo-bar")
        assert ns == "Salesforce"
        assert name == "foo-bar"

    def test_untrusted_raises(self):
        with pytest.raises(UnsafeModelError):
            HFModelPolicy.assert_namespace_trusted("attacker/foo")

    def test_local_path_passes(self, tmp_path):
        ns, name = HFModelPolicy.assert_namespace_trusted(str(tmp_path / "model"))
        # Local path is opaque to the validator; we just don't raise.
        assert ns and name
