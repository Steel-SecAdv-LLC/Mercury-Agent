# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""A configured JWT signing key must be real key material, not merely present.

``JWTAuth`` only ever checked whether ``JWT_SECRET_KEY`` was *absent*. Two
things slipped through:

* ``secret_key or os.getenv("JWT_SECRET_KEY")`` evaluates to ``""`` when the
  variable is set but empty, and ``"" is None`` is false -- so the production
  branch was skipped and every token was signed with the empty string. An
  empty Kubernetes Secret value, a bare ``JWT_SECRET_KEY=`` line, or a failed
  secret-manager lookup all produce exactly that.
* ``k8s/base/secret.yaml`` shipped
  ``JWT_SECRET_KEY: "CHANGE_ME_IN_PRODUCTION_USE_SECURE_RANDOM"`` inside the
  base kustomization's ``resources:``. ``kubectl apply -k k8s/base`` installed
  a signing key that is published in this repository -- and because it was
  present, it satisfied the guard.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.api.auth import JWTAuth

_REAL_KEY = "f3a9c1d2e4b5a6978c0d1e2f3a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d"

#: Values that must never end up signing a production token.
_UNUSABLE = {
    "k8s_base_secret": "CHANGE_ME_IN_PRODUCTION_USE_SECURE_RANDOM",
    "env_example": "your-secure-random-key-here-generate-with-openssl-rand-hex-32",
    "lowercase_placeholder": "changeme-please-before-going-to-production",
    "too_short": "abc123def456",
    "single_character": "a" * 64,
    "two_characters": "ab" * 32,
}


@pytest.fixture()
def production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERCURY_ENV", "production")


@pytest.fixture()
def development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERCURY_ENV", "development")


class TestEmptyIsNotAKey:
    """An empty value is treated as unset, in every environment."""

    @pytest.mark.parametrize("value", ["", "   ", "\n", "\t "])
    def test_blank_values_normalise_to_unset(self, development: None, value: str) -> None:
        assert JWTAuth._validate_configured_key(value) is None

    def test_blank_values_normalise_to_unset_in_production(self, production: None) -> None:
        assert JWTAuth._validate_configured_key("") is None

    def test_empty_env_var_does_not_become_the_signing_key(
        self, development: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: the constructor must not adopt ``""`` as its key."""
        monkeypatch.setenv("JWT_SECRET_KEY", "")
        auth = JWTAuth()
        assert auth.secret_key != ""
        assert auth.using_fallback is True

    def test_unset_env_var_still_reaches_the_dev_fallback(
        self, development: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
        assert JWTAuth().using_fallback is True


class TestProductionRejectsWeakMaterial:
    """In production a placeholder or low-entropy key is a hard failure."""

    @pytest.mark.parametrize(("label", "value"), sorted(_UNUSABLE.items()))
    def test_rejected(self, production: None, label: str, value: str) -> None:
        with pytest.raises(ValueError, match="not usable in production"):
            JWTAuth._validate_configured_key(value)

    @pytest.mark.parametrize(("label", "value"), sorted(_UNUSABLE.items()))
    def test_rejected_through_the_constructor(
        self, production: None, monkeypatch: pytest.MonkeyPatch, label: str, value: str
    ) -> None:
        monkeypatch.setenv("JWT_SECRET_KEY", value)
        with pytest.raises(ValueError, match="not usable in production"):
            JWTAuth()

    def test_real_key_is_accepted(self, production: None) -> None:
        assert JWTAuth._validate_configured_key(_REAL_KEY) == _REAL_KEY

    def test_surrounding_whitespace_is_stripped(self, production: None) -> None:
        """A trailing newline from a mounted secret file must not change the key."""
        assert JWTAuth._validate_configured_key(f"  {_REAL_KEY}\n") == _REAL_KEY

    def test_explicit_argument_is_vetted_too(self, production: None) -> None:
        """The constructor argument is not a bypass for the env-var guard."""
        with pytest.raises(ValueError, match="not usable in production"):
            JWTAuth(secret_key="CHANGE_ME_IN_PRODUCTION_USE_SECURE_RANDOM")  # noqa: S106


class TestDevelopmentDegradesToAWarning:
    """A weak key is not a security boundary on a laptop."""

    @pytest.mark.parametrize(("label", "value"), sorted(_UNUSABLE.items()))
    def test_accepted_with_a_warning(
        self,
        development: None,
        caplog: pytest.LogCaptureFixture,
        label: str,
        value: str,
    ) -> None:
        with caplog.at_level("WARNING"):
            assert JWTAuth._validate_configured_key(value) == value
        assert "would be rejected in production" in caplog.text

    def test_short_test_keys_still_work(self, development: None) -> None:
        """The suite's own fixtures must keep working unchanged."""
        assert (
            JWTAuth._validate_configured_key("test_secret_key_for_testing")
            == "test_secret_key_for_testing"
        )


class TestPlaceholderMarkersAreDistinctive:
    """The substring list must not be able to reject real key material."""

    def test_markers_cannot_occur_in_hex(self) -> None:
        hex_alphabet = set("0123456789abcdef")
        for marker in JWTAuth._PLACEHOLDER_MARKERS:
            assert not set(marker) <= hex_alphabet, marker

    def test_markers_are_long_enough_to_be_improbable(self) -> None:
        """Short markers would make the guard flaky on real random keys.

        A 4-character marker collides with a 64-character url-safe token
        roughly once in 3e5 keys; an 8-character one, once in 3e13.
        """
        for marker in JWTAuth._PLACEHOLDER_MARKERS:
            assert len(marker) >= 8, marker

    def test_random_urlsafe_keys_pass(self) -> None:
        import secrets

        for _ in range(200):
            key = secrets.token_urlsafe(48)
            assert JWTAuth._validate_configured_key(key) == key
