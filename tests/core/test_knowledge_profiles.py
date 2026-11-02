# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The knowledge-profile registry is complete, distinct and grounded.

``core/knowledge_profiles.py`` states, for every domain and every detector,
whose knowledge it is: the discipline, the professions, the institutions, the
lineage and the references. These tests make the authoring rules enforceable:

* coverage is exact against the detector manifest and the loader and hazard
  domains, and no profile is orphaned;
* no two detectors share a mechanism, lineage or practitioner description,
  and no two domains share a method, label rule or observables sentence --
  a copied sentence defeats the purpose, which is distinguishability;
* every profile cites a reference and every domain names an institution with
  an ``https`` URL and a stated role;
* a profile adds to the manifest instead of repeating it: a detector's
  mechanism is never its manifest description verbatim, and an adapter's
  profile names the detector it serves;
* the profile vocabulary stays consistent with the ontology: a detector the
  ontology says cannot detect point anomalies does not claim them in
  ``use_when``.
"""

from __future__ import annotations

import re

import pytest

from omni_mercury_engine.core.detector_ontology import ONTOLOGY
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
from omni_mercury_engine.core.knowledge_profiles import (
    DETECTOR_PROFILES,
    DOMAIN_PROFILES,
    detector_profile,
    domain_profile,
    profile_dict,
    validate_profiles,
)
from omni_mercury_engine.core.loader_declarations import loader_declarations, loader_domains
from omni_mercury_engine.detectors.hazard_diagnostics import KNOWN_HAZARDS

_MANIFEST = {entry.name: entry for entry in DETECTOR_MANIFEST}
_DOMAINS = set(loader_domains()) | set(KNOWN_HAZARDS)


class TestCoverage:
    def test_every_manifest_detector_has_a_profile(self) -> None:
        missing = sorted(set(_MANIFEST) - set(DETECTOR_PROFILES))
        assert not missing, f"detectors without a knowledge profile: {missing}"

    def test_no_detector_profile_is_orphaned(self) -> None:
        orphans = sorted(set(DETECTOR_PROFILES) - set(_MANIFEST))
        assert not orphans, f"profiles for detectors not in the manifest: {orphans}"

    def test_every_loader_and_hazard_domain_has_a_profile(self) -> None:
        missing = sorted(_DOMAINS - set(DOMAIN_PROFILES))
        assert not missing, f"domains without a knowledge profile: {missing}"

    def test_no_domain_profile_is_orphaned(self) -> None:
        orphans = sorted(set(DOMAIN_PROFILES) - _DOMAINS)
        assert not orphans, f"profiles for unknown domains: {orphans}"

    def test_the_registry_validates_itself_against_the_live_sets(self) -> None:
        assert validate_profiles(set(_MANIFEST), _DOMAINS) == []

    def test_lookup_helpers(self) -> None:
        assert detector_profile("bocpd") is DETECTOR_PROFILES["bocpd"]
        assert detector_profile("no-such-detector") is None
        assert domain_profile("sepsis") is DOMAIN_PROFILES["sepsis"]
        assert domain_profile("no-such-domain") is None


class TestDistinctness:
    @pytest.mark.parametrize("field_name", ["mechanism", "lineage", "practitioners", "discipline"])
    def test_detector_text_is_not_copied_between_profiles(self, field_name: str) -> None:
        texts = [
            " ".join(getattr(p, field_name).lower().split()) for p in DETECTOR_PROFILES.values()
        ]
        if field_name == "discipline":
            # Disciplines legitimately repeat within a family, but never across more than a handful.
            counts = {t: texts.count(t) for t in set(texts)}
            crowded = {t: n for t, n in counts.items() if n > 4}
            assert not crowded, f"a discipline label used as a template: {crowded}"
            return
        assert len(set(texts)) == len(texts), f"{field_name} repeated between detector profiles"

    @pytest.mark.parametrize("field_name", ["method", "label_rule", "observables", "title"])
    def test_domain_text_is_not_copied_between_profiles(self, field_name: str) -> None:
        texts = [" ".join(getattr(p, field_name).lower().split()) for p in DOMAIN_PROFILES.values()]
        assert len(set(texts)) == len(texts), f"{field_name} repeated between domain profiles"

    def test_a_mechanism_adds_to_the_manifest_description(self) -> None:
        for name, profile in DETECTOR_PROFILES.items():
            description = " ".join(_MANIFEST[name].description.lower().split())
            mechanism = " ".join(profile.mechanism.lower().split())
            assert mechanism != description, f"{name}: the mechanism merely repeats the manifest"
            assert len(mechanism) > len(
                description
            ), f"{name}: the mechanism says less than the manifest"

    def test_domain_institutions_are_distinct_sets(self) -> None:
        seen: dict[frozenset[str], str] = {}
        for domain, profile in DOMAIN_PROFILES.items():
            key = frozenset(i.name for i in profile.institutions)
            assert key not in seen, f"{domain} and {seen[key]} name the same institutions"
            seen[key] = domain


class TestGrounding:
    def test_every_detector_profile_cites_a_reference(self) -> None:
        for name, profile in DETECTOR_PROFILES.items():
            assert profile.references, f"{name} cites nothing"
            for ref in profile.references:
                assert len(ref.citation) >= 12, f"{name}: empty citation"
                if ref.url:
                    assert ref.url.startswith("https://"), f"{name}: {ref.url} is not https"

    def test_every_domain_names_a_governing_institution(self) -> None:
        for domain, profile in DOMAIN_PROFILES.items():
            assert profile.institutions, f"{domain} names no institution"
            assert any(i.url.startswith("https://") for i in profile.institutions), domain
            for inst in profile.institutions:
                assert inst.role, f"{domain}: {inst.name} has no stated role"

    def test_every_domain_states_professions_expertise_and_limits(self) -> None:
        for domain, profile in DOMAIN_PROFILES.items():
            assert len(profile.professions) >= 2, f"{domain}: fewer than two professions"
            assert len(profile.expertise) >= 2, f"{domain}: fewer than two expertise items"
            assert profile.limits, f"{domain}: no limits stated"

    def test_adapter_profiles_name_the_detector_they_serve(self) -> None:
        for name, entry in _MANIFEST.items():
            if not entry.wraps:
                continue
            wrapped_class = entry.wraps.rsplit(":", 1)[-1]
            text = " ".join([DETECTOR_PROFILES[name].mechanism, DETECTOR_PROFILES[name].lineage])
            # The served detector's entry point (ClassName.method) or the class itself is named.
            stem = re.sub(r"(Detector|Cascade|Adapter)$", "", wrapped_class)
            assert (
                wrapped_class in text or stem.lower() in text.lower()
            ), f"{name} serves {wrapped_class} but its profile never names it"

    def test_use_when_never_claims_a_declared_blind_spot(self) -> None:
        words = {
            "point": r"\bpoint (?:anomal|outlier|spike)",
            "change_point": r"\bchange[- ]point",
            "trend_drift": r"\bdrift\b",
            "level_shift": r"\blevel shift",
        }
        for name, profile in DETECTOR_PROFILES.items():
            for cls in ONTOLOGY[name].cannot_detect:
                pattern = words.get(cls)
                if pattern is None:
                    continue
                assert not re.search(
                    pattern, profile.use_when.lower()
                ), f"{name}: use_when claims {cls}, which the ontology says it cannot detect"

    def test_domain_profile_label_rule_matches_the_loader_declaration(self) -> None:
        for domain, declaration in loader_declarations().items():
            profile = DOMAIN_PROFILES[domain]
            if declaration.label_source == "ground_truth":
                assert profile.label_rule.lower().startswith("ground truth"), domain
            else:
                assert profile.label_rule.lower().startswith("statistical"), domain

    def test_domains_with_a_key_say_so(self) -> None:
        for domain, declaration in loader_declarations().items():
            if declaration.requires_api_key:
                text = " ".join(
                    DOMAIN_PROFILES[domain].limits + (DOMAIN_PROFILES[domain].method,)
                ).lower()
                assert (
                    "key" in text
                ), f"{domain} needs {declaration.api_key_env_var} but its profile does not say so"


class TestSerialisation:
    def test_profiles_serialise_to_plain_json_types(self) -> None:
        import json

        for profile in list(DETECTOR_PROFILES.values()) + list(DOMAIN_PROFILES.values()):
            data = profile_dict(profile)
            json.dumps(data)
            assert isinstance(data["references"], list)

    def test_validation_reports_a_missing_detector(self) -> None:
        problems = validate_profiles(set(_MANIFEST) | {"phantom"}, _DOMAINS)
        assert any("phantom" in p for p in problems)
