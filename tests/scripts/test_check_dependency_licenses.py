# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The dependency license gate: resolution, closure, review file, exit codes.

Distributions are real ``.dist-info`` directories in a temporary site
directory, read through ``importlib.metadata`` exactly as the gate reads an
installed environment.
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import check_dependency_licenses as gate

APACHE_2_TEXT = "Apache License\n Version 2.0, January 2004\n http://www.apache.org/licenses/\n"
APACHE_1_TEXT = "The Apache Software License, Version 1.1\n Copyright (c) 2000 The Apache Group.\n"
BSD_3_TEXT = "Redistribution and use in source and binary forms, with or without modification...\n"
BSD_4_TEXT = BSD_3_TEXT + (
    "3. All advertising materials mentioning features or use of this software must display...\n"
)


def _install(
    site: Path,
    name: str,
    *,
    version: str = "1.0",
    expression: str | None = None,
    license_field: str | None = None,
    classifiers: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    license_text: str | None = None,
    code: bool = True,
) -> md.Distribution:
    """Install a fake distribution into ``site`` and return it."""
    module = name.replace("-", "_")
    info = site / f"{module}-{version}.dist-info"
    info.mkdir(parents=True)
    lines = ["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}"]
    if expression is not None:
        lines.append(f"License-Expression: {expression}")
    if license_field is not None:
        lines.append("License: " + license_field.replace("\n", "\n        "))
    lines += [f"Classifier: {c}" for c in classifiers]
    lines += [f"Requires-Dist: {r}" for r in requires]
    (info / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = [f"{info.name}/METADATA,,"]
    if license_text is not None:
        (info / "LICENSE").write_text(license_text, encoding="utf-8")
        record.append(f"{info.name}/LICENSE,,")
    if code:
        package = site / module
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        record.append(f"{module}/__init__.py,,")
    record.append(f"{info.name}/RECORD,,")
    (info / "RECORD").write_text("\n".join(record) + "\n", encoding="utf-8")
    return md.PathDistribution(info)


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private site directory, first on ``sys.path``."""
    path = tmp_path / "site"
    path.mkdir()
    monkeypatch.syspath_prepend(str(path))
    return path


class TestEvaluateExpression:
    """SPDX expressions: AND needs all, OR needs one, WITH never rescues."""

    @pytest.mark.parametrize(
        ("expression", "ok"),
        [
            ("MIT", True),
            ("mit", True),
            ("MIT AND Apache-2.0", True),
            ("MIT AND GPL-2.0-only", False),
            ("MIT OR GPL-2.0-only", True),
            ("GPL-2.0-only OR MIT", True),
            ("(MIT OR CNRI-Python) AND Apache-2.0", True),
            ("(GPL-2.0-only OR CNRI-Python) AND Apache-2.0", False),
            ("Apache-2.0 AND CNRI-Python", False),
            ("Apache-2.0 WITH LLVM-exception", True),
            ("GPL-2.0-only WITH Classpath-exception-2.0", False),
            ("LicenseRef-NVIDIA-Proprietary", False),
            ("BSD-4-Clause", False),
        ],
    )
    def test_semantics(self, expression: str, ok: bool) -> None:
        """Each operator applies its own rule."""
        assert gate.evaluate_expression(expression) is ok

    @pytest.mark.parametrize("bad", ["MIT AND", "(MIT", "MIT )", "AND MIT", "MIT WITH", "()"])
    def test_malformed_raises(self, bad: str) -> None:
        """A malformed expression is an error, never a verdict."""
        with pytest.raises(ValueError):
            gate.evaluate_expression(bad)


class TestAssess:
    """Metadata first, then the License field, then classifiers; texts settle families."""

    def test_expression_compatible(self, site: Path) -> None:
        """A compatible PEP 639 expression passes, with its signature."""
        a = gate.assess(_install(site, "ga", expression="MIT OR Apache-2.0"))
        assert (a.verdict, a.signature) == ("compatible", "License-Expression: MIT OR Apache-2.0")

    def test_expression_incompatible(self, site: Path) -> None:
        """An incompatible expression fails."""
        a = gate.assess(_install(site, "ga", expression="Apache-2.0 AND CNRI-Python"))
        assert a.verdict == "incompatible"

    def test_expression_wins_over_classifiers(self, site: Path) -> None:
        """The expression is authoritative over a contradicting classifier."""
        a = gate.assess(
            _install(
                site,
                "ga",
                expression="LicenseRef-Proprietary",
                classifiers=("License :: OSI Approved :: MIT License",),
            )
        )
        assert a.verdict == "incompatible"

    def test_invalid_expression_is_unknown(self, site: Path) -> None:
        """An expression packaging cannot parse is unknown, not guessed."""
        a = gate.assess(_install(site, "ga", expression="Not A License"))
        assert (a.verdict, a.detail) == ("unknown", "invalid SPDX expression")

    def test_versioned_alias(self, site: Path) -> None:
        """'Apache 2.0' is Apache-2.0 without needing a text."""
        a = gate.assess(_install(site, "ga", license_field="Apache 2.0"))
        assert (a.verdict, a.signature) == ("compatible", "License: Apache 2.0")

    def test_license_field_expression(self, site: Path) -> None:
        """A License field that is itself an SPDX expression is evaluated."""
        assert gate.assess(_install(site, "ga", license_field="MPL-2.0 AND MIT")).verdict == (
            "compatible"
        )
        assert gate.assess(_install(site, "gb", license_field="GPL-2.0-only")).verdict == (
            "incompatible"
        )

    def test_bsd_family_settled_by_text(self, site: Path) -> None:
        """'BSD': three-clause text passes, advertising clause fails, no text is unknown."""
        three = gate.assess(_install(site, "ga", license_field="BSD", license_text=BSD_3_TEXT))
        four = gate.assess(_install(site, "gb", license_field="BSD", license_text=BSD_4_TEXT))
        none = gate.assess(_install(site, "gc", license_field="BSD"))
        assert three.verdict == "compatible"
        assert four.verdict == "incompatible"
        assert "advertising" in four.detail
        assert none.verdict == "unknown"

    def test_apache_family_settled_by_text(self, site: Path) -> None:
        """Unversioned 'Apache': 2.0 text passes; 1.1 text or no text is unknown."""
        two = gate.assess(_install(site, "ga", license_field="Apache", license_text=APACHE_2_TEXT))
        one = gate.assess(_install(site, "gb", license_field="Apache", license_text=APACHE_1_TEXT))
        none = gate.assess(
            _install(
                site, "gc", classifiers=("License :: OSI Approved :: Apache Software License",)
            )
        )
        assert two.verdict == "compatible"
        assert one.verdict == "unknown"
        assert none.verdict == "unknown"
        assert none.signature == "Classifier: Apache Software License"

    def test_unknown_literal_falls_to_classifiers(self, site: Path) -> None:
        """'UNKNOWN' in the License field is no information; classifiers decide."""
        a = gate.assess(
            _install(
                site,
                "ga",
                license_field="UNKNOWN",
                classifiers=("License :: OSI Approved :: MIT License",),
            )
        )
        assert (a.verdict, a.signature) == ("compatible", "Classifier: MIT License")

    def test_full_license_text_field_falls_to_classifiers(self, site: Path) -> None:
        """A multi-line License field (a pasted text) defers to classifiers."""
        a = gate.assess(
            _install(
                site,
                "ga",
                license_field="Copyright (c) 2015\nAll rights reserved...",
                classifiers=("License :: OSI Approved :: Python Software Foundation License",),
            )
        )
        assert a.verdict == "compatible"

    def test_classifiers_are_alternatives(self, site: Path) -> None:
        """Any compatible classifier suffices (dual licensing)."""
        a = gate.assess(
            _install(
                site,
                "ga",
                classifiers=(
                    "License :: Other/Proprietary License",
                    "License :: OSI Approved :: MIT License",
                ),
            )
        )
        assert a.verdict == "compatible"

    def test_proprietary_classifier_is_incompatible(self, site: Path) -> None:
        """Other/Proprietary alone fails."""
        a = gate.assess(_install(site, "ga", classifiers=("License :: Other/Proprietary License",)))
        assert (a.verdict, a.detail) == ("incompatible", "proprietary")

    def test_metadata_only_distribution_has_nothing_to_license(self, site: Path) -> None:
        """A meta-package that ships only its dist-info passes."""
        a = gate.assess(_install(site, "ga", code=False))
        assert (a.verdict, a.signature) == ("compatible", "none")

    def test_code_without_any_license_is_unknown(self, site: Path) -> None:
        """Code with no license information fails as unknown."""
        a = gate.assess(_install(site, "ga"))
        assert (a.verdict, a.signature) == ("unknown", "none")

    def test_unrecognised_free_text_is_unknown(self, site: Path) -> None:
        """Free text that is neither an alias nor SPDX is unknown, signed as given."""
        a = gate.assess(_install(site, "ga", license_field="NVIDIA Proprietary Software"))
        assert (a.verdict, a.signature) == ("unknown", "License: NVIDIA Proprietary Software")


class TestDeclaredRequirements:
    """The closure starts from pyproject.toml, self-references expanded."""

    @staticmethod
    def _pyproject(tmp_path: Path) -> Path:
        path = tmp_path / "pyproject.toml"
        path.write_text(
            '[project]\nname = "demo-pkg"\nversion = "1"\n'
            'dependencies = ["ga"]\n'
            "[project.optional-dependencies]\n"
            'x = ["gb"]\n'
            'y = ["gc"]\n'
            'all = ["demo_pkg[x,y]"]\n',
            encoding="utf-8",
        )
        return path

    def test_core_only(self, tmp_path: Path) -> None:
        """No extras: the core dependencies."""
        names = [r.name for r in gate.declared_requirements(self._pyproject(tmp_path), [])]
        assert names == ["ga"]

    def test_self_reference_expands(self, tmp_path: Path) -> None:
        """'all' pulls in x and y through the self-reference."""
        names = {r.name for r in gate.declared_requirements(self._pyproject(tmp_path), ["all"])}
        assert names == {"ga", "gb", "gc"}

    def test_unknown_extra_is_an_error(self, tmp_path: Path) -> None:
        """A typo in --extras is a configuration error, never an empty check."""
        with pytest.raises(gate.GateError, match="no extra 'zz'"):
            gate.declared_requirements(self._pyproject(tmp_path), ["zz"])


class TestResolveClosure:
    """Transitive requirements are followed; markers and extras are honoured."""

    def test_transitive_and_extra_gated(self, site: Path) -> None:
        """A dependency's extra-gated requirement is followed only when requested."""
        from packaging.requirements import Requirement

        _install(site, "ga", requires=("gb", 'gc; extra == "fast"'))
        _install(site, "gb")
        _install(site, "gc")
        plain = gate.resolve_closure([Requirement("ga")])
        assert set(plain) == {"ga", "gb"}
        with_extra = gate.resolve_closure([Requirement("ga[fast]")])
        assert set(with_extra) == {"ga", "gb", "gc"}

    def test_inapplicable_marker_is_skipped(self, site: Path) -> None:
        """A requirement for another platform is not demanded."""
        from packaging.requirements import Requirement

        _install(site, "ga", requires=('gz; sys_platform == "nonexistent"',))
        assert set(gate.resolve_closure([Requirement("ga")])) == {"ga"}

    def test_missing_distribution_is_an_error(self, site: Path) -> None:
        """A required but uninstalled distribution stops the gate (exit 2)."""
        from packaging.requirements import Requirement

        _install(site, "ga", requires=("g-missing-dist",))
        with pytest.raises(gate.GateError, match="g-missing-dist"):
            gate.resolve_closure([Requirement("ga")])


class TestReviewAndExitCodes:
    """End to end through main(): pass, warn, fail, and configuration errors."""

    @staticmethod
    def _project(tmp_path: Path) -> Path:
        path = tmp_path / "pyproject.toml"
        path.write_text(
            '[project]\nname = "demo-pkg"\nversion = "1"\ndependencies = ["ga"]\n'
            '[project.optional-dependencies]\nx = ["gp"]\n',
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _review(tmp_path: Path, body: str) -> Path:
        path = tmp_path / "review.toml"
        path.write_text(body, encoding="utf-8")
        return path

    def _main(self, tmp_path: Path, extras: str, review: Path) -> int:
        return gate.main(
            [
                "--extras",
                extras,
                "--pyproject",
                str(self._project(tmp_path)),
                "--review",
                str(review),
            ]
        )

    def test_clean_closure_passes(
        self, site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """All compatible: exit 0."""
        _install(site, "ga", expression="MIT")
        assert self._main(tmp_path, "", tmp_path / "absent.toml") == 0
        assert "failed 0" in capsys.readouterr().out

    def test_unreviewed_incompatible_fails(
        self, site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An incompatible dependency with no review: exit 1, named."""
        _install(site, "ga", expression="MIT")
        _install(site, "gp", expression="LicenseRef-Vendor-Proprietary")
        assert self._main(tmp_path, "x", tmp_path / "absent.toml") == 1
        assert "FAIL gp 1.0: incompatible" in capsys.readouterr().out

    def test_flagged_review_passes_loudly(
        self, site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A flagged entry passes, and says so on every run."""
        _install(site, "ga", expression="MIT")
        _install(site, "gp", expression="LicenseRef-Vendor-Proprietary")
        review = self._review(
            tmp_path,
            '[[review]]\nname = "g*"\nstatus = "flagged"\n'
            'signatures = ["License-Expression: LicenseRef-Vendor-Proprietary"]\n'
            'reason = "vendor runtime; owner decides"\n',
        )
        assert self._main(tmp_path, "x", review) == 0
        out = capsys.readouterr().out
        assert "WARNING flagged (owner decision) -- g*: gp 1.0" in out
        assert "vendor runtime; owner decides" in out

    def test_verified_review_passes(self, site: Path, tmp_path: Path) -> None:
        """A verified entry covers an unknown license with evidence."""
        _install(site, "ga", expression="MIT")
        _install(site, "gp")  # code, no license metadata
        review = self._review(
            tmp_path,
            '[[review]]\nname = "gp"\nstatus = "verified"\nsignatures = ["none"]\n'
            'evidence = "the upstream LICENSE file is MIT"\n',
        )
        assert self._main(tmp_path, "x", review) == 0

    def test_permitted_review_passes_and_says_so(
        self, site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A section 7 permission covers the dependency, reported without alarm."""
        _install(site, "ga", expression="MIT")
        _install(site, "gp", expression="LicenseRef-Vendor-Proprietary")
        (tmp_path / "GRANT").write_text("Additional permission under section 7 ...", "utf-8")
        review = self._review(
            tmp_path,
            '[[review]]\nname = "gp"\nstatus = "permitted"\npermission = "GRANT"\n'
            'signatures = ["License-Expression: LicenseRef-Vendor-Proprietary"]\n'
            'reason = "vendor GPU runtime"\n',
        )
        assert self._main(tmp_path, "x", review) == 0
        out = capsys.readouterr().out
        assert "permitted by section 7 1, flagged 0, failed 0" in out
        assert "permitted (AGPL section 7, GRANT) -- gp: 1 distribution(s)" in out
        assert "WARNING" not in out

    @pytest.mark.parametrize(
        "grant",
        [None, "A permission under section 6, mistakenly."],
        ids=["missing", "not-section-7"],
    )
    def test_permitted_needs_a_real_section_7_grant(
        self, site: Path, tmp_path: Path, grant: str | None
    ) -> None:
        """A permission file that is absent, or is not a section 7 grant, is refused."""
        _install(site, "ga", expression="MIT")
        if grant is not None:
            (tmp_path / "GRANT").write_text(grant, "utf-8")
        review = self._review(
            tmp_path,
            '[[review]]\nname = "gp"\nstatus = "permitted"\npermission = "GRANT"\n'
            'signatures = ["none"]\nreason = "r"\n',
        )
        assert self._main(tmp_path, "", review) == 2

    def test_permission_file_must_be_inside_the_project(self, site: Path, tmp_path: Path) -> None:
        """A grant outside the project root is refused, even if it says section 7."""
        _install(site, "ga", expression="MIT")
        project = tmp_path / "project"
        project.mkdir()
        (tmp_path / "OUTSIDE").write_text("permission under section 7", "utf-8")
        pyproject = self._project(project)
        review = self._review(
            project,
            '[[review]]\nname = "gp"\nstatus = "permitted"\npermission = "../OUTSIDE"\n'
            'signatures = ["none"]\nreason = "r"\n',
        )
        argv = ["--pyproject", str(pyproject), "--review", str(review)]
        assert gate.main(argv) == 2

    def test_changed_metadata_invalidates_the_review(
        self, site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A review pinned to old metadata no longer covers the dependency."""
        _install(site, "ga", expression="MIT")
        _install(site, "gp", expression="LicenseRef-Vendor-Proprietary-v2")
        review = self._review(
            tmp_path,
            '[[review]]\nname = "gp"\nstatus = "flagged"\n'
            'signatures = ["License-Expression: LicenseRef-Vendor-Proprietary"]\n'
            'reason = "reviewed against v1"\n',
        )
        assert self._main(tmp_path, "x", review) == 1
        assert "no longer matches its metadata" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "body",
        [
            '[[review]]\nname = "gp"\nstatus = "accepted"\nsignatures = ["none"]\nreason = "r"\n',
            '[[review]]\nname = "gp"\nstatus = "flagged"\nsignatures = []\nreason = "r"\n',
            '[[review]]\nname = "gp"\nstatus = "verified"\nsignatures = ["none"]\nreason = "r"\n',
            '[[review]]\nstatus = "flagged"\nsignatures = ["none"]\nreason = "r"\n',
            '[[review]]\nname = "gp"\nstatus = "permitted"\nsignatures = ["none"]\nreason = "r"\n',
        ],
    )
    def test_malformed_review_is_a_configuration_error(
        self, site: Path, tmp_path: Path, body: str
    ) -> None:
        """Unknown status, no signatures, missing evidence, name or permission: exit 2."""
        _install(site, "ga", expression="MIT")
        assert self._main(tmp_path, "", self._review(tmp_path, body)) == 2

    def test_missing_distribution_exits_2(self, site: Path, tmp_path: Path) -> None:
        """The requested extras are not installed: exit 2, not a pass."""
        _install(site, "ga", expression="MIT")
        assert self._main(tmp_path, "x", tmp_path / "absent.toml") == 2


class TestRepositoryReviewFile:
    """The shipped review file is well-formed and every entry carries its case."""

    def test_loads(self) -> None:
        """Parses, with valid statuses and non-empty notes."""
        reviews = gate.load_reviews(gate.DEFAULT_REVIEW)
        assert reviews
        for review in reviews:
            assert review.status in ("verified", "permitted", "flagged")
            assert review.signatures
            assert len(review.note) > 40

    def test_nvidia_libraries_are_permitted_by_the_shipped_exception(self) -> None:
        """The NVIDIA entry cites LICENSE-EXCEPTION, which exists and is a section 7 grant."""
        nvidia = [r for r in gate.load_reviews(gate.DEFAULT_REVIEW) if r.pattern == "nvidia-*"]
        assert len(nvidia) == 1
        assert nvidia[0].status == "permitted"
        assert nvidia[0].permission == "LICENSE-EXCEPTION"
        assert "section 7" in (_REPO_ROOT / "LICENSE-EXCEPTION").read_text(encoding="utf-8")


MIT_TEXT = (
    "MIT License\n\nCopyright (c) 2024 Example\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy "
    'of this software and associated documentation files (the "Software"), to deal '
    "in the Software without restriction...\n"
)


class TestFileOnlyMetadata:
    """PEP 639 file-only metadata (setuptools 77-79): the shipped text decides.

    Regression: setuptools 79.0.1 on the CI runner declares its license only
    via ``License-File`` -- no expression, no License field, no classifier --
    and the gate read it as 'unknown [none] no recognisable license', failing
    the type-checking lane's dependency gate.
    """

    def test_mit_license_file_only_is_compatible(self, site: Path) -> None:
        a = gate.assess(_install(site, "ga", license_text=MIT_TEXT))
        assert a.verdict == "compatible"
        assert a.signature == "License-File text"
        assert "MIT" in a.detail

    def test_non_mit_license_file_only_stays_unknown(self, site: Path) -> None:
        a = gate.assess(_install(site, "gb", license_text="Proprietary. All rights reserved."))
        assert a.verdict == "unknown"
