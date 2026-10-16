# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Monitoring config stays loadable and only queries series the code emits."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROM_DIR = REPO_ROOT / "monitoring" / "prometheus"


def _load_script() -> Any:
    path = REPO_ROOT / "scripts" / "render_prometheus_rules.py"
    spec = importlib.util.spec_from_file_location("render_prometheus_rules", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render_rules = _load_script()


def _emitted_series() -> set[str]:
    """Metric names core/metrics.py registers (histograms also expose _bucket/_count/_sum)."""
    source = (REPO_ROOT / "src" / "omni_mercury_engine" / "core" / "metrics.py").read_text()
    names: set[str] = set()
    for kind, name in re.findall(r'_create_(counter|histogram|gauge)\(\s*"([a-z0-9_]+)"', source):
        names.add(name)
        if kind == "histogram":
            names.update(f"{name}{suffix}" for suffix in ("_bucket", "_count", "_sum"))
    return names


def _queried_app_series(text: str) -> set[str]:
    """``omni_*`` / ``http_*`` series named in PromQL, minus recording-rule names."""
    text = re.sub(r"[a-z_]+:[a-z0-9_:]+", " ", text)
    return set(re.findall(r"\b((?:omni|http)_[a-z0-9_]+)\b", text))


class TestRenderedRuleFile:
    def test_committed_rule_file_is_current(self) -> None:
        assert render_rules.main(["--check"]) == 0

    def test_rendered_file_is_plain_groups_matching_the_crd(self) -> None:
        crd = yaml.safe_load((PROM_DIR / "prometheus-rules.yaml").read_text())
        plain = yaml.safe_load((PROM_DIR / "rules.yml").read_text())
        assert list(plain) == ["groups"]
        assert plain["groups"] == crd["spec"]["groups"]

    def test_check_detects_drift(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = tmp_path / "prometheus-rules.yaml"
        target = tmp_path / "rules.yml"
        source.write_text((PROM_DIR / "prometheus-rules.yaml").read_text())
        monkeypatch.setattr(render_rules, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(render_rules, "SOURCE", source)
        monkeypatch.setattr(render_rules, "TARGET", target)
        monkeypatch.setattr(render_rules, "HELM_TARGET", tmp_path / "helm" / "prometheusrule.yaml")
        assert render_rules.main(["--check"]) == 1  # missing
        assert render_rules.main([]) == 0
        assert render_rules.main(["--check"]) == 0
        source.write_text(source.read_text().replace("> 0.05", "> 0.06", 1))
        assert render_rules.main(["--check"]) == 1

    def test_crd_without_groups_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"spec\.groups"):
            render_rules.render("apiVersion: v1\nkind: PrometheusRule\nspec: {}\n")

    def test_compose_prometheus_loads_the_rendered_file_and_labels_targets(self) -> None:
        config = yaml.safe_load((PROM_DIR / "prometheus.yml").read_text())
        assert config["rule_files"] == ["rules.yml"]
        api = next(j for j in config["scrape_configs"] if j["job_name"] == "mercury-agent-api")
        labels = api["static_configs"][0]["labels"]
        assert labels == {"app": "mercury-agent", "service": "api"}


class TestHelmRuleTemplate:
    """The chart's PrometheusRule is generated from the CRD, not maintained by hand."""

    HELM = REPO_ROOT / "helm" / "mercury-agent" / "templates"

    @staticmethod
    def _unescape(text: str) -> str:
        """Undo the Helm escaping the way ``helm template`` renders it."""
        text = re.sub(r"\{\{ `(\{\{.*?\}\})` \}\}", r"\1", text)
        return text.replace(
            'namespace="{{ include "mercury-agent.namespace" . }}"', 'namespace="mercury-agent"'
        )

    def test_committed_helm_template_is_current(self) -> None:
        crd = (PROM_DIR / "prometheus-rules.yaml").read_text()
        assert (self.HELM / "prometheusrule.yaml").read_text() == render_rules.render_helm(crd)

    def test_helm_groups_equal_the_crd_groups(self) -> None:
        text = self._unescape((self.HELM / "prometheusrule.yaml").read_text())
        start = text.index("  groups:\n")
        end = text.index("{{- with .Values.metrics.prometheusRule.rules }}")
        groups = yaml.safe_load(text[start:end])["groups"]
        crd = yaml.safe_load((PROM_DIR / "prometheus-rules.yaml").read_text())
        assert groups == crd["spec"]["groups"]

    def test_prometheus_templates_are_escaped_from_helm(self) -> None:
        rendered = render_rules.render_helm(
            "spec:\n  groups:\n  - name: g\n    rules:\n    - alert: A\n"
            "      expr: up == 0\n      annotations:\n"
            "        description: '{{ $value }} on {{ $labels.pod }}'\n"
        )
        assert "{{ `{{ $value }}` }} on {{ `{{ $labels.pod }}` }}" in rendered
        assert "{{ $value }}' " not in rendered

    def test_drift_and_ethical_alerts_reach_helm(self) -> None:
        text = (self.HELM / "prometheusrule.yaml").read_text()
        for alert in (
            "OmniMercuryDataDrift",
            "OmniMercuryDriftBaselineMissing",
            "OmniMercuryBenevolenceViolation",
            "OmniMercurySigmaAlignmentViolation",
        ):
            assert f"alert: {alert}" in text

    def test_servicemonitor_attaches_the_labels_the_rules_select(self) -> None:
        text = (self.HELM / "servicemonitor.yaml").read_text()
        assert "targetLabel: app\n          replacement: mercury-agent" in text
        assert "targetLabel: job\n          replacement: mercury-agent-$1" in text
        assert "__meta_kubernetes_service_label_app_kubernetes_io_component" in text

    def test_api_metrics_port_targets_the_http_listener(self) -> None:
        # The API serves /metrics on its HTTP port; nothing listens on metrics.port.
        service = (self.HELM / "service.yaml").read_text()
        api = service.split("---")[0]
        assert "name: metrics" in api and "targetPort: http" in api


class TestEveryQueriedSeriesIsEmitted:
    def test_alert_and_recording_rules(self) -> None:
        queried = _queried_app_series((PROM_DIR / "prometheus-rules.yaml").read_text())
        assert queried, "rule file references no application series"
        missing = queried - _emitted_series()
        assert not missing, f"rules query series no code emits: {sorted(missing)}"

    def test_hpa_metrics_resolve_through_the_adapter(self) -> None:
        adapter = yaml.safe_load((PROM_DIR / "prometheus-adapter-rules.yaml").read_text())
        served: set[str] = set()
        for rule in adapter["rules"] + adapter["externalRules"]:
            raw = re.search(r"^([a-z0-9_]+)\{", rule["seriesQuery"])
            assert raw is not None
            assert raw.group(1) in _emitted_series(), raw.group(1)
            served.add(rule["name"].get("as") or raw.group(1))

        hpa_metrics: list[tuple[str, dict[str, str]]] = []
        for doc in yaml.safe_load_all((REPO_ROOT / "k8s" / "base" / "hpa.yaml").read_text()):
            if not doc or doc.get("kind") != "HorizontalPodAutoscaler":
                continue
            for metric in doc["spec"]["metrics"]:
                kind = metric["type"]
                if kind in ("Pods", "External"):
                    spec = metric[kind.lower()]["metric"]
                    hpa_metrics.append(
                        (spec["name"], spec.get("selector", {}).get("matchLabels", {}))
                    )
        assert hpa_metrics
        for name, selector in hpa_metrics:
            assert name in served, f"HPA metric {name} has no prometheus-adapter rule"
            for key in selector:
                # Selector keys become PromQL label matchers; a Kubernetes-style
                # key (app.kubernetes.io/name) is not a legal Prometheus label.
                assert re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", key), key

    def test_podmonitors_attach_the_labels_the_rules_select(self) -> None:
        docs = [d for d in yaml.safe_load_all((PROM_DIR / "podmonitors.yaml").read_text()) if d]
        components = set()
        for doc in docs:
            assert doc["kind"] == "PodMonitor"
            components.add(doc["spec"]["selector"]["matchLabels"]["app.kubernetes.io/component"])
            for endpoint in doc["spec"]["podMetricsEndpoints"]:
                attached = {r["targetLabel"]: r["replacement"] for r in endpoint["relabelings"]}
                assert attached["app"] == "mercury-agent"
                assert attached["job"] == f"mercury-agent-{attached['service']}"
        assert components == {"api", "engine"}
