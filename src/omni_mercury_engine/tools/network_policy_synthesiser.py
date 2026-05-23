"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: synthesise a Kubernetes ``NetworkPolicy`` from the
egress allow-list discovered by :mod:`network_egress_recorder` plus
:mod:`loader_reachability_probe`.

The companion :mod:`helm_values_linter` checks for *presence* of a
NetworkPolicy; this tool produces the *concrete rules* — namespace
selectors, host CIDRs, port lists, protocol pins.

Output: a single YAML manifest, written atomically.  Always reviewable
by an operator (read-only string template, no jinja).
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

_SCHEMA = "mercury.tools.network_policy_synthesiser/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.network_policy_synthesiser",
        description=("Synthesise a NetworkPolicy from the operator-supplied egress " "allow-list."),
    )
    parser.add_argument("--allow-list", required=True)
    parser.add_argument("--namespace", default="mercury")
    parser.add_argument("--pod-selector", default="app.kubernetes.io/name=mercury-agent")
    parser.add_argument("--manifest", default=None, help="Output YAML path (atomic).")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return []
    ips: set[str] = set()
    for fam, _, _, _, sockaddr in infos:
        # ``getaddrinfo`` returns ``(host, port)`` for IPv4 and
        # ``(host, port, flowinfo, scopeid)`` for IPv6; the host slot
        # is typed as ``str | int`` in the stdlib stubs because
        # ``AF_UNIX`` packs an int there.  We only care about IPv4/v6
        # results so we coerce + validate explicitly.
        addr = sockaddr[0]
        if not isinstance(addr, str):
            continue
        try:
            ipaddress.ip_address(addr)
        except ValueError:
            continue
        ips.add(addr)
    return sorted(ips)


def _to_rules(prefixes: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    egress_rules: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    for p in prefixes:
        parsed = urlparse(p if "://" in p else f"https://{p}")
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        ips = _resolve(host)
        resolved.append({"prefix": p, "host": host, "port": port, "ips": ips})
        for ip in ips:
            cidr = f"{ip}/{32 if ':' not in ip else 128}"
            egress_rules.append(
                {
                    "to": [{"ipBlock": {"cidr": cidr}}],
                    "ports": [{"protocol": "TCP", "port": port}],
                }
            )
    return egress_rules, resolved


def _render_yaml(namespace: str, selector: str, egress: list[dict[str, Any]]) -> str:
    key, _, value = selector.partition("=")
    lines = [
        "apiVersion: networking.k8s.io/v1",
        "kind: NetworkPolicy",
        "metadata:",
        "  name: mercury-agent-egress",
        f"  namespace: {namespace}",
        "spec:",
        "  podSelector:",
        "    matchLabels:",
        f"      {key}: {value}",
        "  policyTypes:",
        "    - Egress",
        "  egress:",
    ]
    for rule in egress:
        cidr = rule["to"][0]["ipBlock"]["cidr"]
        port = rule["ports"][0]["port"]
        proto = rule["ports"][0]["protocol"]
        lines.append("    - to:")
        lines.append("        - ipBlock:")
        lines.append(f"            cidr: {cidr}")
        lines.append("      ports:")
        lines.append(f"        - protocol: {proto}")
        lines.append(f"          port: {port}")
    return "\n".join(lines) + "\n"


def _collect(args: argparse.Namespace) -> Certificate:
    allow = [
        line.strip()
        for line in Path(args.allow_list).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    egress, resolved = _to_rules(allow)
    manifest = _render_yaml(args.namespace, args.pod_selector, egress)
    if args.manifest and not args.dry_run:
        atomic_write_text(Path(args.manifest), manifest)
    body: dict[str, Any] = {
        "allow_list_path": args.allow_list,
        "allow_count": len(allow),
        "resolved": resolved,
        "egress_rule_count": len(egress),
        "manifest_path": args.manifest,
        "manifest_yaml": manifest,
        "dry_run": bool(args.dry_run),
    }
    unresolved = [r for r in resolved if not r["ips"]]
    status = "warn" if unresolved else "ok"
    warnings = [f"could not resolve {r['host']}" for r in unresolved]
    return Certificate(
        tool="network_policy_synthesiser",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
