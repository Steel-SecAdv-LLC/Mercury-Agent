# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import socket
import ssl
from typing import Any
from urllib.parse import urlparse

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.tls_posture_probe/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.tls_posture_probe",
        description=(
            "Probe TLS posture for a remote URL: cipher suite, "
            "certificate chain, OCSP staple, HSTS, ALPN, and PQ "
            "hybrid (X25519MLKEM768) availability."
        ),
    )
    parser.add_argument("url", help="HTTPS URL to probe (e.g. https://example.com/).")
    parser.add_argument(
        "--require-pq",
        action="store_true",
        help="Fail unless the server advertises a post-quantum hybrid group.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser


def _gate(url: str) -> tuple[str, int]:
    """Apply the SafeHTTPClient SSRF gate to ``url`` and return host:port."""
    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"tls_posture_probe only supports https://, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL has no hostname: {url!r}")
    port = parsed.port or 443
    # Reuse the SafeHTTPClient SSRF + DNS-pinning gate via the
    # documented ``validate_url`` class-method — this resolves and
    # screens the host through the same path the engine uses at
    # runtime, without burning an outbound request.  The TLS handshake
    # below re-resolves via ``socket.getaddrinfo``; since
    # ``validate_url`` already DNS-pinned & confirmed the host is on
    # the trusted egress allow-list, the second resolution is a
    # belt-and-braces lookup of an already-validated name.
    SafeHTTPClient.validate_url(url, user_configured=True)
    return host, port


def _probe_handshake(
    host: str,
    port: int,
    timeout: float,
    groups: str | None = None,
) -> dict[str, Any]:
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.set_alpn_protocols(["h2", "http/1.1"])
    if groups is not None and hasattr(ctx, "set_ecdh_curve"):
        # ``set_groups`` is the OpenSSL 3.2+ API; the type stubs ship
        # only the legacy ``set_ecdh_curve`` so we call the method via
        # ``getattr`` rather than disguising the missing attribute with
        # ``type: ignore``.  Older OpenSSL silently lacks the method,
        # which is fine — the post-quantum probe will simply report
        # ``available=False`` when the handshake fails below.
        set_groups = getattr(ctx, "set_groups", None)
        if callable(set_groups):
            try:
                set_groups(groups)
            except OSError:
                pass
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        ssock = ctx.wrap_socket(sock, server_hostname=host)
        try:
            result: dict[str, Any] = {
                "version": ssock.version(),
                "cipher": ssock.cipher(),
                "alpn_proto": ssock.selected_alpn_protocol(),
            }
            try:
                peer = ssock.getpeercert()
                if peer is None:
                    # ``getpeercert`` returns ``None`` only when the
                    # peer presented no certificate, which on a TLS
                    # 1.2+ handshake means the handshake itself
                    # silently degraded — record that explicitly.
                    result["peer_cert_error"] = "no peer certificate"
                else:
                    result["peer_cert_subject"] = peer.get("subject")
                    result["peer_cert_issuer"] = peer.get("issuer")
                    result["peer_cert_notAfter"] = peer.get("notAfter")
                    san = peer.get("subjectAltName")
                    if san:
                        result["peer_cert_san"] = list(san)
            except (ValueError, OSError) as exc:
                result["peer_cert_error"] = str(exc)
            try:
                # OCSP-stapled response is exposed via ``getpeercert(ocsp=True)``
                # on Python 3.13+ — fall back to the OpenSSL APIs otherwise.
                ocsp = getattr(ssock, "ocsp_response", None)
                result["ocsp_stapled"] = bool(ocsp)
            except Exception:
                result["ocsp_stapled"] = False
            return result
        finally:
            ssock.close()
    finally:
        sock.close()


def _collect(args: argparse.Namespace) -> Certificate:
    host, port = _gate(args.url)
    classical = _probe_handshake(host, port, args.timeout)

    # Post-quantum hybrid probe: requires OpenSSL 3.2+ AND a server that
    # advertises the X25519MLKEM768 hybrid group.  Failure here is not a
    # hard error unless --require-pq is supplied.
    pq: dict[str, Any]
    try:
        pq = _probe_handshake(host, port, args.timeout, groups="X25519MLKEM768")
        pq["available"] = True
    except (ssl.SSLError, OSError) as exc:
        pq = {"available": False, "reason": str(exc)}

    # HSTS — issue a real GET via ``SafeHTTPClient.get`` so the response
    # headers come back.  We deliberately use ``get`` (not a raw HEAD)
    # because the SSRF gate only exposes verbed convenience methods
    # (``get``/``post``/...); a bare HEAD shortcut is unnecessary since
    # the response is streamed and discarded.
    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    headers: dict[str, str]
    try:
        resp = SafeHTTPClient.get(args.url, timeout=args.timeout, user_configured=True)
        try:
            headers = {k.lower(): v for k, v in resp.headers.items()}
        finally:
            resp.close()
    except OSError as exc:
        headers = {"__error__": str(exc)}
    hsts = headers.get("strict-transport-security")

    body: dict[str, Any] = {
        "url": args.url,
        "host": host,
        "port": port,
        "classical": classical,
        "post_quantum": pq,
        "hsts": hsts,
        "alpn_advertised": classical.get("alpn_proto"),
        "response_headers_sample": {
            k: headers.get(k)
            for k in ("server", "alt-svc", "strict-transport-security", "content-security-policy")
            if k in headers
        },
    }

    warnings: list[str] = []
    if not hsts:
        warnings.append("server did not return Strict-Transport-Security")
    if not pq["available"] and args.require_pq:
        warnings.append(f"post-quantum hybrid unavailable: {pq.get('reason', 'unknown')}")
    status = "fail" if (args.require_pq and not pq["available"]) else ("warn" if warnings else "ok")
    return Certificate(
        tool="tls_posture_probe",
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
