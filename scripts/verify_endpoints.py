#!/usr/bin/env python3
"""
Mercury Agent ♱ — Endpoint Verification Script

Performs an audit of every URL in TrustedEndpoints:
  1. HTTP HEAD/GET each URL with a 10-second timeout.
  2. Log HTTP status, content-type, first 256 bytes.
  3. Flag non-2xx or unexpected content types.

Output: JSON report to stdout (or file via --output).

Usage:
    python scripts/verify_endpoints.py
    python scripts/verify_endpoints.py --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


def get_all_endpoints() -> dict[str, str]:
    """Extract all endpoint URLs from TrustedEndpoints class."""
    from omni_mercury_engine.security.input_validation import TrustedEndpoints

    endpoints: dict[str, str] = {}
    for attr_name in dir(TrustedEndpoints):
        if attr_name.startswith("_"):
            continue
        if attr_name == "TRUSTED_DOMAINS":
            continue
        val = getattr(TrustedEndpoints, attr_name)
        if isinstance(val, str) and val.startswith("https://"):
            endpoints[attr_name] = val
    return endpoints


def check_endpoint(name: str, url: str, timeout: int = 10) -> dict[str, Any]:
    """Check a single endpoint and return status info."""
    result: dict[str, Any] = {
        "name": name,
        "url": url,
        "status": None,
        "content_type": None,
        "first_bytes": None,
        "validated": False,
        "error": None,
        "response_time_ms": None,
    }

    start = time.monotonic()
    try:
        req = urllib.request.Request(
            url,
            method="GET",
            headers={"User-Agent": "Mercury-Agent/1.0 EndpointVerifier"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            result["status"] = resp.status
            result["content_type"] = resp.headers.get("Content-Type", "")
            body = resp.read(256)
            # Try to decode as text, fall back to hex repr
            try:
                result["first_bytes"] = body.decode("utf-8", errors="replace")[:256]
            except Exception:
                result["first_bytes"] = body.hex()[:256]

            # Validate: 2xx status and not an HTML error page
            is_2xx = 200 <= result["status"] < 300
            ct = (result["content_type"] or "").lower()
            is_html_error = "text/html" in ct and "<html" in (result["first_bytes"] or "").lower()
            result["validated"] = is_2xx and not is_html_error

    except urllib.error.HTTPError as e:
        result["status"] = e.code
        result["error"] = str(e)
        result["validated"] = False
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["validated"] = False

    elapsed = time.monotonic() - start
    result["response_time_ms"] = round(elapsed * 1000, 1)
    return result


def main() -> int:
    """Run endpoint verification and output JSON report."""
    parser = argparse.ArgumentParser(description="Verify Mercury-Agent TrustedEndpoints")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout per request in seconds")
    args = parser.parse_args()

    endpoints = get_all_endpoints()
    print(f"Checking {len(endpoints)} endpoints...", file=sys.stderr)

    results: list[dict[str, Any]] = []
    failed = 0
    for name, url in sorted(endpoints.items()):
        print(f"  {name}: {url[:80]}...", file=sys.stderr, end=" ")
        result = check_endpoint(name, url, timeout=args.timeout)
        results.append(result)
        status = "OK" if result["validated"] else "FAIL"
        if not result["validated"]:
            failed += 1
        print(f"[{status}] {result.get('status', 'N/A')}", file=sys.stderr)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_endpoints": len(endpoints),
        "validated": len(endpoints) - failed,
        "failed": failed,
        "results": results,
    }

    output = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nReport written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Exit with non-zero if any endpoints failed
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
