#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury Agent API Server Launcher.

Starts the FastAPI server for anomaly detection.

Usage:
    # Development mode (with reload)
    python scripts/run_api.py

    # Production mode
    python scripts/run_api.py --production

    # Custom host/port
    python scripts/run_api.py --host 0.0.0.0 --port 8080

    # With uvicorn directly
    uvicorn omni_mercury_engine.api.server:app --host 0.0.0.0 --port 8000

Environment Variables:
    MERCURY_AGENT_ENV: Set to "production" for production settings
    MERCURY_CORS_ORIGINS: Comma-separated allowed origins
    OMNI_RATE_LIMIT_ENABLED: Enable rate limiting (default: true)
    JWT_SECRET_KEY: Secret key for JWT authentication
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def main():
    parser = argparse.ArgumentParser(
        description="Mercury Agent API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start development server with auto-reload
    python scripts/run_api.py

    # Start production server
    python scripts/run_api.py --production

    # Run with custom settings
    python scripts/run_api.py --host 0.0.0.0 --port 8080 --workers 4

    # Run load tests against the server
    locust -f tests/load/locustfile.py --host http://localhost:8000
        """,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, use more for production)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run in production mode (no reload, optimized settings)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    # Set environment for production
    if args.production:
        os.environ.setdefault("MERCURY_AGENT_ENV", "production")

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)

    # Verify the app can be imported
    try:
        from omni_mercury_engine.api.server import app

        print(f"Mercury Agent API v{app.version}")
    except ImportError as e:
        print(f"ERROR: Could not import API server: {e}")
        print("Make sure all dependencies are installed: pip install -e .[api]")
        sys.exit(1)

    print("\nStarting Mercury Agent API Server")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Mode: {'production' if args.production else 'development'}")
    print(f"  Workers: {args.workers}")
    print(f"\nAPI Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print("-" * 50)

    uvicorn_config = {
        "app": "omni_mercury_engine.api.server:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
    }

    if args.production:
        # Production settings
        uvicorn_config.update(
            {
                "workers": args.workers,
                "access_log": True,
                "proxy_headers": True,
                "forwarded_allow_ips": "*",
            }
        )
    else:
        # Development settings
        uvicorn_config.update(
            {
                "reload": True,
                "reload_dirs": [str(src_path)],
            }
        )

    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
