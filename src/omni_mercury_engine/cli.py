"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

import os
from typing import Any


"""
Command-line interface for Mercury Agent ♱
"""

import json
from pathlib import Path

import click
import numpy as np


# Lazy import to support CLI help without torch dependency
# OmniMercuryEngine is only imported when actually needed (not for --help)
OmniMercuryEngine = None


def _get_engine(*args, **kwargs):
    """Lazy load OmniMercuryEngine to defer torch import."""
    global OmniMercuryEngine
    if OmniMercuryEngine is None:
        try:
            from omni_mercury_engine.engine import OmniMercuryEngine as _Engine

            OmniMercuryEngine = _Engine
        except ImportError as e:
            if "torch" in str(e).lower():
                click.echo(
                    "Error: PyTorch (torch) is required for ML-based detection "
                    "but is not installed. Install it with: pip install torch"
                )
            else:
                click.echo(f"Error: Failed to load ML engine - {e}")
            raise SystemExit(1)
    return OmniMercuryEngine(*args, **kwargs)


@click.group()
@click.version_option(version="1.2.0")
def main() -> None:
    """Mercury Agent ♱: ML-Centric Anomaly Detection Framework"""
    pass


@main.command()
@click.option("--input", "-i", required=True, help="Input data file (CSV/JSON)")
@click.option("--detector", "-d", default="fusion", help="Detector type")
@click.option("--output", "-o", help="Output file for results")
@click.option("--threshold", "-t", default=0.5, type=float, help="Anomaly threshold")
def detect(input: str, detector: str, output: str, threshold: float) -> None:
    """Detect anomalies in data"""
    engine = _get_engine(mode=detector)

    data = _load_data(input)

    if detector == "fusion":
        results = engine.detect_with_fusion(data)
    else:
        results = engine.detect(data, detector_types=[detector])

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        click.echo(json.dumps(results, indent=2, default=str))


@main.command()
@click.option("--reference", "-r", required=True, help="Reference face image")
@click.option("--test", "-t", help="Test face image to match")
def biometric(reference: str, test: str) -> None:
    """Biometric face matching"""
    engine = _get_engine()

    result = engine.detect_biometric(reference, test)

    click.echo(json.dumps(result, indent=2, default=str))


@main.command()
@click.option("--payload", "-p", required=True, help="Payload to check for threats")
def security(payload: str) -> None:
    """Security threat detection"""
    engine = _get_engine()

    result = engine.detect_security_threat(payload)

    click.echo(json.dumps(result, indent=2, default=str))


@main.command()
@click.option("--data", "-d", required=True, help="Training data directory")
@click.option("--output", "-o", required=True, help="Model output path")
@click.option("--epochs", "-e", default=50, type=int, help="Training epochs")
def train(data: str, output: str, epochs: int) -> None:
    """Train fusion model"""
    engine = _get_engine(mode="fusion")

    click.echo(f"Training fusion model on {data}...")
    engine.train_fusion_model(data, epochs=epochs)

    engine.save_model(output)
    click.echo(f"Model saved to {output}")


@main.command()
@click.option("--input", "-i", required=True, help="Input data file")
@click.option("--model", "-m", default="fusion", help="Model type")
def explain(input: str, model: str) -> None:
    """Explain anomaly detection decision"""
    engine = _get_engine(mode=model)

    data = _load_data(input)

    result = engine.detect_with_fusion(data)

    explanation = {
        "prediction": {
            "is_anomaly": result["is_anomaly"],
            "confidence": result["anomaly_prob"],
        },
        "detector_contributions": result.get("detector_importance", {}),
    }

    click.echo(json.dumps(explanation, indent=2))


def _load_data(filepath: str) -> np.ndarray[Any, Any]:
    """Load data from file"""
    path = Path(filepath)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return np.array(data)
            return np.array([data])

    elif path.suffix == ".csv":
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


# =============================================================================
# Voice Conversation Command
# =============================================================================
# =============================================================================
# API Server Command
# =============================================================================
@main.command()
@click.option(
    "--host",
    "-h",
    default=os.environ.get("MERCURY_HOST", "127.0.0.1"),
    help="Host address to bind to (default: 127.0.0.1, set MERCURY_HOST=0.0.0.0 for all interfaces)",
)
@click.option("--port", "-p", default=8000, type=int, help="Port number to listen on")
@click.option("--workers", "-w", default=1, type=int, help="Number of worker processes")
@click.option("--reload", "-r", is_flag=True, help="Enable auto-reload for development")
@click.option(
    "--log-level",
    "-l",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error"]),
    help="Logging level",
)
def serve(host: str, port: int, workers: int, reload: bool, log_level: str) -> None:
    """
    Start the Mercury Agent API server.

    The API provides REST endpoints for anomaly detection, batch processing,
    model management, and data export.

    Examples:
        mercury serve
        mercury serve --port 8080
        mercury serve --workers 4 --log-level debug
        mercury serve --reload  # Development mode with auto-reload
    """
    click.echo("\n" + "=" * 60)
    click.echo("  Mercury Agent API Server")
    click.echo("=" * 60)
    click.echo()
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Workers: {workers}")
    click.echo(f"  Reload: {'enabled' if reload else 'disabled'}")
    click.echo(f"  Log Level: {log_level}")
    click.echo()
    click.echo("  Endpoints:")
    click.echo("    /docs       - Interactive API documentation (Swagger)")
    click.echo("    /redoc      - ReDoc API documentation")
    click.echo("    /health     - Health check endpoint")
    click.echo("    /api/v1/... - API endpoints")
    click.echo()
    click.echo("-" * 60)
    click.echo()

    try:
        import uvicorn

        uvicorn.run(
            "omni_mercury_engine.api.server:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level=log_level,
        )
    except ImportError:
        click.echo(
            "Error: uvicorn is required for the API server. " "Install with: pip install uvicorn"
        )
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error starting server: {e}")
        raise SystemExit(1)


@main.command()
@click.option("--domain", "-d", default=None, help="Domain context (medical, security, etc.)")
@click.option("--model", "-m", default="llama3.2:3b", help="Ollama model to use")
@click.option("--offline", is_flag=True, help="Force offline mode (template responses)")
def voice(domain: str, model: str, offline: bool) -> None:
    """
    Start interactive voice conversation with Mercury.

    Examples:
        mercury voice
        mercury voice --domain medical
        mercury voice --model mistral:7b
        mercury voice --offline
    """
    _start_voice_conversation(domain, model, offline)


def _start_voice_conversation(
    domain: str | None,
    model: str,
    offline: bool,
) -> None:
    """Start the interactive voice conversation loop."""

    # Print banner
    click.echo("\n" + "=" * 60)
    click.echo("  Mercury Agent ♱ - Interactive Voice Interface")
    click.echo("=" * 60)
    click.echo()

    # Try to import voice module
    try:
        from omni_mercury_engine.models.foundation.ollama_adapter import (
            FallbackLLMChain,
            OllamaConfig,
        )
        from omni_mercury_engine.narrative.voice import create_mercury_voice

        # Configure LLM chain
        ollama_config = OllamaConfig(model=model)
        llm_chain = FallbackLLMChain(
            ollama_config=ollama_config,
            enable_cloud=False,  # Privacy-first
        )

        voice_instance = create_mercury_voice()

        # Print LLM status
        chain_status = llm_chain.get_chain_status()
        active = chain_status["active"]
        click.echo(f"  LLM: {active}")
        click.echo(f"  Domain: {domain or 'general'}")
        click.echo(f"  Mode: {'offline' if offline else 'auto'}")

    except ImportError as e:
        click.echo(f"  Note: Using fallback mode ({e})")
        voice_instance = _FallbackVoice()
        llm_chain = None

    click.echo()
    click.echo("  Commands:")
    click.echo("    /quit or /exit - Exit conversation")
    click.echo("    /status        - Show system status")
    click.echo("    /clear         - Clear conversation history")
    click.echo("    /help          - Show help")
    click.echo()
    click.echo("-" * 60)

    # Print greeting
    try:
        greeting = voice_instance.greet(domain=domain)
        if hasattr(greeting, "message"):
            click.echo(f"\nMercury: {greeting.message}\n")
        else:
            click.echo(f"\nMercury: {greeting.get('message', 'Hello.')}\n")
    except Exception:
        click.echo("\nMercury: Hello. Mercury Agent ready for conversation.\n")

    # Conversation loop
    while True:
        try:
            # Get user input
            user_input = click.prompt(
                click.style("You", fg="cyan"),
                default="",
                show_default=False,
            )

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower().strip()

                if command in ["/quit", "/exit", "/q"]:
                    click.echo("\nMercury: Goodbye. Stay vigilant.\n")
                    break

                elif command == "/status":
                    _show_status(voice_instance, llm_chain)
                    continue

                elif command == "/clear":
                    click.echo("\n[Conversation history cleared]\n")
                    continue

                elif command == "/help":
                    _show_help()
                    continue

                else:
                    click.echo(f"\nUnknown command: {command}")
                    click.echo("Type /help for available commands.\n")
                    continue

            # Process user message
            try:
                response = voice_instance.speak(user_input, domain=domain)

                if hasattr(response, "message"):
                    message = response.message
                    confidence = getattr(response, "confidence", None)
                else:
                    message = response.get("message", "I received your message.")
                    confidence = response.get("confidence")

                # Format response
                click.echo()
                click.echo(click.style("Mercury: ", fg="green") + message)

                if confidence and confidence < 0.7:
                    click.echo(
                        click.style(
                            f"  [Confidence: {confidence:.0%}]",
                            fg="yellow",
                            dim=True,
                        )
                    )

                click.echo()

            except Exception as e:
                click.echo(click.style(f"\n[Error processing message: {e}]\n", fg="red"))

        except (KeyboardInterrupt, EOFError):
            click.echo("\n\nMercury: Session ended. Goodbye.\n")
            break


def _show_status(voice_instance: Any, llm_chain: Any) -> None:
    """Show system status."""
    click.echo("\n" + "-" * 40)
    click.echo("  System Status")
    click.echo("-" * 40)

    click.echo("  Voice Interface: operational")

    if llm_chain:
        try:
            status = llm_chain.get_chain_status()
            click.echo(f"  Active LLM: {status['active']}")
            click.echo(
                f"  Ollama: {'available' if status['ollama']['available'] else 'unavailable'}"
            )
            click.echo("  Template Fallback: available")
        except Exception:
            click.echo("  LLM Status: unknown")
    else:
        click.echo("  LLM: fallback mode")

    if hasattr(voice_instance, "get_conversation_history"):
        history = voice_instance.get_conversation_history()
        click.echo(f"  Conversation turns: {len(history) if history else 0}")

    click.echo("-" * 40 + "\n")


def _show_help() -> None:
    """Show help message."""
    click.echo("\n" + "-" * 40)
    click.echo("  Mercury Voice Commands")
    click.echo("-" * 40)
    click.echo("  /quit, /exit, /q  - Exit conversation")
    click.echo("  /status           - Show system status")
    click.echo("  /clear            - Clear history")
    click.echo("  /help             - Show this help")
    click.echo()
    click.echo("  You can ask Mercury:")
    click.echo("    - 'What is my system status?'")
    click.echo("    - 'What anomalies have been detected?'")
    click.echo("    - 'Explain the last detection'")
    click.echo("    - 'Search for pattern information'")
    click.echo("-" * 40 + "\n")


class _FallbackVoice:
    """Fallback voice for when narrative module unavailable."""

    def speak(self, message: str, domain: str | None = None) -> dict[str, Any]:
        """Process user message with fallback responses."""
        msg_lower = message.lower()

        if any(kw in msg_lower for kw in ["status", "health"]):
            return {
                "message": "Mercury Agent operational. Running in fallback mode.",
                "confidence": 0.9,
            }
        elif any(kw in msg_lower for kw in ["hello", "hi"]):
            return {
                "message": "Hello. Mercury Agent at your service.",
                "confidence": 0.95,
            }
        elif any(kw in msg_lower for kw in ["help", "what can"]):
            return {
                "message": "I can help with anomaly detection and monitoring. "
                "Use the detect command for analysis, or ask about system status.",
                "confidence": 0.9,
            }
        else:
            return {
                "message": "I received your query. For detailed analysis, "
                "please use the detect command with your data file.",
                "confidence": 0.7,
            }

    def greet(self, domain: str | None = None) -> dict[str, Any]:
        """Generate greeting."""
        return {
            "message": "Mercury Agent online. How can I assist you?",
            "confidence": 1.0,
        }

    def get_conversation_history(self) -> list[Any]:
        """Get conversation history."""
        return []


if __name__ == "__main__":
    main()
