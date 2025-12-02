"""
OMNI ♱ AVA (O♱A)
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

"""
Command-line interface for OMNI ♱ AVA
"""

import click
import json
import numpy as np
from pathlib import Path

# Lazy import to support CLI help without torch dependency
# OmniAnomalyEngine is only imported when actually needed (not for --help)
OmniAnomalyEngine = None


def _get_engine(*args, **kwargs):
    """Lazy load OmniAnomalyEngine to defer torch import."""
    global OmniAnomalyEngine
    if OmniAnomalyEngine is None:
        try:
            from omni_anomaly_engine.engine import OmniAnomalyEngine as _Engine

            OmniAnomalyEngine = _Engine
        except ImportError as e:
            if "torch" in str(e).lower():
                click.echo(
                    "Error: PyTorch (torch) is required for ML-based detection but is not installed. "
                    "Install it with: pip install torch"
                )
            else:
                click.echo(f"Error: Failed to load ML engine - {e}")
            raise SystemExit(1)
    return OmniAnomalyEngine(*args, **kwargs)


@click.group()
@click.version_option(version="1.0.0")
def main() -> None:
    """OMNI ♱ AVA: ML-Centric Anomaly Detection Framework"""
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


def _load_data(filepath: str) -> np.ndarray:
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


if __name__ == "__main__":
    main()
