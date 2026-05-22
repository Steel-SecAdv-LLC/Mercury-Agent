"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

import os
from typing import Any

"""
Command-line interface for Mercury Agent
"""

import json
from pathlib import Path

import click
import numpy as np

# Lazy import to support CLI help without torch dependency
# OmniMercuryEngine is only imported when actually needed (not for --help)
OmniMercuryEngine: Any = None


def _get_engine(*args: Any, **kwargs: Any) -> Any:
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
@click.version_option(version="1.7.0")
def main() -> None:
    """Mercury Agent: Neuro-Symbolic AI Framework (CLI entry point)."""
    pass


@main.command()
@click.option("--input", "-i", required=True, help="Input data file (CSV/JSON)")
@click.option("--detector", "-d", default="fusion", help="Detector type")
@click.option("--output", "-o", help="Output file for results")
@click.option("--threshold", "-t", default=0.5, type=float, help="Anomaly threshold")
def detect(input: str, detector: str, output: str, threshold: float) -> None:
    """Detect anomalies in data."""
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
    """Biometric face matching."""
    engine = _get_engine()

    result = engine.detect_biometric(reference, test)

    click.echo(json.dumps(result, indent=2, default=str))


@main.command()
@click.option("--payload", "-p", required=True, help="Payload to check for threats")
def security(payload: str) -> None:
    """Security threat detection."""
    engine = _get_engine()

    result = engine.detect_security_threat(payload)

    click.echo(json.dumps(result, indent=2, default=str))


@main.command()
@click.option("--data", "-d", required=True, help="Training data directory")
@click.option("--output", "-o", required=True, help="Model output path")
@click.option("--epochs", "-e", default=50, type=int, help="Training epochs")
def train(data: str, output: str, epochs: int) -> None:
    """Train fusion model."""
    try:
        engine = _get_engine(mode="fusion")

        click.echo(f"Training fusion model on {data}...")
        engine.train_fusion_model(data, epochs=epochs)

        engine.save_model(output)
        click.echo(f"Model saved to {output}")
    except (RuntimeError, ValueError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@main.command()
@click.option("--input", "-i", required=True, help="Input data file")
@click.option("--model", "-m", default="fusion", help="Model type")
def explain(input: str, model: str) -> None:
    """Explain anomaly detection decision."""
    try:
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
    except (KeyError, ValueError, RuntimeError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


def _load_data(filepath: str) -> np.ndarray[Any, Any]:
    """Load data from file."""
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
        return np.asarray(data)  # type: ignore[no-any-return, unused-ignore]

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


# =============================================================================
# Physics-Inspired Anomaly Detection Commands
# =============================================================================


@main.command("verify-corpus")
@click.option(
    "--corpus",
    "-c",
    default=None,
    help="Path to sigma_immutable_corpus.json (defaults to the in-tree copy).",
)
@click.option(
    "--signature",
    "-s",
    default=None,
    help="Path to sigma_immutable_corpus.sig.json (defaults to <corpus>.sig.json).",
)
@click.option(
    "--require-mldsa/--no-require-mldsa",
    default=False,
    help="Fail if the ML-DSA-65 signature cannot be verified (default: allow Ed25519-only).",
)
def verify_corpus(corpus: str | None, signature: str | None, require_mldsa: bool) -> None:
    """Verify the σ_Immutable corpus signature bundle (Ed25519 + ML-DSA-65)."""
    argv: list[str] = []
    if corpus:
        argv += ["--corpus", corpus]
    if signature:
        argv += ["--signature", signature]
    if require_mldsa:
        argv += ["--require-mldsa"]
    from omni_mercury_engine.tools.sigma_immutable_verifier import main as _verifier_main

    raise SystemExit(_verifier_main(argv))


@main.command("tool", context_settings={"ignore_unknown_options": True})
@click.argument("name", required=True)
@click.argument("tool_args", nargs=-1, type=click.UNPROCESSED)
def tool(name: str, tool_args: tuple[str, ...]) -> None:
    """Run an operator tool by name (see `mercury-agent tool list`).

    Examples:

    \b
        mercury-agent tool list
        mercury-agent tool sigma_immutable_verifier
        mercury-agent tool algorithm_name_drift_gate
        mercury-agent tool config_validator --strict
    """
    import importlib
    import pkgutil

    import omni_mercury_engine.tools as _tools_pkg

    available = sorted(
        m.name
        for m in pkgutil.iter_modules(_tools_pkg.__path__)
        if not m.name.startswith("_") and m.name != "migrate_pkl"
    )
    if name == "list":
        for n in available:
            click.echo(n)
        return
    if name not in available:
        click.echo(f"Unknown tool: {name!r}. Run `mercury-agent tool list` for available tools.")
        raise SystemExit(2)
    module = importlib.import_module(f"omni_mercury_engine.tools.{name}")
    if not hasattr(module, "main"):
        click.echo(f"Tool {name!r} has no main() entry point")
        raise SystemExit(2)
    raise SystemExit(module.main(list(tool_args)))


@main.group()
def physics() -> None:
    """Physics-inspired anomaly detection commands."""
    pass


@physics.command("spectral")
@click.option("--input", "-i", required=True, help="Input signal data file (CSV/JSON)")
@click.option("--output", "-o", help="Output file for results")
@click.option("--threshold", "-t", default=0.5, type=float, help="Anomaly threshold")
@click.option(
    "--mode",
    "-m",
    default="comprehensive",
    type=click.Choice(["comprehensive", "fft_only", "wavelet_only", "phonon", "predictive"]),
    help="Analysis mode",
)
@click.option("--sample-rate", "-s", default=1000.0, type=float, help="Signal sample rate (Hz)")
def physics_spectral(
    input: str, output: str, threshold: float, mode: str, sample_rate: float
) -> None:
    """
    Spectral vibration analysis using GNN and CNN.

    Analyzes signals for frequency-domain anomalies using advanced
    physics-inspired techniques including phonon interaction modeling,
    harmonic analysis, and predictive maintenance signatures.

    Examples:
        mercury physics spectral -i vibration_data.csv
        mercury physics spectral -i sensor.json -m phonon -t 0.6
        mercury physics spectral -i machine.csv -m predictive -o results.json
    """
    try:
        from omni_mercury_engine.detectors.spectral_vibration import (
            SpectralAnalysisMode,
            SpectralVibrationDetector,
        )

        # Map mode string to enum
        mode_map = {
            "comprehensive": SpectralAnalysisMode.HYBRID_FUSION,
            "fft_only": SpectralAnalysisMode.FFT_STANDARD,
            "wavelet_only": SpectralAnalysisMode.WAVELET_MULTIRESOLUTION,
            "phonon": SpectralAnalysisMode.PHONON_INTERACTION,
            "predictive": SpectralAnalysisMode.MLIP_VIBRATIONAL,
        }

        config = {
            "threshold": threshold,
            "sample_rate": sample_rate,
            "analysis_mode": mode_map[mode],
        }

        detector = SpectralVibrationDetector(config)
        data = _load_data(input)

        # Fit on the data (self-supervised for single-sample analysis)
        detector.fit(data.flatten() if data.ndim > 1 else data)

        # Detect anomalies
        result = detector.detect(data.flatten() if data.ndim > 1 else data)

        # Format output
        output_data = {
            "detector": "SpectralVibrationDetector",
            "mode": mode,
            "is_anomaly": result.get("is_anomaly", False),
            "anomaly_score": float(result.get("anomaly_score", 0.0)),
            "dominant_frequencies": result.get("dominant_frequencies", []),
            "spectral_entropy": result.get("spectral_entropy", 0.0),
            "harmonic_distortion": result.get("harmonic_distortion", 0.0),
            "vibration_signature": result.get("vibration_signature", "unknown"),
        }

        if output:
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(json.dumps(output_data, indent=2, default=str))

    except ImportError as e:
        click.echo(f"Error: Required dependencies not available - {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error during spectral analysis: {e}")
        raise SystemExit(1)


@physics.command("dynamics")
@click.option("--input", "-i", required=True, help="Input motion/position data file")
@click.option("--output", "-o", help="Output file for results")
@click.option("--threshold", "-t", default=0.5, type=float, help="Anomaly threshold")
@click.option("--time-step", default=0.01, type=float, help="Time step between samples (seconds)")
@click.option("--jerk-sensitivity", default=2.0, type=float, help="Jerk anomaly sensitivity")
@click.option("--chaos-threshold", default=0.1, type=float, help="Chaos detection threshold")
def physics_dynamics(
    input: str,
    output: str,
    threshold: float,
    time_step: float,
    jerk_sensitivity: float,
    chaos_threshold: float,
) -> None:
    """
    Acceleration dynamics analysis with phase space reconstruction.

    Analyzes motion data for kinematic anomalies using velocity, acceleration,
    and jerk analysis. Includes Lyapunov exponent estimation for chaos detection
    and energy conservation monitoring.

    Examples:
        mercury physics dynamics -i motion_data.csv
        mercury physics dynamics -i trajectory.json --time-step 0.001
        mercury physics dynamics -i sensor.csv --chaos-threshold 0.05 -o results.json
    """
    try:
        from omni_mercury_engine.detectors.acceleration_dynamics import (
            AccelerationDynamicsDetector,
        )

        config = {
            "threshold": threshold,
            "time_step": time_step,
            "jerk_sensitivity": jerk_sensitivity,
            "chaos_threshold": chaos_threshold,
        }

        detector = AccelerationDynamicsDetector(config)
        data = _load_data(input)

        # Fit on the data
        signal = data.flatten() if data.ndim > 1 else data
        detector.fit(signal)

        # Detect anomalies
        result = detector.detect(signal)

        # Format output
        output_data = {
            "detector": "AccelerationDynamicsDetector",
            "is_anomaly": result.get("is_anomaly", False),
            "anomaly_score": float(result.get("anomaly_score", 0.0)),
            "motion_state": result.get("motion_state", "unknown"),
            "energy_state": result.get("energy_state", "unknown"),
            "lyapunov_exponent": float(result.get("lyapunov_exponent", 0.0)),
            "is_chaotic": result.get("is_chaotic", False),
            "jerk_anomaly": result.get("jerk_anomaly", False),
            "energy_anomaly": float(result.get("energy_anomaly", 0.0)),
            "kinematic_summary": {
                "mean_velocity": float(result.get("mean_velocity", 0.0)),
                "mean_acceleration": float(result.get("mean_acceleration", 0.0)),
                "max_jerk": float(result.get("max_jerk", 0.0)),
            },
        }

        if output:
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(json.dumps(output_data, indent=2, default=str))

    except ImportError as e:
        click.echo(f"Error: Required dependencies not available - {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error during dynamics analysis: {e}")
        raise SystemExit(1)


@physics.command("uiux")
@click.option("--input", "-i", required=True, help="Input user interaction data file (JSON)")
@click.option("--output", "-o", help="Output file for results")
@click.option("--threshold", "-t", default=0.5, type=float, help="Anomaly threshold")
@click.option(
    "--rage-threshold", default=0.3, type=float, help="Rage click time threshold (seconds)"
)
@click.option("--bot-threshold", default=0.7, type=float, help="Bot detection threshold")
def physics_uiux(
    input: str, output: str, threshold: float, rage_threshold: float, bot_threshold: float
) -> None:
    """
    UI/UX behavioral anomaly detection.

    Analyzes user interaction patterns for anomalies including rage clicks,
    dead clicks, erratic scrolling, navigation loops, and bot-like behavior.

    Input format (JSON array of interactions):
        [{"timestamp": 0.0, "type": "click", "x": 100, "y": 200, ...}, ...]

    Examples:
        mercury physics uiux -i interactions.json
        mercury physics uiux -i session.json --rage-threshold 0.2 --bot-threshold 0.8
        mercury physics uiux -i user_data.json -o analysis.json
    """
    try:
        from omni_mercury_engine.detectors.uiux_anomaly import (
            InteractionType,
            UIUXAnomalyDetector,
            UserInteraction,
        )

        config = {
            "threshold": threshold,
            "rage_click_threshold": rage_threshold,
            "bot_detection_threshold": bot_threshold,
        }

        detector = UIUXAnomalyDetector(config)

        # Load interaction data
        with open(input) as f:
            raw_data = json.load(f)

        # Convert to UserInteraction objects
        type_map = {
            "click": InteractionType.CLICK,
            "scroll": InteractionType.SCROLL,
            "mousemove": InteractionType.MOUSE_MOVE,
            "keypress": InteractionType.KEY_PRESS,
            "page_view": InteractionType.PAGE_VIEW,
            "form_submit": InteractionType.FORM_SUBMIT,
            "hover": InteractionType.HOVER,
        }

        interactions = []
        for item in raw_data:
            int_type = type_map.get(item.get("type", "click").lower(), InteractionType.CLICK)
            interaction = UserInteraction(
                timestamp=float(item.get("timestamp", 0)),
                interaction_type=int_type,
                x=item.get("x", 0),
                y=item.get("y", 0),
                element_id=item.get("element_id"),
                element_type=item.get("element_type"),
                page_url=item.get("page_url"),
                scroll_delta=item.get("scroll_delta", 0),
                viewport_width=item.get("viewport_width", 1920),
                viewport_height=item.get("viewport_height", 1080),
            )
            interactions.append(interaction)

        if len(interactions) < 5:
            click.echo("Error: Need at least 5 interactions for analysis")
            raise SystemExit(1)

        # Fit and detect
        detector.fit(interactions)
        result = detector.detect(interactions)

        # Format output
        click_analysis = result.get("click_analysis")
        scroll_analysis = result.get("scroll_analysis")
        nav_analysis = result.get("navigation_analysis")
        session_analysis = result.get("session_analysis")

        output_data = {
            "detector": "UIUXAnomalyDetector",
            "is_anomaly": result.get("is_anomaly", False),
            "anomaly_score": float(result.get("anomaly_score", 0.0)),
            "behavior_class": result.get("behavior_class", "unknown"),
            "bot_probability": float(result.get("bot_probability", 0.0)),
            "anomaly_categories": result.get("anomaly_categories", []),
            "click_analysis": {
                "total_clicks": click_analysis.total_clicks if click_analysis else 0,
                "rage_clicks": click_analysis.rage_clicks if click_analysis else 0,
                "dead_clicks": click_analysis.dead_clicks if click_analysis else 0,
                "click_accuracy": float(click_analysis.click_accuracy) if click_analysis else 0.0,
            },
            "scroll_analysis": {
                "total_scrolls": scroll_analysis.total_scrolls if scroll_analysis else 0,
                "rapid_scrolls": scroll_analysis.rapid_scrolls if scroll_analysis else 0,
                "scroll_reversals": scroll_analysis.scroll_reversals if scroll_analysis else 0,
            },
            "navigation_analysis": {
                "unique_pages": nav_analysis.unique_pages if nav_analysis else 0,
                "navigation_loops": nav_analysis.navigation_loops if nav_analysis else 0,
                "back_button_usage": nav_analysis.back_button_usage if nav_analysis else 0,
            },
            "session_analysis": {
                "session_duration": (
                    float(session_analysis.session_duration) if session_analysis else 0.0
                ),
                "total_interactions": (
                    session_analysis.total_interactions if session_analysis else 0
                ),
                "engagement_score": (
                    float(session_analysis.engagement_score) if session_analysis else 0.0
                ),
            },
        }

        if output:
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(json.dumps(output_data, indent=2, default=str))

    except ImportError as e:
        click.echo(f"Error: Required dependencies not available - {e}")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON in input file - {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error during UI/UX analysis: {e}")
        raise SystemExit(1)


@physics.command("integrated")
@click.option("--spectral-input", "-s", help="Spectral/vibration data file")
@click.option("--dynamics-input", "-d", help="Dynamics/motion data file")
@click.option("--uiux-input", "-u", help="UI/UX interaction data file (JSON)")
@click.option("--output", "-o", help="Output file for results")
@click.option("--threshold", "-t", default=0.5, type=float, help="Anomaly threshold")
@click.option(
    "--fusion-weights", "-w", default="0.4,0.3,0.3", help="Fusion weights (spectral,dynamics,uiux)"
)
def physics_integrated(
    spectral_input: str,
    dynamics_input: str,
    uiux_input: str,
    output: str,
    threshold: float,
    fusion_weights: str,
) -> None:
    """
    Integrated physics-inspired anomaly detection using all modules.

    Combines spectral vibration, acceleration dynamics, and UI/UX analysis
    with golden-ratio weighted fusion aligned with GOSNN ethical governance.

    Examples:
        mercury physics integrated -s vibration.csv -d motion.csv
        mercury physics integrated -s sensor.csv -u interactions.json -o report.json
        mercury physics integrated -s data.csv -d data.csv -u session.json -w 0.5,0.3,0.2
    """
    if not any([spectral_input, dynamics_input, uiux_input]):
        click.echo("Error: At least one input file required (-s, -d, or -u)")
        raise SystemExit(1)

    try:
        from omni_mercury_engine.detectors.advanced_physics_integration import (
            AdvancedPhysicsIntegratedDetector,
            PhysicsDetectorType,
        )
        from omni_mercury_engine.detectors.uiux_anomaly import (
            InteractionType,
            UserInteraction,
        )

        # Parse fusion weights
        weights = [float(w) for w in fusion_weights.split(",")]
        if len(weights) != 3:
            weights = [0.4, 0.3, 0.3]

        # Determine which detectors to enable
        enabled_detectors = []
        if spectral_input:
            enabled_detectors.append(PhysicsDetectorType.SPECTRAL_VIBRATION)
        if dynamics_input:
            enabled_detectors.append(PhysicsDetectorType.ACCELERATION_DYNAMICS)
        if uiux_input:
            enabled_detectors.append(PhysicsDetectorType.UIUX_ANOMALY)

        config = {
            "threshold": threshold,
            "enabled_detectors": enabled_detectors,
            "fusion_weights": {
                "spectral": weights[0],
                "dynamics": weights[1],
                "uiux": weights[2],
            },
        }

        detector = AdvancedPhysicsIntegratedDetector(config)

        # Load data for each input type
        spectral_data = None
        dynamics_data = None
        uiux_data = None

        if spectral_input:
            data = _load_data(spectral_input)
            spectral_data = data.flatten() if data.ndim > 1 else data

        if dynamics_input:
            data = _load_data(dynamics_input)
            dynamics_data = data.flatten() if data.ndim > 1 else data

        if uiux_input:
            with open(uiux_input) as f:
                raw_data = json.load(f)

            type_map = {
                "click": InteractionType.CLICK,
                "scroll": InteractionType.SCROLL,
                "mousemove": InteractionType.MOUSE_MOVE,
                "keypress": InteractionType.KEY_PRESS,
                "page_view": InteractionType.PAGE_VIEW,
            }

            uiux_data = []
            for item in raw_data:
                int_type = type_map.get(item.get("type", "click").lower(), InteractionType.CLICK)
                interaction = UserInteraction(
                    timestamp=float(item.get("timestamp", 0)),
                    interaction_type=int_type,
                    x=item.get("x", 0),
                    y=item.get("y", 0),
                    element_id=item.get("element_id"),
                    page_url=item.get("page_url"),
                    scroll_delta=item.get("scroll_delta", 0),
                )
                uiux_data.append(interaction)

        # Fit the detector
        fit_data = {
            "spectral": spectral_data,
            "dynamics": dynamics_data,
            "uiux": uiux_data,
        }
        detector.fit(fit_data)

        # Detect anomalies
        detect_data = {
            "spectral": spectral_data,
            "dynamics": dynamics_data,
            "uiux": uiux_data,
        }
        result = detector.detect(detect_data)

        # Format output
        output_data = {
            "detector": "AdvancedPhysicsIntegratedDetector",
            "is_anomaly": result.get("is_anomaly", False),
            "fused_anomaly_score": float(result.get("fused_anomaly_score", 0.0)),
            "enabled_detectors": [d.value for d in enabled_detectors],
            "fusion_weights": config["fusion_weights"],
            "detector_results": {},
            "gosnn_governance": result.get("gosnn_governance", {}),
            "three_r_alignment": result.get("three_r_alignment", {}),
        }

        # Add individual detector results
        if "spectral_result" in result:
            output_data["detector_results"]["spectral"] = {
                "anomaly_score": float(result["spectral_result"].get("anomaly_score", 0.0)),
                "is_anomaly": result["spectral_result"].get("is_anomaly", False),
            }
        if "dynamics_result" in result:
            output_data["detector_results"]["dynamics"] = {
                "anomaly_score": float(result["dynamics_result"].get("anomaly_score", 0.0)),
                "is_anomaly": result["dynamics_result"].get("is_anomaly", False),
                "motion_state": result["dynamics_result"].get("motion_state", "unknown"),
            }
        if "uiux_result" in result:
            output_data["detector_results"]["uiux"] = {
                "anomaly_score": float(result["uiux_result"].get("anomaly_score", 0.0)),
                "is_anomaly": result["uiux_result"].get("is_anomaly", False),
                "behavior_class": result["uiux_result"].get("behavior_class", "unknown"),
            }

        if output:
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(json.dumps(output_data, indent=2, default=str))

    except ImportError as e:
        click.echo(f"Error: Required dependencies not available - {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error during integrated analysis: {e}")
        raise SystemExit(1)


@physics.command("list")
def physics_list() -> None:
    """List available physics-inspired detectors and their capabilities."""
    click.echo("\n" + "=" * 65)
    click.echo("  Mercury Agent - Physics-Inspired Anomaly Detectors")
    click.echo("=" * 65)

    click.echo("\n  1. SpectralVibrationDetector")
    click.echo("     ─────────────────────────────────────────────────────")
    click.echo("     Frequency-domain anomaly detection using advanced")
    click.echo("     physics-inspired techniques.")
    click.echo()
    click.echo("     Features:")
    click.echo("       • GNN-based spectral graph analysis")
    click.echo("       • CNN spectral pattern recognition")
    click.echo("       • Phonon interaction modeling (quantum-inspired)")
    click.echo("       • MLIP energy potential encoding")
    click.echo("       • Wavelet multi-scale decomposition")
    click.echo("       • Predictive maintenance signatures")
    click.echo()
    click.echo("     Command: mercury physics spectral -i <file>")

    click.echo("\n  2. AccelerationDynamicsDetector")
    click.echo("     ─────────────────────────────────────────────────────")
    click.echo("     Kinematic analysis with phase space reconstruction")
    click.echo("     for motion and trajectory anomalies.")
    click.echo()
    click.echo("     Features:")
    click.echo("       • Velocity, acceleration, jerk analysis")
    click.echo("       • Phase space trajectory reconstruction")
    click.echo("       • Lyapunov exponent for chaos detection")
    click.echo("       • Energy conservation monitoring")
    click.echo("       • Motion state classification")
    click.echo("       • Impulse and momentum tracking")
    click.echo()
    click.echo("     Command: mercury physics dynamics -i <file>")

    click.echo("\n  3. UIUXAnomalyDetector")
    click.echo("     ─────────────────────────────────────────────────────")
    click.echo("     User interaction pattern analysis for behavioral")
    click.echo("     anomaly detection and bot identification.")
    click.echo()
    click.echo("     Features:")
    click.echo("       • Rage click detection (Fitts's Law)")
    click.echo("       • Dead click identification")
    click.echo("       • Erratic scrolling patterns")
    click.echo("       • Navigation loop detection")
    click.echo("       • Bot behavior classification")
    click.echo("       • Session engagement scoring")
    click.echo()
    click.echo("     Command: mercury physics uiux -i <file.json>")

    click.echo("\n  4. AdvancedPhysicsIntegratedDetector")
    click.echo("     ─────────────────────────────────────────────────────")
    click.echo("     Unified detector combining all physics modules with")
    click.echo("     3R mechanism and GOSNN ethical governance.")
    click.echo()
    click.echo("     Features:")
    click.echo("       • Golden-ratio weighted fusion (φ = 1.618...)")
    click.echo("       • 3R alignment (Recursion-Resonance-Refactoring)")
    click.echo("       • GOSNN ethical scalar governance")
    click.echo("       • Cross-domain correlation analysis")
    click.echo("       • Adaptive threshold calibration")
    click.echo()
    click.echo("     Command: mercury physics integrated -s <spectral> -d <dynamics> -u <uiux>")

    click.echo("\n" + "=" * 65 + "\n")


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
    click.echo("  Mercury Agent - Interactive Voice Interface")
    click.echo("=" * 60)
    click.echo()

    # Declare voice_instance with Any type to handle both MercuryVoice and _FallbackVoice
    voice_instance: Any
    llm_chain: Any = None

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
