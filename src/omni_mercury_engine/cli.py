# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line interface for Mercury Agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
import numpy as np

from omni_mercury_engine._version import __version__ as _DISTRIBUTION_VERSION

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
@click.version_option(version=_DISTRIBUTION_VERSION)
def main() -> None:
    """Mercury Agent: Neuro-Symbolic AI Framework (CLI entry point)."""
    pass


@main.command()
@click.option("--input", "-i", required=True, help="Input data file (CSV/JSON)")
@click.option(
    "--detector",
    "-d",
    default="statistical",
    help=(
        "Detector type. Defaults to 'statistical' (the trained Mercury "
        "ensemble). Use 'fusion' for the neural fusion path."
    ),
)
@click.option("--output", "-o", help="Output file for results")
@click.option(
    "--threshold",
    "-t",
    default=None,
    type=float,
    help=(
        "Anomaly decision threshold in [0, 1]. Overrides the fusion model's "
        "adaptive threshold: is_anomaly becomes (anomaly_prob >= threshold). "
        "Only the fusion path emits a probability; on other detectors it is "
        "reported as inapplicable. Omit to use the model's own threshold."
    ),
)
def detect(input: str, detector: str, output: str, threshold: float | None) -> None:
    """Detect anomalies in data."""
    if threshold is not None and not (0.0 <= threshold <= 1.0):
        raise click.BadParameter("--threshold must be in [0, 1]", param_hint="--threshold")
    # require_explicit_fit=False: this command has no train/test split to fit
    # the base detectors on (it scores whatever single file the caller gives
    # it), so it relies on detect_with_fusion's legacy auto-fit-on-first-batch
    # path -- load_default_fusion_checkpoint() below restores the trained
    # *fusion network* but never touches base-detector fit state. Without this
    # override `detect -d fusion` would raise RuntimeError on every run (the 5
    # base detectors are always constructed unfitted; see _init_detectors).
    engine = _get_engine(mode=detector, require_explicit_fit=False)

    data = _load_data(input)

    if detector == "fusion":
        # Load the shipped default fusion checkpoint so detection runs with a
        # trained, calibrated network out of the box (no training step needed).
        if engine.load_default_fusion_checkpoint():
            # Notice on stderr so stdout stays valid JSON for piping.
            click.echo("Loaded shipped default fusion checkpoint.", err=True)
        else:
            click.echo(
                "No default fusion checkpoint found; using untrained fusion "
                "(run `mercury-agent train` or scripts/train_default_fusion.py).",
                err=True,
            )
        # Deployment posture (mirrors /detect/flagship and the MCP server):
        # served fusion detections close the loop with a ``decision`` record.
        # Additive key in the JSON output; the core engine default stays
        # opt-in.
        engine.enable_decision_layer()
        results = engine.detect_with_fusion(data)
        if threshold is not None and "anomaly_prob" in results:
            # Honour the operator's explicit decision boundary over the model's
            # adaptive threshold (previously the flag was accepted and ignored).
            # NOTE: the ``decision`` record is computed against the engine's
            # own calibrated threshold, not this override -- is_anomaly may
            # therefore differ from the record's disposition by design.
            results["is_anomaly"] = bool(float(results["anomaly_prob"]) >= threshold)
            results["threshold_used"] = float(threshold)
            results["threshold_source"] = "cli_override"
    else:
        results = engine.detect(data, detector_types=[detector])
        if threshold is not None:
            # The statistical path returns an aggregate decision with no single
            # probability to threshold; say so rather than silently ignore.
            click.echo(
                f"--threshold={threshold} is not applied to detector "
                f"'{detector}' (no probability output); use '-d fusion' to "
                "apply a decision threshold.",
                err=True,
            )

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        click.echo(json.dumps(results, indent=2, default=str))


@main.command("tier-detect")
@click.option(
    "--input", "-i", required=True, help="Input data file (.csv/.npy/.json); a 1-D series"
)
@click.option(
    "--labels",
    "-l",
    default=None,
    help="Optional 0/1 labels file (enables supervised stacking / BMA)",
)
@click.option(
    "--method",
    "-m",
    default=None,
    type=click.Choice(["stacking", "bma", "average", "consensus"]),
    help="Ensemble combiner (default: stacking with labels, else average)",
)
@click.option(
    "--subset",
    default=None,
    help="Comma-separated detector names (default: the full streaming tier)",
)
@click.option(
    "--contamination", "-c", default=0.05, type=float, help="Expected anomaly fraction [0,1]"
)
@click.option(
    "--conformal-alpha",
    default=None,
    type=float,
    help="Distribution-free false-positive rate (e.g. 0.05); adds conformal flags",
)
@click.option(
    "--attribution",
    is_flag=True,
    default=False,
    help="Also emit the calibrated per-detector score matrix (which detectors fired)",
)
@click.option(
    "--counterfactual",
    is_flag=True,
    default=False,
    help=(
        "Also emit a verified minimal counterfactual for one point: the "
        "replacement value that flips its decision, re-scored through the "
        "same fitted ensemble"
    ),
)
@click.option(
    "--cf-index",
    default=None,
    type=int,
    help="Point to explain (default: highest-scoring flagged point)",
)
@click.option(
    "--cf-method",
    default="prototype",
    type=click.Choice(["wachter", "dice", "growing_spheres", "prototype", "genetic"]),
    help="Counterfactual search method",
)
@click.option("--output", "-o", default=None, help="Output JSON file (stdout if omitted)")
def tier_detect(
    input: str,
    labels: str | None,
    method: str | None,
    subset: str | None,
    contamination: float,
    conformal_alpha: float | None,
    attribution: bool,
    counterfactual: bool,
    cf_index: int | None,
    cf_method: str,
    output: str | None,
) -> None:
    """Run the streaming detector-tier ensemble on a 1-D series (torch-free).

    Exposes the statistical / state-space / streaming detector tier's calibrated
    ensemble — per-point anomaly probabilities, flags, cross-detector
    uncertainty, optional distribution-free (conformal) false-positive control,
    and optional per-detector attribution — without requiring the [ml] extra.
    """
    from omni_mercury_engine.detectors.detection_tier import run_tier_ensemble

    series = np.asarray(_load_data(input), dtype=float).ravel()
    y = _load_labels(labels) if labels else None
    subset_names = tuple(s.strip() for s in subset.split(",")) if subset else None

    result = run_tier_ensemble(
        series,
        labels=y,
        subset=subset_names,
        method=method,
        contamination=contamination,
        conformal_alpha=conformal_alpha,
        include_attribution=attribution,
        include_counterfactual=counterfactual,
        counterfactual_index=cf_index,
        counterfactual_method=cf_method,
    )

    text = json.dumps(result, indent=2, default=str)
    if output:
        Path(output).write_text(text)
        click.echo(f"Wrote tier detection result to {output}", err=True)
    else:
        click.echo(text)


@main.command("rca")
@click.option(
    "--input",
    "-i",
    required=True,
    help="Observations file (.csv/.npy/.json); rows x nodes, last row is the anomaly",
)
@click.option(
    "--train",
    default=None,
    help="Optional normal-behaviour rows for the per-node baselines (default: --input)",
)
@click.option(
    "--adjacency",
    default=None,
    help="Optional (n_nodes x n_nodes) causal adjacency (default: inferred from correlations)",
)
@click.option("--top-k", "-k", default=None, type=int, help="Return only the top-K root causes")
@click.option("--node-names", default=None, help="Comma-separated node labels (one per column)")
@click.option("--output", "-o", default=None, help="Output JSON file (stdout if omitted)")
def rca(
    input: str,
    train: str | None,
    adjacency: str | None,
    top_k: int | None,
    node_names: str | None,
    output: str | None,
) -> None:
    """Attribute a multivariate anomaly to its root-cause nodes (torch-free).

    Runs the tier's graph-based root-cause analysis: a reverse personalised
    random walk over a causal / service adjacency ranks which node (sensor,
    service, channel) most likely originated the anomaly in the final row.
    """
    from omni_mercury_engine.detectors.detection_tier import localize_root_cause

    observations = np.asarray(_load_data(input), dtype=float)
    train_rows = np.asarray(_load_data(train), dtype=float) if train else None
    adj = np.asarray(_load_data(adjacency), dtype=float) if adjacency else None
    names = [s.strip() for s in node_names.split(",")] if node_names else None

    result = localize_root_cause(
        observations, adjacency=adj, train=train_rows, top_k=top_k, node_names=names
    )

    text = json.dumps(result, indent=2, default=str)
    if output:
        Path(output).write_text(text)
        click.echo(f"Wrote root-cause analysis to {output}", err=True)
    else:
        click.echo(text)


_HAZARD_CHOICES = (
    "earthquake",
    "tsunami",
    "meteor",
    "wildfire",
    "tornado",
    "hurricane",
    "volcanic",
    "landslide",
    "schumann",
)


@main.command("hazard-viz")
@click.option(
    "--input",
    "-i",
    "input_path",
    default=None,
    help="Diagnostics payload file (.npz from HazardDiagnostics.to_npz, or its .json form)",
)
@click.option(
    "--detector",
    "-d",
    "hazard",
    default=None,
    type=click.Choice(_HAZARD_CHOICES),
    help="Run this hazard detector on --data (instead of loading a prior payload)",
)
@click.option(
    "--data",
    "data_path",
    default=None,
    help=(
        "Detector input file (.csv/.npy/.json single array, or .npz with named arrays "
        "such as wind_u/wind_v)"
    ),
)
@click.option(
    "--format",
    "-f",
    "fmt",
    default="png",
    type=click.Choice(["png", "geojson"]),
    help="Artifact format",
)
@click.option("--output", "-o", required=True, help="Output artifact path")
@click.option(
    "--geotransform",
    default=None,
    help=(
        "JSON file with origin_lon/origin_lat/deg_per_pixel_lon/deg_per_pixel_lat "
        "(required for --format geojson; pixel coordinates are never presented as lat/lon)"
    ),
)
@click.option("--sampling-rate", default=None, type=float, help="Sampling rate in Hz")
@click.option("--pixel-size-km", default=None, type=float, help="Wildfire ground resolution")
@click.option("--grid-spacing-m", default=None, type=float, help="Hurricane wind-grid spacing")
def hazard_viz(
    input_path: str | None,
    hazard: str | None,
    data_path: str | None,
    fmt: str,
    output: str,
    geotransform: str | None,
    sampling_rate: float | None,
    pixel_size_km: float | None,
    grid_spacing_m: float | None,
) -> None:
    """Render a hazard detector's persisted diagnostics to PNG or GeoJSON.

    Renders the REAL intermediate arrays the hazard detectors compute
    (spectrograms, STA/LTA series, Doppler velocity fields, thermal hotspot
    masks, wind/vorticity fields, harmonic spectra) -- either from a prior
    diagnostics payload (--input) or by running a detector on raw input
    (--detector + --data). GeoJSON output (wildfire ignition hotspots) needs a
    --geotransform: the detectors work in pixel space and coordinates are
    never fabricated.
    """
    from omni_mercury_engine.detectors.hazard_diagnostics import (
        HazardDiagnostics,
        run_hazard_detector,
    )
    from omni_mercury_engine.detectors.hazard_visuals import (
        build_hazard_geojson,
        render_hazard_png,
    )

    if (input_path is None) == (hazard is None):
        raise click.UsageError("provide exactly one of --input (payload) or --detector (run)")

    if input_path is not None:
        path = Path(input_path)
        if path.suffix == ".npz":
            diagnostics = HazardDiagnostics.from_npz(path)
        elif path.suffix == ".json":
            with open(path) as fh:
                diagnostics = HazardDiagnostics.from_jsonable(json.load(fh))
        else:
            raise click.UsageError(
                f"unsupported diagnostics file format {path.suffix!r} (use .npz or .json)"
            )
    else:
        if data_path is None:
            raise click.UsageError("--detector requires --data with the detector input")
        assert hazard is not None
        arrays = _load_hazard_arrays(data_path, hazard)
        params: dict[str, Any] = {}
        if sampling_rate is not None:
            params["sampling_rate_hz"] = sampling_rate
        if pixel_size_km is not None:
            params["pixel_size_km"] = pixel_size_km
        if grid_spacing_m is not None:
            params["grid_spacing_m"] = grid_spacing_m
        try:
            diagnostics = run_hazard_detector(hazard, arrays, params=params)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    out = Path(output)
    if fmt == "geojson":
        gt: dict[str, float] | None = None
        if geotransform is not None:
            with open(geotransform) as fh:
                gt = json.load(fh)
        try:
            feature_collection = build_hazard_geojson(diagnostics, geotransform=gt)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        out.write_text(json.dumps(feature_collection, indent=2))
        click.echo(
            f"Wrote GeoJSON ({len(feature_collection['features'])} features) to {out}", err=True
        )
    else:
        try:
            png = render_hazard_png(diagnostics)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        out.write_bytes(png)
        click.echo(f"Wrote {diagnostics.hazard} PNG ({len(png)} bytes) to {out}", err=True)


def _load_hazard_arrays(data_path: str, hazard: str) -> dict[str, Any]:
    """Load detector input arrays for ``hazard-viz --detector`` runs.

    ``.npz`` archives contribute every named member (e.g. ``wind_u``/``wind_v``).
    Single-array files (.csv/.npy/.json) map to the hazard's primary input
    array name. 1-D series stay 1-D (no reshape).

    Args:
        data_path: Input file path.
        hazard: The hazard the arrays feed (selects the primary array name).

    Returns:
        Named arrays for :func:`run_hazard_detector`.
    """
    from omni_mercury_engine.detectors.hazard_diagnostics import PRIMARY_INPUT_ARRAY

    path = Path(data_path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name]) for name in archive.files}
    if path.suffix == ".npy":
        return {PRIMARY_INPUT_ARRAY[hazard]: np.load(path, allow_pickle=False)}
    if path.suffix == ".csv":
        return {PRIMARY_INPUT_ARRAY[hazard]: np.loadtxt(path, delimiter=",")}
    if path.suffix == ".json":
        with open(path) as fh:
            return {PRIMARY_INPUT_ARRAY[hazard]: np.asarray(json.load(fh), dtype=float)}
    raise click.UsageError(
        f"unsupported data file format {path.suffix!r} (use .npz, .npy, .csv, or .json)"
    )


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
@click.option(
    "--data",
    "-d",
    required=True,
    help=(
        "Training data file. Raw samples as .csv or .npy "
        "(shape [n_samples, n_features]) are fed to fit_fusion, which "
        "extracts the full inference feature pipeline internally. A pre-built "
        "feature .npz archive (per-detector/-model arrays + a 'labels' array) "
        "is also accepted and routed through the feature-archive trainer."
    ),
)
@click.option(
    "--labels",
    "-l",
    default=None,
    help=(
        "Optional labels file (.csv/.npy/.json, 1=anomaly/0=normal) for "
        "supervised training of raw .csv/.npy data. If omitted, "
        "semi-supervised consensus labels are derived from detector agreement."
    ),
)
@click.option("--output", "-o", required=True, help="Model checkpoint output path (.pt)")
@click.option("--epochs", "-e", default=50, type=int, help="Training epochs")
def train(data: str, labels: str | None, output: str, epochs: int) -> None:
    r"""Train the fusion model.

    Two input modes are supported:

    \b
    * Raw features (.csv / .npy): the engine fits all base detectors, extracts
      the full fusion feature pipeline, and trains via the
      supervised/semi-supervised ``fit_fusion`` path. Provide ``--labels`` for
      supervised training, or omit it for detector-consensus labels.
    * Pre-extracted feature archive (.npz): trained directly via
      ``train_fusion_model`` (see ``mercury-agent build-features``).
    """
    try:
        engine = _get_engine(mode="fusion")

        if data.endswith(".npz"):
            click.echo(f"Training fusion model on feature archive {data}...")
            metrics = engine.train_fusion_model(data, epochs=epochs)
        else:
            X = _load_data(data)
            y = _load_labels(labels) if labels else None
            if y is not None and len(y) != len(X):
                raise ValueError(f"Label count ({len(y)}) does not match sample count ({len(X)}).")
            mode_str = "supervised" if y is not None else "semi-supervised (consensus labels)"
            click.echo(f"Training fusion model on {len(X)} samples from {data} [{mode_str}]...")
            metrics = engine.fit_fusion(X, y, epochs=epochs)

        click.echo(f"Best validation loss: {metrics.get('best_loss', float('nan')):.4f}")
        engine.save_model(output)
        click.echo(f"Model saved to {output}")
    except (RuntimeError, ValueError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@main.command("tune")
@click.option(
    "--data",
    "-d",
    required=True,
    help="Raw input samples as .csv or .npy (shape [n_samples, n_features]).",
)
@click.option(
    "--labels",
    "-l",
    required=True,
    help="Labels file (.csv/.npy/.json, 1=anomaly/0=normal). Required: tuning "
    "scores each trial by held-out AUC, which needs both classes.",
)
@click.option("--output", "-o", required=True, help="Output path for the tuned model")
@click.option("--n-trials", default=20, type=int, help="Hyperparameter configs to try")
@click.option(
    "--sampler",
    default="tpe",
    type=click.Choice(["tpe", "gp", "random"]),
    help="Bayesian sampler (default: tpe).",
)
@click.option("--epochs", default=10, type=int, help="Epochs per trial (and final refit)")
def tune(data: str, labels: str, output: str, n_trials: int, sampler: str, epochs: int) -> None:
    """Bayesian hyperparameter search for the fusion model (maximise held-out AUC).

    Runs Mercury's own Bayesian optimizer over the ``fit_fusion`` hyperparameters
    (learning rate, batch size, focal-loss params, early-stopping patience,
    symbolic weight), scoring each configuration by the ROC-AUC of the calibrated
    fusion probability on a held-out split. The engine is refit on the full
    dataset with the best configuration and saved to ``--output``.
    """
    try:
        engine = _get_engine(mode="fusion")
        X = _load_data(data)
        y = _load_labels(labels)
        if len(y) != len(X):
            raise ValueError(f"Label count ({len(y)}) does not match sample count ({len(X)}).")
        click.echo(
            f"Tuning fusion hyperparameters: {n_trials} trials ({sampler}) on "
            f"{len(X)} samples from {data}..."
        )
        # tune_fusion refits on the winning config and raises RuntimeError if
        # every trial failed (no model to save), which the handler below turns
        # into a clean non-zero exit -- so reaching here means a real tuned model.
        result = engine.tune_fusion(X, y, n_trials=n_trials, sampler=sampler, tuning_epochs=epochs)
        click.echo(f"Best held-out AUC: {result['best_auc']:.4f}")
        for key, value in result.get("best_config", {}).items():
            click.echo(f"  {key}: {value}")
        engine.save_model(output)
        click.echo(f"Tuned model saved to {output}")
    except (RuntimeError, ValueError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@main.command("build-features")
@click.option(
    "--data",
    "-d",
    required=True,
    help="Raw input samples as .csv or .npy (shape [n_samples, n_features]).",
)
@click.option(
    "--labels",
    "-l",
    default=None,
    help="Optional labels file (.csv/.npy/.json, 1=anomaly/0=normal); omit for "
    "semi-supervised consensus labels.",
)
@click.option("--output", "-o", required=True, help="Output feature archive path (.npz)")
def build_features(data: str, labels: str | None, output: str) -> None:
    """Build a reusable fusion-training feature archive (.npz) from raw data.

    Delegates to ``engine.build_feature_npz``, which runs the *same* feature
    extraction that ``fit_fusion`` performs on raw input, so the archive is
    byte-for-byte the feature set the trainer would have used. Caching it to an
    ``.npz`` lets repeated training runs skip the (expensive) extraction step.
    The archive is consumable by ``mercury-agent train --data <archive>.npz``.
    """
    try:
        if not output.endswith(".npz"):
            raise ValueError(f"Output must be a .npz archive (got {output!r}).")

        engine = _get_engine(mode="fusion")
        X = _load_data(data)
        y = _load_labels(labels) if labels else None
        if y is not None and len(y) != len(X):
            raise ValueError(f"Label count ({len(y)}) does not match sample count ({len(X)}).")
        path = engine.build_feature_npz(X, output, y=y)
        click.echo(f"Feature archive written to {path}")
    except (RuntimeError, ValueError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@main.command()
@click.option("--input", "-i", required=True, help="Input data file (CSV/NPY)")
@click.option("--output", "-o", help="Output file for the causal graph (JSON)")
@click.option("--names", "-n", default=None, help="Comma-separated variable names")
@click.option("--temporal", is_flag=True, help="Use Granger temporal causation instead of PC")
@click.option("--seed", default=0, type=int, help="Seed for reproducible discovery")
@click.option("--significance", default=0.05, type=float, help="Independence-test alpha")
def causal(
    input: str,
    output: str | None,
    names: str | None,
    temporal: bool,
    seed: int,
    significance: float,
) -> None:
    """Discover causal structure (PC + Fisher-Z, or Granger temporal).

    Surfaces the causal-discovery subsystem. Structure discovery is
    deterministic for a fixed input and ``--seed``.

    Examples:
        mercury-agent causal -i data.csv --names A,B,C,D
        mercury-agent causal -i series.csv --temporal --seed 0 -o graph.json
    """
    try:
        engine = _get_engine(mode="fusion")
        data = _load_data(input)
        var_names = [s.strip() for s in names.split(",")] if names else None

        if temporal:
            graph = engine.discover_temporal_causation(
                data, var_names, significance_level=significance, seed=seed
            )
        else:
            graph = engine.discover_causal_structure(
                data, var_names, significance_level=significance, seed=seed
            )

        if output:
            with open(output, "w") as f:
                json.dump(graph, f, indent=2, default=str)
            click.echo(f"Causal graph saved to {output}")
        else:
            click.echo(json.dumps(graph, indent=2, default=str))
    except (RuntimeError, ValueError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@main.command("symbolic-rules")
@click.option("--output", "-o", help="Output file for the rule graph (JSON)")
def symbolic_rules(output: str | None) -> None:
    """Export the symbolic logic layer's rule graph (nodes, edges, rules)."""
    try:
        engine = _get_engine(mode="fusion")
        graph = engine.symbolic_rule_graph()
        if output:
            with open(output, "w") as f:
                json.dump(graph, f, indent=2, default=str)
            click.echo(f"Rule graph saved to {output}")
        else:
            click.echo(json.dumps(graph, indent=2, default=str))
    except (RuntimeError, ValueError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@main.command()
@click.option("--input", "-i", required=True, help="Input data file")
@click.option("--model", "-m", default="fusion", help="Model type")
def explain(input: str, model: str) -> None:
    """Explain anomaly detection decision."""
    try:
        # See the `detect` command above: require_explicit_fit=False is
        # needed because this command has no train/test split to fit the
        # base detectors on, only an inference file to explain.
        engine = _get_engine(mode=model, require_explicit_fit=False)
        # Use the same trained checkpoint the fusion detect path loads so
        # explanations reflect the shipped model, not random-init weights.
        if model == "fusion":
            engine.load_default_fusion_checkpoint()

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

    elif path.suffix == ".npy":
        data = np.load(path, allow_pickle=False)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return np.asarray(data, dtype=np.float32)

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _load_labels(filepath: str) -> np.ndarray[Any, Any]:
    """Load a 1-D label vector from CSV, NPY, or JSON."""
    path = Path(filepath)
    if path.suffix == ".npy":
        y = np.load(path, allow_pickle=False)
    elif path.suffix == ".csv":
        y = np.loadtxt(path, delimiter=",")
    elif path.suffix == ".json":
        with open(path) as f:
            y = np.array(json.load(f))
    else:
        raise ValueError(f"Unsupported label file format: {path.suffix} (use .csv, .npy, or .json)")
    return np.asarray(y).reshape(-1)


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
    default=True,
    help="Deprecated compatibility flag; ML-DSA-65 verification is always required.",
)
def verify_corpus(corpus: str | None, signature: str | None, require_mldsa: bool) -> None:
    """Verify the σ_Immutable corpus signature bundle (Ed25519 + ML-DSA-65)."""
    _ = require_mldsa
    # The underlying ``sigma_immutable_verifier`` argparse names are
    # ``--corpus-path`` / ``--sig-path``. The user-facing CLI surface keeps
    # the more intuitive names and translates here.
    argv: list[str] = []
    if corpus:
        argv += ["--corpus-path", corpus]
    if signature:
        argv += ["--sig-path", signature]
    from omni_mercury_engine.tools.sigma_immutable_verifier import main as _verifier_main

    raise SystemExit(_verifier_main(argv))


@main.command("tool", context_settings={"ignore_unknown_options": True})
@click.argument("name", required=True)
@click.argument("tool_args", nargs=-1, type=click.UNPROCESSED)
def tool(name: str, tool_args: tuple[str, ...]) -> None:
    r"""Run an operator tool by name (see `mercury-agent tool list`).

    Examples:
        \b
        mercury-agent tool list
        mercury-agent tool sigma_immutable_verifier
        mercury-agent tool algorithm_name_drift_gate
        mercury-agent tool config_validator --strict
    """
    from omni_mercury_engine.tools import TOOL_REGISTRY

    if name == "list":
        for n in TOOL_REGISTRY.names():
            click.echo(n)
        return
    if name not in TOOL_REGISTRY:
        click.echo(f"Unknown tool: {name!r}. Run `mercury-agent tool list` for available tools.")
        raise SystemExit(2)
    raise SystemExit(TOOL_REGISTRY[name](list(tool_args)))


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
    """Spectral vibration analysis using GNN and CNN.

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
    """Acceleration dynamics analysis with phase space reconstruction.

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
    """UI/UX behavioral anomaly detection.

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
                "pages_visited": nav_analysis.pages_visited if nav_analysis else 0,
                "navigation_loops": nav_analysis.navigation_loops if nav_analysis else 0,
                "backtrack_rate": float(nav_analysis.backtrack_rate) if nav_analysis else 0.0,
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
    """Integrated physics-inspired anomaly detection using all modules.

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
    """Start the Mercury Agent API server.

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
def mcp() -> None:
    """Run Mercury as an MCP server on stdio (the universal interconnect).

    Exposes Mercury's capabilities -- anomaly detection, ethics scoring, web
    research, document generation, and confidence calibration -- as Model Context
    Protocol tools, so ANY MCP client (desktop assistants, IDE agents, etc.) can
    discover and run them. Speaks newline-delimited JSON-RPC 2.0 on stdin/stdout;
    standard-library only, no extra dependency.

    Point an MCP client at this command, e.g. in a client config:

        {"command": "mercury-agent", "args": ["mcp"]}
    """
    from omni_mercury_engine.mcp_server import MercuryMCPServer

    MercuryMCPServer().serve_stdio()


@main.group()
def intel() -> None:
    """Mercury intelligence layer: verifier, provenance, feedback loop, red-team, value board.

    Operator surface for the learning + decision-geometry streams -- the same
    controls the engine wires into its live request/emission path, exposed here
    so every stream is runnable and selectable (no unselectable options).
    """


@intel.command("verify")
@click.argument("text")
@click.option(
    "--mode",
    type=click.Choice(["hard", "soft"]),
    default=None,
    help="Override MERCURY_VERIFIER_MODE (hard=block, soft=flag).",
)
def intel_verify(text: str, mode: str | None) -> None:
    """Route the checkable claims in TEXT through the oracle verifiers.

    Exits non-zero when a claim is oracle-refuted and the mode is ``hard`` (the
    emission would be blocked on the live path).
    """
    from omni_mercury_engine.intel.verifier_loop import VerifierLoop, VerifierMode

    loop = VerifierLoop(mode=VerifierMode(mode)) if mode else VerifierLoop()
    decision = loop.guard_emission(text, source="cli")
    click.echo(json.dumps(decision.as_dict(), indent=2))
    if not decision.allowed:
        raise SystemExit(2)


@intel.command("provenance")
@click.argument("text")
@click.option(
    "--source", "-s", multiple=True, help="A source/citation carried with TEXT (repeatable)."
)
@click.option("--verified", is_flag=True, help="The sources were independently checked.")
def intel_provenance(text: str, source: tuple[str, ...], verified: bool) -> None:
    """Enforce provenance at the output boundary for TEXT given its --source citations.

    Uses Mercury's own weapons/mass-casualty gate to decide when attribution is
    required. Exits non-zero when the boundary withholds the emission.
    """
    from omni_mercury_engine.intel.provenance import (
        Provenance,
        ProvenanceOrigin,
        enforce_at_boundary,
    )

    prov = Provenance(origin=ProvenanceOrigin.EXTRACTIVE, sources=tuple(source), verified=verified)
    decision = enforce_at_boundary(text, text=text, provenance=prov, source="cli")
    click.echo(json.dumps(decision.as_dict(), indent=2))
    if not decision.emitted:
        raise SystemExit(2)


@intel.command("self-consistency")
@click.argument("answers", nargs=-1, required=True)
@click.option(
    "--prob", type=float, default=None, help="Base calibrated probability for the decision rule."
)
def intel_self_consistency(answers: tuple[str, ...], prob: float | None) -> None:
    """Score sampled ANSWERS for disagreement and (with --prob) apply the decision rule."""
    from collections import Counter

    from omni_mercury_engine.intel.self_consistency import (
        self_consistency_decision,
        vote_disagreement,
    )

    disagreement = vote_disagreement(list(answers))
    top, top_count = Counter(answers).most_common(1)[0]
    out: dict[str, Any] = {
        "n_samples": len(answers),
        "plurality_answer": top,
        "disagreement": round(float(disagreement), 6),
        "agreement": round(1.0 - float(disagreement), 6),
        "plurality_vote_fraction": round(top_count / len(answers), 6),
    }
    if prob is not None:
        dec = self_consistency_decision(prob, disagreement)
        out["decision"] = {
            "decision": dec.decision,
            "widened_prob": round(float(dec.widened_prob), 6),
            "abstained": dec.abstained,
        }
    click.echo(json.dumps(out, indent=2))


@intel.command("value-board")
@click.option("--json", "as_json", is_flag=True, help="Emit the raw JSON board.")
def intel_value_board(as_json: bool) -> None:
    """Print the intelligence-layer value board (each stream's baseline/target)."""
    from omni_mercury_engine.intel.value_metrics import VALUE_METRICS

    if as_json:
        click.echo(json.dumps({k: v.as_dict() for k, v in VALUE_METRICS.items()}, indent=2))
        return
    for key, metric in VALUE_METRICS.items():
        arrow = "^" if metric.direction.value == "higher_is_better" else "v"
        click.echo(
            f"{key:26s} {metric.metric:34s} "
            f"baseline={metric.baseline:<7g} target={metric.target:<7g} {arrow} {metric.unit}"
        )


@intel.command("audit-log")
@click.option(
    "--path", default=None, help="Gate-audit JSONL path (default: the gate's resolved sink)."
)
@click.option("--limit", default=20, type=int, help="Return only the last N labelable events.")
@click.option(
    "--decisions",
    default=None,
    help="Comma-separated decision filter, e.g. 'refuse_redact,hard_refuse,escalate'.",
)
def intel_audit_log(path: str | None, limit: int, decisions: str | None) -> None:
    """Dump the live gate-audit log as labelable events (the closed loop's real input).

    Reads the same durable ``gate_decisions.jsonl`` the harm gate writes, so a
    reviewer sees the gate's actual recent decisions to label -- not synthetic
    records.
    """
    from omni_mercury_engine.intel.feedback_loop import read_audit_log

    dec = {d.strip() for d in decisions.split(",") if d.strip()} if decisions else None
    events = read_audit_log(path, decisions=dec, limit=limit)
    click.echo(
        json.dumps(
            [
                {
                    "ts": e.ts,
                    "decision": e.decision,
                    "disposition": e.disposition,
                    "hazard_domain": e.hazard_domain,
                    "source": e.source,
                    "query": e.query,
                }
                for e in events
            ],
            indent=2,
        )
    )


@intel.command("rollback")
@click.option("--staging-dir", required=True, help="Staging registry directory to roll back.")
def intel_rollback(staging_dir: str) -> None:
    """One-command rollback of a staged model registry (restore previous, monotonically)."""
    from omni_mercury_engine.intel.feedback_loop import rollback_staging

    result = rollback_staging(staging_dir)
    click.echo(json.dumps(result.as_dict(), indent=2))
    if not result.rolled_back:
        raise SystemExit(1)


@intel.command("red-team")
@click.option(
    "--append",
    is_flag=True,
    help="Append surviving bypasses to corpus/pending (default: report only).",
)
@click.option(
    "--config", default=None, help="Red-team config YAML (default: configs/red_team.yaml)."
)
def intel_red_team(append: bool, config: str | None) -> None:
    """Run the adversarial red-team harness against the LIVE harm gate.

    Reports the surviving-bypass rate (the ``adversarial_co_training`` value
    metric) and, with --append, triages survivors into ``corpus/pending``. Exits
    non-zero if the survival rate exceeds the pinned no-weakening floor.
    """
    from omni_mercury_engine.intel.red_team import (
        RedTeamConfig,
        append_survivors,
        run_red_team,
    )
    from omni_mercury_engine.intel.value_metrics import get_value_metric

    cfg = RedTeamConfig.load(config) if config else None
    result = run_red_team(cfg)
    summary = result.summary()
    floor = get_value_metric("adversarial_co_training").baseline
    click.echo(json.dumps(summary, indent=2))
    if append:
        n = append_survivors(result.survivors)
        click.echo(f"appended {n} new surviving bypass(es) to corpus/pending", err=True)
    rate = result.survival_rate
    if rate > floor + 1e-9:
        click.echo(
            f"FAIL: survival rate {rate:.4f} exceeds no-weakening floor {floor:.4f}", err=True
        )
        raise SystemExit(1)


@intel.command("cascade")
def intel_cascade() -> None:
    """Route the measured cascade workload and report compute saved at bounded accuracy.

    Reuses the same ``evaluate()`` the ``confidence_cascade`` value metric and CI
    lane use (single source of truth), so the operator sees the real routing
    outcome -- not a re-derived number. Degrades transparently when run from an
    installed wheel that does not ship the ``benchmarks`` tree.
    """
    import sys
    from pathlib import Path as _Path

    bench = _Path(__file__).resolve().parents[2] / "benchmarks"
    if not (bench / "confidence_cascade_report.py").is_file():
        click.echo(
            "confidence cascade report unavailable (benchmarks/ not present in this "
            "install); run from a source checkout.",
            err=True,
        )
        raise SystemExit(1)
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    import confidence_cascade_report as ccr  # type: ignore[import-not-found]

    click.echo(json.dumps(ccr.evaluate(), indent=2))


@main.command()
@click.option(
    "--input-topic",
    "-i",
    default=os.environ.get("MERCURY_STREAM_INPUT_TOPIC", "mercury-detections"),
    help="Input topic/stream to consume data points from.",
)
@click.option(
    "--output-topic",
    "-o",
    default=os.environ.get("MERCURY_STREAM_OUTPUT_TOPIC", "mercury-anomalies"),
    help="Output topic/stream to publish anomaly results to.",
)
@click.option(
    "--consumer-group",
    "-g",
    default=os.environ.get("MERCURY_STREAM_CONSUMER_GROUP", "mercury-streaming-workers"),
    help="Consumer group id for load-balanced, at-least-once consumption.",
)
@click.option(
    "--backend",
    "-b",
    default=os.environ.get("STREAMING_BACKEND", "memory"),
    type=click.Choice(["kafka", "redis", "memory"]),
    help=(
        "Streaming backend. 'kafka'/'redis' require a reachable broker "
        "(KAFKA_BOOTSTRAP_SERVERS / REDIS_URL); 'memory' is an in-process, "
        "single-pod default for local development and smoke tests."
    ),
)
@click.option(
    "--stats-interval",
    default=30.0,
    type=float,
    help="Seconds between throughput/latency stat lines on stderr (0 disables).",
)
@click.option(
    "--metrics-port",
    # Pass the env value as a string and let Click coerce it via type=int at
    # invocation time. Calling int() here (decorator/import time) would crash the
    # whole CLI — including `serve`, `--help`, and the k8s probes that import the
    # package — if MERCURY_METRICS_PORT were ever set to a non-integer.
    default=os.environ.get("MERCURY_METRICS_PORT", "9090"),
    type=int,
    help=(
        "Port for the Prometheus /metrics endpoint exposing live pipeline stats "
        "(0 disables). Matches the engine deployment's metrics container port."
    ),
)
def stream(
    input_topic: str,
    output_topic: str,
    consumer_group: str,
    backend: str,
    stats_interval: float,
    metrics_port: int,
) -> None:
    r"""Run a streaming anomaly-detection worker (consume -> detect -> publish).

    This is the long-running worker entrypoint used by the Kubernetes engine /
    streaming-worker deployments. It wires the production ``StreamingAnomalyPipeline``
    (back-pressure handling, a circuit breaker, and full throughput/latency
    observability) to the configured backend and runs until the process receives
    SIGTERM/SIGINT, at which point it drains and shuts the pipeline down cleanly.

    Examples:
        mercury stream                              # in-memory dev worker
        mercury stream --backend kafka \\
            --input-topic mercury-detections \\
            --output-topic mercury-anomalies \\
            --consumer-group mercury-streaming-workers
    """
    import asyncio

    try:
        from omni_mercury_engine.infrastructure.streaming import (
            StreamConfig,
            StreamingAnomalyPipeline,
            StreamingBackend,
        )
    except ImportError as exc:  # optional aiokafka/redis extras not installed
        click.echo(
            "Error: streaming dependencies are not installed. "
            f"Install with: pip install 'mercury-agent[streaming]' ({exc})",
            err=True,
        )
        raise SystemExit(1) from exc

    click.echo("\n" + "=" * 60, err=True)
    click.echo("  Mercury Agent Streaming Worker", err=True)
    click.echo("=" * 60, err=True)
    click.echo(f"  Backend:        {backend}", err=True)
    click.echo(f"  Input topic:    {input_topic}", err=True)
    click.echo(f"  Output topic:   {output_topic}", err=True)
    click.echo(f"  Consumer group: {consumer_group}", err=True)
    click.echo("-" * 60 + "\n", err=True)

    def _start_metrics_server(pipeline: Any) -> Any:
        """Serve live pipeline stats as Prometheus exposition on ``metrics_port``.

        Hand-rolls the text exposition with the stdlib (matching
        ``api/health.py``'s ``/metrics`` endpoint) so the worker needs no extra
        runtime dependency. Returns the server (for shutdown) or ``None`` when
        metrics are disabled.
        """
        if not metrics_port or metrics_port <= 0:
            return None

        import http.server
        import threading

        # (metric_name, stats_key, prom_type, help_text)
        metrics_spec = (
            (
                "mercury_stream_messages_processed",
                "messages_processed",
                "counter",
                "Messages processed",
            ),
            (
                "mercury_stream_anomalies_detected",
                "anomalies_detected",
                "counter",
                "Anomalies detected",
            ),
            ("mercury_stream_errors_total", "errors", "counter", "Processing errors"),
            (
                "mercury_stream_messages_per_second",
                "messages_per_second",
                "gauge",
                "Throughput (msg/s)",
            ),
            (
                "mercury_stream_anomaly_rate",
                "anomaly_rate",
                "gauge",
                "Fraction of messages flagged",
            ),
            (
                "mercury_stream_uptime_seconds",
                "uptime_seconds",
                "gauge",
                "Worker uptime in seconds",
            ),
        )

        def _exposition() -> bytes:
            stats = pipeline.get_stats()
            lines: list[str] = []
            for metric_name, stats_key, prom_type, help_text in metrics_spec:
                lines.append(f"# HELP {metric_name} {help_text}")
                lines.append(f"# TYPE {metric_name} {prom_type}")
                lines.append(f"{metric_name} {float(stats.get(stats_key, 0.0) or 0.0)}")
            return ("\n".join(lines) + "\n").encode()

        class _MetricsHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = _exposition()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: Any) -> None:  # silence per-request logging
                return

        # Secure-by-default bind: loopback unless the operator widens it. On a
        # bare host (VPS, laptop) the endpoint must not be implicitly exposed
        # to the network. Kubernetes needs Prometheus to reach it cross-pod at
        # <pod-ip>:<metrics_port>, so the in-repo manifests (k8s configmap +
        # Helm values) set MERCURY_METRICS_HOST=0.0.0.0 explicitly — the
        # exposure decision lives in the deployment, not the code. Set
        # MERCURY_METRICS_PORT=0 to disable the server entirely.
        bind_host = os.environ.get("MERCURY_METRICS_HOST", "127.0.0.1")
        server = http.server.ThreadingHTTPServer((bind_host, metrics_port), _MetricsHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        click.echo(f"  Metrics:        http://{bind_host}:{metrics_port}/metrics", err=True)
        return server

    async def _run() -> None:
        config = StreamConfig(backend=StreamingBackend(backend))
        pipeline = StreamingAnomalyPipeline(
            input_topic=input_topic,
            output_topic=output_topic,
            backend=config.backend,
            config=config,
            group_id=consumer_group,
        )
        metrics_server = _start_metrics_server(pipeline)

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig_name in ("SIGTERM", "SIGINT"):
            import signal

            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):
                # Signal handlers are unavailable on some platforms (e.g. Windows);
                # KeyboardInterrupt below still triggers a clean shutdown.
                pass

        # ``pipeline.start()`` lives inside the try so a broker-connection
        # failure (kafka/redis backends) still runs the finally below and the
        # metrics server is torn down rather than left bound until process exit.
        try:
            await pipeline.start()
            while not stop.is_set():
                if stats_interval and stats_interval > 0:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=stats_interval)
                    except TimeoutError:
                        s = pipeline.get_stats()
                        click.echo(
                            f"[stream] processed={s['messages_processed']} "
                            f"anomalies={s['anomalies_detected']} "
                            f"errors={s['errors']} "
                            f"mps={s['messages_per_second']:.1f}",
                            err=True,
                        )
                else:
                    await stop.wait()
        finally:
            # Nested so the metrics server is shut down even if pipeline.stop()
            # itself raises (e.g. a half-connected backend during teardown).
            try:
                await pipeline.stop()
            finally:
                if metrics_server is not None:
                    metrics_server.shutdown()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        click.echo("\nStreaming worker interrupted; shutting down.", err=True)


@main.command()
@click.option("--domain", "-d", default=None, help="Domain context (medical, security, etc.)")
@click.option(
    "--model",
    "-m",
    default="",
    help="Ollama model to use (no default ships; falls back to MERCURY_OLLAMA_MODEL)",
)
@click.option("--offline", is_flag=True, help="Force offline mode (template responses)")
def voice(domain: str, model: str, offline: bool) -> None:
    """Start interactive voice conversation with Mercury.

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


# ---------------------------------------------------------------------------
# platform operator commands (local admin plane over the shared SQLite store)
# ---------------------------------------------------------------------------
@main.group()
def platform() -> None:
    """Operate the account platform: accounts, quotas, usage, audit.

    These commands are the operator surface for runtime platform state
    (account tiers, per-account quota overrides, usage reports, audit-chain
    verification). They run *on the box* against the shared SQLite store
    (`MERCURY_KEYSTORE_PATH`) and the audit directory
    (`MERCURY_AUDIT_LOG_DIR`) — deliberately not a privileged HTTP surface,
    so there is nothing extra to protect on a solo deployment. Secrets
    (password hashes, sealed TOTP material, key hashes) are never printed.
    """


def _require_env_path(name: str, purpose: str) -> str:
    """Return the value of env var ``name`` or exit with a clear refusal."""
    value = os.getenv(name, "").strip()
    if not value:
        click.echo(
            f"Error: {name} is not set — {purpose}. Export it to point at the "
            "deployment's state before using `mercury-agent platform`.",
            err=True,
        )
        raise SystemExit(1)
    return value


def _resolve_account(identifier: str) -> Any:
    """Resolve an account by email (preferred) or id, or exit 1."""
    from omni_mercury_engine.api.identity_store import build_identity_store

    store = build_identity_store()
    account = store.get_account_by_email(identifier) or store.get_account_by_id(identifier)
    if account is None:
        click.echo(f"Error: no account matches {identifier!r} (by email or id).", err=True)
        raise SystemExit(1)
    return account


def _account_view(account: Any) -> dict[str, Any]:
    """Project an account to its operator-safe view (no secret material)."""
    return {
        "id": account.id,
        "email": account.email,
        "tier": account.tier,
        "is_verified": account.is_verified,
        "is_active": account.is_active,
        "totp_enabled": account.totp_enabled,
        "created_at": account.created_at.isoformat(),
    }


def _effective_quota(account: Any) -> dict[str, Any]:
    """The account's resolved quota config (override > tier > default)."""
    from omni_mercury_engine.api.quota import build_quota_enforcer

    config = build_quota_enforcer().config_for(account.id, account.tier)
    return {
        "window_seconds": config.window_seconds,
        "max_requests": config.max_requests,
        "max_compute_ms": config.max_compute_ms,
    }


@platform.group("account")
def platform_account() -> None:
    """Inspect and administer accounts (tier, enable/disable)."""


@platform_account.command("show")
@click.argument("identifier")
def platform_account_show(identifier: str) -> None:
    """Show one account (by email or id) with its effective quota config."""
    _require_env_path("MERCURY_KEYSTORE_PATH", "accounts live in the shared SQLite store")
    account = _resolve_account(identifier)
    payload = _account_view(account)
    payload["effective_quota"] = _effective_quota(account)
    click.echo(json.dumps(payload, indent=2))


@platform_account.command("list")
def platform_account_list() -> None:
    """List every account (operator-safe fields only)."""
    _require_env_path("MERCURY_KEYSTORE_PATH", "accounts live in the shared SQLite store")
    from omni_mercury_engine.api.identity_store import build_identity_store

    accounts = [_account_view(account) for account in build_identity_store().iter_accounts()]
    click.echo(json.dumps(accounts, indent=2))


@platform_account.command("set-tier")
@click.argument("identifier")
@click.argument("tier")
def platform_account_set_tier(identifier: str, tier: str) -> None:
    """Move an account onto TIER and echo the resulting effective quota.

    Tier ceilings come from `MERCURY_QUOTA_TIER_<NAME>`; a name with no
    definition resolves to the default ceilings (the echoed config shows
    exactly what enforcement will use, so a typo is immediately visible).
    """
    _require_env_path("MERCURY_KEYSTORE_PATH", "accounts live in the shared SQLite store")
    from omni_mercury_engine.api.identity_store import build_identity_store

    account = _resolve_account(identifier)
    account.tier = tier.strip().lower()
    build_identity_store().update_account(account)
    payload = _account_view(account)
    payload["effective_quota"] = _effective_quota(account)
    click.echo(json.dumps(payload, indent=2))


def _set_account_active(identifier: str, active: bool) -> None:
    """Shared body of ``account disable`` / ``account enable``."""
    _require_env_path("MERCURY_KEYSTORE_PATH", "accounts live in the shared SQLite store")
    from omni_mercury_engine.api.identity_store import build_identity_store

    store = build_identity_store()
    account = _resolve_account(identifier)
    account.is_active = active
    store.update_account(account)
    if not active:
        # A disabled account must not keep riding an existing login.
        store.delete_sessions_for_account(account.id)
    click.echo(json.dumps(_account_view(account), indent=2))


@platform_account.command("disable")
@click.argument("identifier")
def platform_account_disable(identifier: str) -> None:
    """Deactivate an account (blocks login and drops its live sessions)."""
    _set_account_active(identifier, active=False)


@platform_account.command("enable")
@click.argument("identifier")
def platform_account_enable(identifier: str) -> None:
    """Reactivate a previously disabled account."""
    _set_account_active(identifier, active=True)


@platform.group("quota")
def platform_quota() -> None:
    """Manage per-account quota overrides."""


@platform_quota.group("override")
def platform_quota_override() -> None:
    """Set, clear, or show one account's quota override (top precedence)."""


@platform_quota_override.command("set")
@click.argument("identifier")
@click.option("--max-requests", required=True, type=int, help="Request ceiling per window")
@click.option("--max-compute-ms", required=True, type=float, help="Compute-ms ceiling per window")
@click.option(
    "--window-seconds",
    default=None,
    type=int,
    help="Rolling window length (defaults to the deployment's default window)",
)
def platform_quota_override_set(
    identifier: str, max_requests: int, max_compute_ms: float, window_seconds: int | None
) -> None:
    """Give an account its own ceilings, overriding tier and default."""
    if (
        max_requests < 1
        or max_compute_ms <= 0
        or (window_seconds is not None and window_seconds < 1)
    ):
        raise click.BadParameter("ceilings and window must be positive")
    _require_env_path("MERCURY_KEYSTORE_PATH", "quota overrides live in the shared SQLite store")
    from omni_mercury_engine.api.quota import QuotaConfig, build_quota_enforcer

    account = _resolve_account(identifier)
    enforcer = build_quota_enforcer()
    config = QuotaConfig(
        window_seconds=(
            window_seconds if window_seconds is not None else QuotaConfig.from_env().window_seconds
        ),
        max_requests=max_requests,
        max_compute_ms=max_compute_ms,
    )
    enforcer.override_store.set_override(account.id, config)
    payload = _account_view(account)
    payload["effective_quota"] = _effective_quota(account)
    click.echo(json.dumps(payload, indent=2))


@platform_quota_override.command("clear")
@click.argument("identifier")
def platform_quota_override_clear(identifier: str) -> None:
    """Remove an account's override; tier/default resolution applies again."""
    _require_env_path("MERCURY_KEYSTORE_PATH", "quota overrides live in the shared SQLite store")
    from omni_mercury_engine.api.quota import build_quota_enforcer

    account = _resolve_account(identifier)
    build_quota_enforcer().override_store.set_override(account.id, None)
    payload = _account_view(account)
    payload["effective_quota"] = _effective_quota(account)
    click.echo(json.dumps(payload, indent=2))


@platform_quota_override.command("show")
@click.argument("identifier")
def platform_quota_override_show(identifier: str) -> None:
    """Show an account's stored override (if any) and its effective config."""
    _require_env_path("MERCURY_KEYSTORE_PATH", "quota overrides live in the shared SQLite store")
    from omni_mercury_engine.api.quota import build_quota_enforcer

    account = _resolve_account(identifier)
    override = build_quota_enforcer().override_store.get_override(account.id)
    payload = _account_view(account)
    payload["override"] = (
        {
            "window_seconds": override.window_seconds,
            "max_requests": override.max_requests,
            "max_compute_ms": override.max_compute_ms,
        }
        if override is not None
        else None
    )
    payload["effective_quota"] = _effective_quota(account)
    click.echo(json.dumps(payload, indent=2))


@platform.group("usage")
def platform_usage() -> None:
    """Report metered usage from the ledger."""


@platform_usage.command("report")
@click.option("--account", "identifier", default=None, help="Report one account (email or id)")
@click.option(
    "--top", "top_n", default=10, type=int, help="How many accounts to list (by requests)"
)
@click.option(
    "--window-seconds",
    default=None,
    type=int,
    help="Look-back window (defaults to the deployment's default quota window)",
)
def platform_usage_report(identifier: str | None, top_n: int, window_seconds: int | None) -> None:
    """Summarise in-window usage — one account, or the top consumers."""
    if top_n < 1 or (window_seconds is not None and window_seconds < 1):
        raise click.BadParameter("--top and --window-seconds must be positive")
    _require_env_path("MERCURY_KEYSTORE_PATH", "the usage ledger lives in the shared SQLite store")
    from datetime import UTC, datetime, timedelta

    from omni_mercury_engine.api.identity_store import build_identity_store
    from omni_mercury_engine.api.quota import QuotaConfig
    from omni_mercury_engine.api.usage_ledger import build_usage_ledger

    window = window_seconds if window_seconds is not None else QuotaConfig.from_env().window_seconds
    since = datetime.now(UTC) - timedelta(seconds=window)
    ledger = build_usage_ledger()

    def _row(account: Any) -> dict[str, Any]:
        summary = ledger.summary_since(account.id, since)
        return {
            "id": account.id,
            "email": account.email,
            "tier": account.tier,
            "requests": summary.request_count,
            "compute_ms": summary.compute_ms,
        }

    if identifier is not None:
        rows = [_row(_resolve_account(identifier))]
    else:
        rows = sorted(
            (_row(account) for account in build_identity_store().iter_accounts()),
            key=lambda row: (-int(row["requests"]), str(row["email"])),
        )[:top_n]
    click.echo(json.dumps({"window_seconds": window, "accounts": rows}, indent=2))


@platform.group("audit")
def platform_audit() -> None:
    """Verify the tamper-evident audit trail."""


@platform_audit.command("verify")
@click.option(
    "--file",
    "file_path",
    default=None,
    help="Verify a specific segment file (defaults to the active audit.jsonl)",
)
def platform_audit_verify(file_path: str | None) -> None:
    """Verify the audit log's HMAC hash chain; exit 1 on any break.

    Verification recomputes the chain with the key derived from
    `AMA_MASTER_SEED` — run it with the same environment the server writes
    with, or every line will (correctly) fail to verify.
    """
    audit_dir = _require_env_path(
        "MERCURY_AUDIT_LOG_DIR", "audit verification reads the deployment's audit directory"
    )
    from omni_mercury_engine.security.secure_audit_logging import SecureAuditLogger

    logger = SecureAuditLogger(log_dir=audit_dir)
    try:
        target = Path(file_path) if file_path else None
        ok, message = logger.verify_log_integrity(target)
    finally:
        logger.shutdown()
    click.echo(json.dumps({"ok": ok, "message": message}, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
