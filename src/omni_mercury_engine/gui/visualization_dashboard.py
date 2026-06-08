# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enhanced Visualization Dashboard.

Enhanced visualization dashboard providing:
- Interactive Plotly-based visualizations
- Real-time anomaly monitoring
- Time series analysis with annotations
- Feature importance visualization
- Detector performance comparison
- Correlation heatmaps
- 3D anomaly visualization
- Exportable reports
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Optional imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None
    make_subplots = None


class ChartType(StrEnum):
    """Available chart types."""

    TIME_SERIES = "time_series"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    BAR = "bar"
    BOX = "box"
    VIOLIN = "violin"
    HISTOGRAM = "histogram"
    PIE = "pie"
    SUNBURST = "sunburst"
    SANKEY = "sankey"
    SCATTER_3D = "scatter_3d"
    SURFACE_3D = "surface_3d"
    RADAR = "radar"


@dataclass
class ChartConfig:
    """Configuration for chart generation."""

    title: str = ""
    x_label: str = ""
    y_label: str = ""
    width: int = 800
    height: int = 500
    theme: str = "plotly_dark"
    show_legend: bool = True
    animation: bool = False
    export_format: str = "html"


@dataclass
class DashboardConfig:
    """Configuration for the dashboard."""

    title: str = "Mercury Agent Anomaly Detection Dashboard"
    refresh_interval_ms: int = 5000
    theme: str = "plotly_dark"
    enable_streaming: bool = True
    max_data_points: int = 10000
    anomaly_threshold: float = 0.5
    export_path: str = "reports"


@dataclass
class AnomalyDataPoint:
    """Single data point for visualization."""

    timestamp: datetime
    score: float
    is_anomaly: bool
    features: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    detector: str = "unknown"


class AnomalyVisualizer:
    """Core visualizer for anomaly detection results.

    Provides multiple visualization types for comprehensive analysis.
    """

    def __init__(
        self,
        config: ChartConfig | None = None,
    ):
        """Initialize visualizer.

        Args:
            config: Chart configuration
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for visualization. Install with: pip install plotly"
            )

        self.config = config or ChartConfig()
        self._data_buffer: list[AnomalyDataPoint] = []

    def time_series_plot(
        self,
        timestamps: list[datetime] | NDArray,  # type: ignore[type-arg, unused-ignore]
        scores: NDArray[np.float64],
        threshold: float = 0.5,
        anomaly_mask: NDArray[np.bool_] | None = None,
        title: str = "Anomaly Scores Over Time",
    ) -> go.Figure:
        """Create time series plot of anomaly scores.

        Args:
            timestamps: Time points
            scores: Anomaly scores
            threshold: Anomaly threshold
            anomaly_mask: Boolean mask for anomalies
            title: Chart title

        Returns:
            Plotly figure
        """
        if anomaly_mask is None:
            anomaly_mask = scores > threshold

        fig = go.Figure()

        # Normal points
        normal_mask = ~anomaly_mask
        fig.add_trace(
            go.Scatter(
                x=np.array(timestamps)[normal_mask],
                y=scores[normal_mask],
                mode="markers",
                name="Normal",
                marker=dict(color="green", size=6, opacity=0.6),
            )
        )

        # Anomaly points
        fig.add_trace(
            go.Scatter(
                x=np.array(timestamps)[anomaly_mask],
                y=scores[anomaly_mask],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=10, symbol="x"),
            )
        )

        # Score line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=scores,
                mode="lines",
                name="Score",
                line=dict(color="rgba(100, 150, 255, 0.5)", width=1),
            )
        )

        # Threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Threshold: {threshold}",
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            template=self.config.theme,
            hovermode="x unified",
            showlegend=True,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def feature_importance_plot(
        self,
        feature_names: list[str],
        importances: NDArray[np.float64],
        top_k: int = 15,
        title: str = "Feature Importance",
    ) -> go.Figure:
        """Create horizontal bar chart of feature importance.

        Args:
            feature_names: Feature names
            importances: Importance values
            top_k: Number of top features to show
            title: Chart title

        Returns:
            Plotly figure
        """
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1][:top_k]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = importances[sorted_idx]

        # Color based on value
        colors = ["red" if v > np.mean(importances) else "blue" for v in sorted_values]

        fig = go.Figure(
            go.Bar(
                x=sorted_values[::-1],
                y=sorted_names[::-1],
                orientation="h",
                marker_color=colors[::-1],
                text=[f"{v:.3f}" for v in sorted_values[::-1]],
                textposition="outside",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            template=self.config.theme,
            width=self.config.width,
            height=max(self.config.height, top_k * 30),
        )

        return fig

    def correlation_heatmap(
        self,
        data: NDArray[np.float64],
        feature_names: list[str] | None = None,
        title: str = "Feature Correlation Matrix",
    ) -> go.Figure:
        """Create correlation heatmap.

        Args:
            data: Feature matrix
            feature_names: Feature names
            title: Chart title

        Returns:
            Plotly figure
        """
        corr_matrix = np.corrcoef(data.T)

        if feature_names is None:
            feature_names = [f"F{i}" for i in range(data.shape[1])]

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix,
                x=feature_names,
                y=feature_names,
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def detector_comparison_plot(
        self,
        detector_scores: dict[str, NDArray[np.float64]],
        labels: NDArray[np.int64] | None = None,
        title: str = "Detector Performance Comparison",
    ) -> go.Figure:
        """Compare multiple detectors' performance.

        Args:
            detector_scores: Dict of detector name to scores
            labels: True labels (optional)
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Score Distribution",
                "ROC Curves",
                "Score Correlation",
                "Performance Metrics",
            ],
            specs=[
                [{"type": "box"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}],
            ],
        )

        # Box plots of score distributions
        for name, scores in detector_scores.items():
            fig.add_trace(
                go.Box(y=scores, name=name, boxmean=True),
                row=1,
                col=1,
            )

        # ROC curves (if labels available)
        if labels is not None:
            for name, scores in detector_scores.items():
                fpr, tpr = self._compute_roc(labels, scores)
                _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined, unused-ignore]
                auc = _trapz(tpr, fpr)
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})", mode="lines"),
                    row=1,
                    col=2,
                )
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash")),
                row=1,
                col=2,
            )
        else:
            # Placeholder if no labels
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text="Labels required for ROC curves",
                showarrow=False,
                row=1,
                col=2,
            )

        # Score correlation heatmap
        detector_names = list(detector_scores.keys())
        n_detectors = len(detector_names)
        score_matrix = np.array([detector_scores[name] for name in detector_names])

        if n_detectors > 1:
            corr_matrix = np.corrcoef(score_matrix)
            fig.add_trace(
                go.Heatmap(z=corr_matrix, x=detector_names, y=detector_names, colorscale="RdBu"),
                row=2,
                col=1,
            )

        # Performance metrics
        if labels is not None:
            metrics = []
            for name, scores in detector_scores.items():
                preds = scores > 0.5
                tp = np.sum((preds == 1) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                fn = np.sum((preds == 0) & (labels == 1))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                )
                metrics.append({"name": name, "F1": f1, "Precision": precision, "Recall": recall})

            for metric_name in ["F1", "Precision", "Recall"]:
                values = [m[metric_name] for m in metrics]
                fig.add_trace(
                    go.Bar(x=[m["name"] for m in metrics], y=values, name=metric_name),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            title=title,
            template=self.config.theme,
            showlegend=True,
            width=self.config.width * 1.5,
            height=self.config.height * 1.5,
        )

        return fig

    def _compute_roc(
        self,
        labels: NDArray[np.int64],
        scores: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute ROC curve points."""
        thresholds = np.linspace(0, 1, 100)
        fpr_list = []
        tpr_list = []

        for thresh in thresholds:
            preds = (scores >= thresh).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            tn = np.sum((preds == 0) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

            fpr_list.append(fpr)
            tpr_list.append(tpr)

        return np.array(fpr_list), np.array(tpr_list)

    def anomaly_scatter_3d(
        self,
        data: NDArray[np.float64],
        scores: NDArray[np.float64],
        anomaly_mask: NDArray[np.bool_] | None = None,
        title: str = "3D Anomaly Visualization",
    ) -> go.Figure:
        """Create 3D scatter plot of anomalies.

        Args:
            data: Feature matrix (uses first 3 features or PCA)
            scores: Anomaly scores
            anomaly_mask: Boolean mask for anomalies
            title: Chart title

        Returns:
            Plotly figure
        """
        # Reduce to 3D if needed
        if data.shape[1] > 3:
            try:
                from omni_mercury_engine.ml.mercury_ml import PCA

                pca = PCA(n_components=3)
                data_3d = pca.fit_transform(data)
                axis_labels = ["PC1", "PC2", "PC3"]
            except ImportError:
                data_3d = data[:, :3]
                axis_labels = ["Feature 1", "Feature 2", "Feature 3"]
        else:
            data_3d = (
                data[:, :3]
                if data.shape[1] >= 3
                else np.pad(data, ((0, 0), (0, 3 - data.shape[1])))
            )
            axis_labels = ["Feature 1", "Feature 2", "Feature 3"]

        if anomaly_mask is None:
            anomaly_mask = scores > 0.5

        # Size based on anomaly status (larger for anomalies)
        sizes = np.where(anomaly_mask, 10, 5)

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=data_3d[:, 0],
                    y=data_3d[:, 1],
                    z=data_3d[:, 2],
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color=scores,
                        colorscale="RdYlGn_r",
                        colorbar=dict(title="Anomaly Score"),
                        opacity=0.8,
                    ),
                    text=[f"Score: {s:.3f}" for s in scores],
                    hoverinfo="text",
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2],
            ),
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def radar_chart(
        self,
        categories: list[str],
        values: dict[str, list[float]],
        title: str = "Multi-Detector Radar Chart",
    ) -> go.Figure:
        """Create radar chart comparing detectors across metrics.

        Args:
            categories: Metric categories
            values: Dict of detector name to metric values
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for name, vals in values.items():
            fig.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=categories,
                    fill="toself",
                    name=name,
                )
            )

        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
            ),
            template=self.config.theme,
            showlegend=True,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def anomaly_timeline(
        self,
        events: list[AnomalyDataPoint],
        title: str = "Anomaly Event Timeline",
    ) -> go.Figure:
        """Create timeline visualization of anomaly events.

        Args:
            events: List of anomaly data points
            title: Chart title

        Returns:
            Plotly figure
        """
        # Filter to anomalies only
        anomalies = [e for e in events if e.is_anomaly]

        if not anomalies:
            fig = go.Figure()
            fig.add_annotation(text="No anomalies detected", x=0.5, y=0.5, showarrow=False)
            return fig

        timestamps = [e.timestamp for e in anomalies]
        scores = [e.score for e in anomalies]
        detectors = [e.detector for e in anomalies]
        hover_text = [f"Detector: {e.detector}<br>Score: {e.score:.3f}" for e in anomalies]

        # Color by detector
        unique_detectors = list(set(detectors))
        color_map = {d: i for i, d in enumerate(unique_detectors)}
        colors = [color_map[d] for d in detectors]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=timestamps,
                    y=scores,
                    mode="markers+text",
                    marker=dict(
                        size=15,
                        color=colors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Detector"),
                    ),
                    text=[f"{s:.2f}" for s in scores],
                    textposition="top center",
                    hovertext=hover_text,
                    hoverinfo="text",
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def distribution_plot(
        self,
        scores: NDArray[np.float64],
        threshold: float = 0.5,
        title: str = "Score Distribution",
    ) -> go.Figure:
        """Create distribution plot of anomaly scores.

        Args:
            scores: Anomaly scores
            threshold: Anomaly threshold
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Histogram", "Box Plot"],
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=50,
                name="Scores",
                marker_color="rgba(100, 150, 255, 0.7)",
            ),
            row=1,
            col=1,
        )

        # Add threshold line
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold}",
            row=1,
            col=1,
        )

        # Box plot
        fig.add_trace(
            go.Box(
                y=scores,
                name="Scores",
                boxmean="sd",
                marker_color="rgba(100, 150, 255, 0.7)",
            ),
            row=1,
            col=2,
        )

        # Add threshold line to box plot
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            row=1,
            col=2,
        )

        fig.update_layout(
            title=title,
            template=self.config.theme,
            showlegend=False,
            width=self.config.width,
            height=self.config.height // 2,
        )

        return fig


class DashboardBuilder:
    """Builder class for creating comprehensive anomaly detection dashboards."""

    def __init__(
        self,
        config: DashboardConfig | None = None,
    ):
        """Initialize dashboard builder.

        Args:
            config: Dashboard configuration
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for dashboards. Install with: pip install plotly")

        self.config = config or DashboardConfig()
        self.visualizer = AnomalyVisualizer()
        self._figures: dict[str, go.Figure] = {}

    def add_time_series(
        self,
        name: str,
        timestamps: list[datetime] | NDArray,  # type: ignore[type-arg, unused-ignore]
        scores: NDArray[np.float64],
        **kwargs: Any,
    ) -> DashboardBuilder:
        """Add time series chart to dashboard."""
        fig = self.visualizer.time_series_plot(
            timestamps,
            scores,
            threshold=self.config.anomaly_threshold,
            **kwargs,
        )
        self._figures[name] = fig
        return self

    def add_feature_importance(
        self,
        name: str,
        feature_names: list[str],
        importances: NDArray[np.float64],
        **kwargs: Any,
    ) -> DashboardBuilder:
        """Add feature importance chart to dashboard."""
        fig = self.visualizer.feature_importance_plot(
            feature_names,
            importances,
            **kwargs,
        )
        self._figures[name] = fig
        return self

    def add_correlation_heatmap(
        self,
        name: str,
        data: NDArray[np.float64],
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> DashboardBuilder:
        """Add correlation heatmap to dashboard."""
        fig = self.visualizer.correlation_heatmap(
            data,
            feature_names,
            **kwargs,
        )
        self._figures[name] = fig
        return self

    def add_detector_comparison(
        self,
        name: str,
        detector_scores: dict[str, NDArray[np.float64]],
        labels: NDArray[np.int64] | None = None,
        **kwargs: Any,
    ) -> DashboardBuilder:
        """Add detector comparison to dashboard."""
        fig = self.visualizer.detector_comparison_plot(
            detector_scores,
            labels,
            **kwargs,
        )
        self._figures[name] = fig
        return self

    def add_3d_visualization(
        self,
        name: str,
        data: NDArray[np.float64],
        scores: NDArray[np.float64],
        **kwargs: Any,
    ) -> DashboardBuilder:
        """Add 3D anomaly visualization to dashboard."""
        fig = self.visualizer.anomaly_scatter_3d(
            data,
            scores,
            **kwargs,
        )
        self._figures[name] = fig
        return self

    def add_distribution(
        self,
        name: str,
        scores: NDArray[np.float64],
        **kwargs: Any,
    ) -> DashboardBuilder:
        """Add score distribution to dashboard."""
        fig = self.visualizer.distribution_plot(
            scores,
            threshold=self.config.anomaly_threshold,
            **kwargs,
        )
        self._figures[name] = fig
        return self

    def build(self) -> dict[str, go.Figure]:
        """Build and return all figures."""
        return self._figures

    def export_html(self, filepath: str) -> None:
        """Export dashboard as single HTML file.

        Args:
            filepath: Output file path
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{self.config.title}</title>",
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            "<style>",
            "body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }",
            ".chart-container { margin: 20px 0; padding: 20px; background: #16213e; border-radius: 10px; }",
            "h1 { color: #00d9ff; text-align: center; }",
            "h2 { color: #00ff88; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{self.config.title}</h1>",
            f"<p>Generated: {datetime.now(UTC).isoformat()}</p>",
        ]

        for name, fig in self._figures.items():
            div_id = name.replace(" ", "_").lower()
            html_parts.extend(
                [
                    '<div class="chart-container">',
                    f"<h2>{name}</h2>",
                    f'<div id="{div_id}"></div>',
                    "<script>",
                    f"var data_{div_id} = {fig.to_json()};",
                    f"Plotly.newPlot('{div_id}', data_{div_id}.data, data_{div_id}.layout);",
                    "</script>",
                    "</div>",
                ]
            )

        html_parts.extend(
            [
                "</body>",
                "</html>",
            ]
        )

        with open(filepath, "w") as f:
            f.write("\n".join(html_parts))

        logger.info(f"Dashboard exported to {filepath}")

    def export_json(self, filepath: str) -> None:
        """Export dashboard data as JSON.

        Args:
            filepath: Output file path
        """
        dashboard_data = {
            "title": self.config.title,
            "generated_at": datetime.now(UTC).isoformat(),
            "config": {
                "threshold": self.config.anomaly_threshold,
                "theme": self.config.theme,
            },
            "figures": {name: json.loads(fig.to_json()) for name, fig in self._figures.items()},
        }

        with open(filepath, "w") as f:
            json.dump(dashboard_data, f, indent=2)

        logger.info(f"Dashboard data exported to {filepath}")


def create_quick_dashboard(
    scores: NDArray[np.float64],
    data: NDArray[np.float64] | None = None,
    timestamps: list[datetime] | None = None,
    feature_names: list[str] | None = None,
    labels: NDArray[np.int64] | None = None,
    title: str = "Mercury Agent Quick Dashboard",
) -> DashboardBuilder:
    """Create a quick dashboard with common visualizations.

    Args:
        scores: Anomaly scores
        data: Feature data (optional)
        timestamps: Time points (optional)
        feature_names: Feature names (optional)
        labels: True labels (optional)
        title: Dashboard title

    Returns:
        Configured DashboardBuilder
    """
    config = DashboardConfig(title=title)
    builder = DashboardBuilder(config)

    # Add score distribution
    builder.add_distribution("Score Distribution", scores)

    # Add time series if timestamps available
    if timestamps is not None:
        builder.add_time_series("Anomaly Timeline", timestamps, scores)

    # Add correlation if data available
    if data is not None and data.shape[1] > 1:
        builder.add_correlation_heatmap("Feature Correlations", data, feature_names)

        # Add 3D visualization
        if data.shape[1] >= 3:
            builder.add_3d_visualization("3D Anomaly View", data, scores)

        # Compute simple feature importance from anomaly correlation
        if labels is not None:
            correlations = np.array(
                [np.corrcoef(data[:, j], labels)[0, 1] for j in range(data.shape[1])]
            )
            correlations = np.nan_to_num(np.abs(correlations))

            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(data.shape[1])]

            builder.add_feature_importance("Feature Importance", feature_names, correlations)

    return builder


# Exports
__all__ = [
    "AnomalyDataPoint",
    "AnomalyVisualizer",
    "ChartConfig",
    "ChartType",
    "DashboardBuilder",
    "DashboardConfig",
    "create_quick_dashboard",
]
