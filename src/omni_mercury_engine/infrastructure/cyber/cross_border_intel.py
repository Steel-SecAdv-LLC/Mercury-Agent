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


"""Cross-border threat intelligence correlation.

Correlates anomaly patterns across international boundaries (EU-US)
for comprehensive threat intelligence.
"""

from typing import Any

import numpy as np


class CrossBorderIntelligence:
    """Cross-border threat intelligence correlation.

    Correlates anomaly patterns across international boundaries (EU-US)
    for comprehensive threat intelligence.
    """

    def __init__(self) -> None:
        """Initialize Cross-Border Intelligence."""
        self.regions = {
            "eu": ["france", "germany", "italy", "spain", "poland", "netherlands"],
            "us": ["northeast", "southeast", "midwest", "southwest", "west"],
        }

    def correlate_threats(
        self, eu_data: np.ndarray[Any, Any], us_data: np.ndarray[Any, Any], threat_type: str
    ) -> dict[str, Any]:
        """Correlate threat patterns across EU and US data.

        Args:
            eu_data: Anomaly data from EU critical entities
            us_data: Anomaly data from US critical infrastructure
            threat_type: Type of threat to correlate ('cyber', 'supply_chain', 'energy')

        Returns:
            Correlation analysis with shared threat indicators, geographic patterns
        """
        correlation = (
            np.corrcoef(eu_data, us_data)[0, 1] if len(eu_data) > 1 and len(us_data) > 1 else 0.0
        )
        synchronized = correlation > 0.7
        lag_analysis = self._calculate_time_lag(eu_data, us_data)

        return {
            "correlation_coefficient": float(correlation),
            "synchronized_threat": synchronized,
            "threat_type": threat_type,
            "time_lag_hours": lag_analysis["lag_hours"],
            "leading_region": lag_analysis["leader"],
            "confidence": 0.85 if synchronized else 0.60,
            "recommendations": self._generate_cross_border_recommendations(
                synchronized, threat_type
            ),
        }

    def _calculate_time_lag(
        self, eu_data: np.ndarray[Any, Any], us_data: np.ndarray[Any, Any]
    ) -> dict[str, Any]:
        """Calculate time lag between EU and US anomaly patterns."""
        max_lag = min(24, len(eu_data) // 2, len(us_data) // 2)
        if max_lag < 1:
            return {"lag_hours": 0, "leader": "simultaneous", "correlation_at_best_lag": 0.0}

        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = (
                    np.corrcoef(eu_data[:lag], us_data[-lag:])[0, 1]
                    if len(eu_data[:lag]) > 1
                    else 0
                )
            elif lag > 0:
                corr = (
                    np.corrcoef(eu_data[lag:], us_data[:-lag])[0, 1]
                    if len(us_data[:-lag]) > 1
                    else 0
                )
            else:
                corr = np.corrcoef(eu_data, us_data)[0, 1] if len(eu_data) > 1 else 0
            correlations.append((lag, corr))

        best_lag, best_corr = max(correlations, key=lambda x: x[1])

        return {
            "lag_hours": best_lag,
            "leader": "EU" if best_lag < 0 else "US" if best_lag > 0 else "simultaneous",
            "correlation_at_best_lag": best_corr,
        }

    def _generate_cross_border_recommendations(
        self, synchronized: bool, threat_type: str
    ) -> list[Any]:
        """Generate recommendations for cross-border threats."""
        if synchronized:
            return [
                f"ALERT: Synchronized {threat_type} threat detected across EU and US",
                "Activate international incident response coordination",
                "Share threat intelligence with partner agencies",
                "Implement heightened security posture across all sectors",
            ]
        else:
            return [
                f"Regional {threat_type} activity detected",
                "Monitor for potential spread to other regions",
                "Share indicators of compromise with international partners",
            ]
