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

"""FastAPI server for real-time anomaly detection.

Follows Azure AI Anomaly Detector best practices:
https://azure.microsoft.com/en-us/products/ai-services/ai-anomaly-detector
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np


app = FastAPI(
    title="OMNI ♱ AVA API",
    description="REST API for multi-domain anomaly detection",
    version="1.0.0",
)


class UnivariateRequest(BaseModel):
    """Request for univariate anomaly detection."""

    data: List[float]
    sensitivity: Optional[float] = 0.5


class MultivariateRequest(BaseModel):
    """Request for multivariate anomaly detection."""

    data: List[List[float]]
    features: Optional[List[str]] = None
    sensitivity: Optional[float] = 0.5


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/api/v1/detect/univariate")
async def detect_univariate(request: UnivariateRequest) -> Dict[str, Any]:
    """Detect anomalies in univariate time-series data.

    Args:
        request: Univariate detection request

    Returns:
        Detection results with anomalies and scores
    """
    try:
        data = np.array(request.data)

        threshold = 2.0 + (1.0 - request.sensitivity) * 3.0

        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / (std + 1e-8))

        anomalies = (z_scores > threshold).tolist()
        scores = z_scores.tolist()

        return {
            "anomalies": anomalies,
            "scores": scores,
            "method": "univariate",
            "threshold": threshold,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/detect/multivariate")
async def detect_multivariate(request: MultivariateRequest) -> Dict[str, Any]:
    """Detect anomalies in multivariate time-series data.

    Args:
        request: Multivariate detection request

    Returns:
        Detection results with anomalies and scores
    """
    try:
        data = np.array(request.data)

        if len(data.shape) != 2:
            raise HTTPException(status_code=400, detail="Data must be 2D array")

        threshold = 2.0 + (1.0 - request.sensitivity) * 3.0

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        z_scores = np.linalg.norm((data - mean) / std, axis=1)

        anomalies = (z_scores > threshold).tolist()
        scores = z_scores.tolist()

        return {
            "anomalies": anomalies,
            "scores": scores,
            "method": "multivariate",
            "threshold": threshold,
            "features": request.features or [f"feature_{i}" for i in range(data.shape[1])],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
