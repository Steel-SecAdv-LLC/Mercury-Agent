# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury Agent - Hazard Diagnostics Visualization Routes.

Renders the hazard detectors' persisted intermediate arrays (spectrograms,
Doppler fields, thermal masks, wind/vorticity fields, spectra) into artifacts
over HTTP: deterministic PNG (``image/png``) or RFC 7946 GeoJSON
(``application/geo+json``). The same rendering path as the
``mercury-agent hazard-viz`` CLI and the ``mercury_hazard_visualize`` MCP tool.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from omni_mercury_engine.api.auth import APIKeyAuth, JWTAuth, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hazard", tags=["Hazard Visualization"])


class HazardVisualizeRequest(BaseModel):
    """Request for hazard diagnostics visualization.

    Provide EITHER ``diagnostics`` (a payload previously returned by a
    detector run -- the ``to_jsonable`` form) OR ``hazard`` + ``arrays`` (raw
    detector input; the named detector runs with diagnostics enabled).
    """

    hazard: str | None = Field(
        default=None,
        description=(
            "Hazard detector to run on 'arrays' (earthquake, tsunami, meteor, wildfire, "
            "tornado, hurricane, volcanic, landslide, schumann). Ignored when a prior "
            "'diagnostics' payload is supplied."
        ),
    )
    format: Literal["png", "geojson"] = Field(
        default="png",
        description="Artifact format: deterministic PNG or RFC 7946 GeoJSON.",
    )
    diagnostics: dict[str, Any] | None = Field(
        default=None,
        description="A prior HazardDiagnostics payload (hazard/arrays/context).",
    )
    arrays: dict[str, list[Any]] | None = Field(
        default=None,
        description="Raw detector input arrays (nested numeric lists), keyed by name.",
    )
    params: dict[str, float] = Field(
        default_factory=dict,
        description="Detector parameters (sampling_rate_hz, pixel_size_km, grid_spacing_m).",
    )
    geotransform: dict[str, float] | None = Field(
        default=None,
        description=(
            "Pixel->WGS84 mapping (origin_lon, origin_lat, deg_per_pixel_lon, "
            "deg_per_pixel_lat); REQUIRED for GeoJSON -- coordinates are never fabricated."
        ),
    )


def _get_optional_user(
    api_key_user: User | None = Depends(APIKeyAuth(auto_error=False)),
    jwt_user: User | None = Depends(JWTAuth(auto_error=False)),
) -> User | None:
    """Get current user if authenticated."""
    return api_key_user or jwt_user


def _resolve_diagnostics(request: HazardVisualizeRequest) -> Any:
    """Obtain the diagnostics payload: reuse a prior one or run the detector.

    Raises:
        ValueError: When neither (or both) input modes are supplied, or the
            inputs are invalid.
    """
    from omni_mercury_engine.detectors.hazard_diagnostics import (
        HazardDiagnostics,
        run_hazard_detector,
    )

    if request.diagnostics is not None and request.arrays is not None:
        raise ValueError("provide either 'diagnostics' or 'arrays', not both")
    if request.diagnostics is not None:
        return HazardDiagnostics.from_jsonable(request.diagnostics)
    if request.arrays is None:
        raise ValueError(
            "provide a prior 'diagnostics' payload, or 'hazard' + 'arrays' to run a detector"
        )
    if not request.hazard:
        raise ValueError("'hazard' is required when supplying raw 'arrays'")
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for name, value in request.arrays.items():
        try:
            arrays[name] = np.asarray(value, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"input array {name!r} is not numeric: {exc}") from exc
    return run_hazard_detector(request.hazard, arrays, params=dict(request.params))


@router.post(
    "/visualize",
    summary="Render Hazard Diagnostics",
    description="""
Render a hazard detector's persisted intermediate arrays into an artifact.

## Inputs
Either a prior ``diagnostics`` payload (as returned by a detector run with
``keep_diagnostics=True``, in its JSON form) or ``hazard`` + ``arrays`` raw
detector input -- the detector then runs server-side with diagnostics enabled.

## Outputs
- ``format=png``: a deterministic PNG (``image/png``) -- earthquake
  spectrogram + STA/LTA, tornado Doppler field + couplet, wildfire thermal
  map + hotspot mask, hurricane wind/vorticity fields (no track cone: the
  track model was removed as uncomputed), and 1-D spectra for
  tsunami/schumann/meteor plus score series for volcanic/landslide.
- ``format=geojson``: an RFC 7946 FeatureCollection (``application/geo+json``)
  of wildfire ignition hotspots. Requires a caller-supplied ``geotransform``
  (the detector works in pixel space and coordinates are never fabricated).
  Flood/landslide compute no zonal output and are rejected with the reason.
""",
)
async def visualize_hazard(
    request: HazardVisualizeRequest,
    user: User | None = Depends(_get_optional_user),
) -> Response:
    """Render hazard diagnostics to PNG or GeoJSON."""
    try:
        from omni_mercury_engine.detectors.hazard_visuals import (
            build_hazard_geojson,
            render_hazard_png,
        )

        diagnostics = _resolve_diagnostics(request)

        if request.format == "geojson":
            feature_collection = build_hazard_geojson(
                diagnostics, geotransform=request.geotransform
            )
            return Response(
                content=json.dumps(feature_collection),
                media_type="application/geo+json",
            )

        png = render_hazard_png(diagnostics)
        return Response(content=png, media_type="image/png")

    except ValueError as e:
        # Bad request: unknown hazard, malformed arrays, missing geotransform,
        # or a detector that computes no zonal output.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except ImportError as e:
        # Slim install: torch (detector run) or matplotlib (PNG) missing.
        logger.error("Hazard visualization stack unavailable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Hazard visualization failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during hazard visualization.",
        ) from e
