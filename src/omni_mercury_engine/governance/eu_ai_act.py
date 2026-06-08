# Copyright (C) 2025 Steel Security Advisors LLC
"""EU AI Act risk **tier gate / tag** -- by design never a scalar.

The EU AI Act (Regulation (EU) 2024/1689) classifies an AI system into a risk *tier* from
the **declared use case**: it is a categorical legal gate, not a runtime measurement.  This
module therefore deliberately produces a :class:`EuAiActTag` and **registers nothing** into
the GOSNN -- there is no ``*_scalar`` function here, and the family vets ``TAG_ONLY`` in
:data:`omni_mercury_engine.governance.contract.GOVERNANCE_FAMILY_VET`.

Tiers (Art. 5 / Annex III / Art. 50 / recital 'minimal risk'):

* ``unacceptable`` -- prohibited practice (Art. 5), e.g. social scoring, manipulative systems.
* ``high`` -- Annex III high-risk area (biometrics, critical infrastructure, employment, ...).
* ``limited`` -- transparency obligation only (Art. 50), e.g. chatbots, deepfakes.
* ``minimal`` -- everything else.

⚠️ NOT LEGAL ADVICE.  This is a declarative tag from operator-declared use-case flags for
audit/reporting; it is not a conformity assessment or a regulatory determination.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

_FAMILY = "eu_ai_act"


class EuAiActTier(Enum):
    """EU AI Act risk tier, in descending regulatory severity."""

    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


@dataclass(frozen=True)
class EuAiActTag:
    """A declarative EU AI Act tier tag -- a classification, never a registered scalar.

    Attributes:
        tier: The classified :class:`EuAiActTier`.
        family: Always ``"eu_ai_act"``.
        rationale: Which declared flag drove the tier.
        registers: Always ``False`` -- a tag never enters the operational/metric surface.
    """

    tier: EuAiActTier
    family: str
    rationale: str
    registers: bool = False


def eu_ai_act_tier(use_case: dict[str, object]) -> EuAiActTag:
    """Classify a declared AI use case into an EU AI Act risk tier (a tag, not a scalar).

    Args:
        use_case: Operator-declared flags.  Recognised (highest tier wins):
            ``prohibited_practice`` (Art. 5), ``annex_iii_high_risk`` (Annex III),
            ``transparency_obligation`` (Art. 50).  Anything else is ``minimal``.

    Returns:
        An :class:`EuAiActTag`.  It never registers a scalar -- ``tag.registers`` is ``False``.
    """
    if bool(use_case.get("prohibited_practice")):
        return EuAiActTag(
            EuAiActTier.UNACCEPTABLE, _FAMILY, "declared prohibited practice (Art. 5)"
        )
    if bool(use_case.get("annex_iii_high_risk")):
        return EuAiActTag(EuAiActTier.HIGH, _FAMILY, "declared Annex III high-risk area")
    if bool(use_case.get("transparency_obligation")):
        return EuAiActTag(EuAiActTier.LIMITED, _FAMILY, "declared Art. 50 transparency obligation")
    return EuAiActTag(EuAiActTier.MINIMAL, _FAMILY, "no prohibited/high-risk/transparency flag")
