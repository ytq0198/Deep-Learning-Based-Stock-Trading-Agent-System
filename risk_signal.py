from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskSignal:
    score: float
    allow_buy: bool
    reason: str


@dataclass(frozen=True)
class RiskSignalConfig:
    max_risk_score_to_buy: float = 0.55
    high_social_heat: float = 2.0
    high_volume_spike: float = 1.5
    severe_5d_drop: float = -0.03
    positive_5d_momentum: float = 0.02


class RiskSignalScorer:
    """Scores risk from the selected minimal_signal feature set."""

    def __init__(self, config: RiskSignalConfig | None = None) -> None:
        self.config = config or RiskSignalConfig()

    def score(self, row: pd.Series) -> RiskSignal:
        pre_5d_return = _float(row.get("pre_5d_return", row.get("return_5d", 0.0)))
        volume_spike = _float(row.get("volume_spike", 0.0))
        social_heat = _float(row.get("social_heat", 0.0))
        event_risk_count = _float(row.get("event_risk_count", 0.0))
        regime_code = int(_float(row.get("regime_code", -1)))

        score = 0.0
        reasons: list[str] = []

        if social_heat >= self.config.high_social_heat:
            score += 0.25
            reasons.append("high_social_heat")
        if volume_spike >= self.config.high_volume_spike:
            score += 0.2
            reasons.append("high_volume_spike")
        if event_risk_count > 0:
            score += min(0.35, 0.15 * event_risk_count)
            reasons.append("event_risk")
        if pre_5d_return <= self.config.severe_5d_drop:
            score += 0.2
            reasons.append("recent_drop")
        elif pre_5d_return >= self.config.positive_5d_momentum and event_risk_count == 0:
            score -= 0.1
            reasons.append("positive_momentum")

        if regime_code >= 0:
            # Regime codes are unsupervised; only use them as mild context here.
            if regime_code == 0 and event_risk_count > 0:
                score += 0.1
                reasons.append("risk_regime_context")

        score = max(0.0, min(1.0, score))
        allow_buy = score <= self.config.max_risk_score_to_buy
        reason = "pass" if allow_buy else "risk_score_block"
        if reasons:
            reason = f"{reason}:{'|'.join(reasons)}"
        return RiskSignal(score=score, allow_buy=allow_buy, reason=reason)


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
