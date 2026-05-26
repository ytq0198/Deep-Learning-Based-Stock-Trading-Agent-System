from __future__ import annotations

import numpy as np
import pandas as pd

from priced_in_analyzer import EventPricedInAnalyzer


class RuleEventStrategy:
    """Pure rule strategy: trade only on strong events, otherwise hold.

    Used as a baseline to verify whether announcement/event features contain edge
    before relying on PPO to learn them.
    """

    def __init__(
        self,
        buy_amount: float = 0.6,
        force_sell_on_negative: bool = True,
        use_planner_priced_in: bool = False,
    ) -> None:
        self.buy_amount = buy_amount
        self.force_sell_on_negative = force_sell_on_negative
        self.use_planner_priced_in = use_planner_priced_in
        self.priced_in_analyzer = EventPricedInAnalyzer()

    def decide(self, market_row: pd.Series) -> tuple[np.ndarray, str]:
        super_positive = float(market_row.get("event_super_positive_count", 0.0))
        super_negative = float(market_row.get("event_super_negative_count", 0.0))
        event_strength = float(market_row.get("event_strength_mean", 0.0))
        priced_in_count = float(market_row.get("event_priced_in_count", 0.0))
        priced_in_score = float(market_row.get("priced_in_score", 0.0))

        if self.use_planner_priced_in and super_positive > 0:
            priced_in_result = self.priced_in_analyzer.analyze(market_row)
            priced_in_score = max(priced_in_score, priced_in_result.priced_in_score)
            if priced_in_result.is_priced_in:
                priced_in_count = max(priced_in_count, 1.0)

        if super_negative > 0 or event_strength <= -0.75:
            if self.force_sell_on_negative:
                return _action(2.0, 1.0), "super_negative_sell"
            return _action(3.0, 0.0), "super_negative_hold"

        if super_positive > 0 and priced_in_count <= 0 and priced_in_score < 0.5:
            return _action(1.0, self.buy_amount), "super_positive_buy"

        if super_positive > 0 and (priced_in_count > 0 or priced_in_score >= 0.5):
            return _action(3.0, 0.0), "positive_priced_in_hold"

        if event_strength >= 0.65 and priced_in_count <= 0 and priced_in_score < 0.5:
            return _action(1.0, min(self.buy_amount, 0.4)), "strong_positive_buy"

        return _action(3.0, 0.0), "no_event_hold"


def _action(action_type: float, amount: float) -> np.ndarray:
    return np.array([[action_type, amount]], dtype=np.float32)
