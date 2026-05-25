from __future__ import annotations

import numpy as np
import pandas as pd

from .interfaces import AccountState, MarketState, PolicyAction, Scenario


class HistoricalBootstrapWorldModel:
    """Lightweight world model based on bootstrapped historical returns."""

    def __init__(self, history: pd.DataFrame, seed: int = 42) -> None:
        close = pd.to_numeric(history["close"], errors="coerce").dropna()
        returns = close.pct_change().dropna()
        self.returns = returns.to_numpy(dtype=np.float64)
        if len(self.returns) == 0:
            self.returns = np.array([0.0], dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        market: MarketState,
        account: AccountState,
        candidates: list[PolicyAction],
        horizon: int,
    ) -> list[Scenario]:
        scenarios: list[Scenario] = []
        for candidate in candidates:
            sampled = self.rng.choice(self.returns, size=max(horizon, 1), replace=True)
            exposure = _action_exposure(candidate, account, market.close)
            path_returns = sampled * exposure
            equity_curve = account.net_worth * np.cumprod(1 + path_returns)
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (running_max - equity_curve) / np.maximum(running_max, 1e-9)

            scenarios.append(
                Scenario(
                    name=f"{candidate.action_type}_{candidate.amount:.2f}",
                    horizon=horizon,
                    expected_return=float(equity_curve[-1] / account.net_worth - 1),
                    worst_case_return=float(np.min(path_returns)),
                    max_drawdown=float(np.max(drawdowns)),
                    path=[],
                )
            )
        return scenarios


def _action_exposure(action: PolicyAction, account: AccountState, price: float) -> float:
    current_position = account.shares * price
    if account.net_worth <= 0:
        return 0.0

    current_exposure = current_position / account.net_worth
    if action.action_type == "buy":
        buy_value = account.cash * action.amount
        return min(1.0, (current_position + buy_value) / account.net_worth)
    if action.action_type == "sell":
        return max(0.0, current_exposure * (1 - action.amount))
    return current_exposure
