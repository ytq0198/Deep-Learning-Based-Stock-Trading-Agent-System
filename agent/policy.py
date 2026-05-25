from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rlenv.StockTradingEnv0 import (
    FEATURE_SCALES,
    MAX_ACCOUNT_BALANCE,
    MAX_NUM_SHARES,
    MAX_SHARE_PRICE,
)

from .interfaces import AccountState, MarketState, PolicyAction


class PPOPolicyModel:
    """Wraps a trained stable-baselines3 PPO model for the Agent layer."""

    def __init__(self, model_path: str | Path) -> None:
        self.model = PPO.load(str(model_path))

    def predict(self, market: MarketState, account: AccountState) -> PolicyAction:
        obs = build_observation(market, account)
        raw_action, _ = self.model.predict(obs, deterministic=True)
        raw_action = np.asarray(raw_action, dtype=np.float32).flatten()
        action_type_value = float(np.clip(raw_action[0], 1.0, 3.0))
        amount = float(np.clip(raw_action[1], 0.0, 1.0))

        if action_type_value < 1.5:
            action_type = "buy"
        elif action_type_value < 2.5:
            action_type = "sell"
        else:
            action_type = "hold"

        return PolicyAction(
            action_type=action_type,
            amount=amount,
            source="ppo",
            raw_action=raw_action.tolist(),
        )


def build_observation(market: MarketState, account: AccountState) -> np.ndarray:
    values = {
        "open": market.open,
        "high": market.high,
        "low": market.low,
        "close": market.close,
        "volume": market.volume,
        **market.indicators,
    }

    market_features = [
        _scaled(values.get(column, 0.0), scale)
        for column, scale in FEATURE_SCALES.items()
    ]
    account_features = [
        _scaled(account.cash, MAX_ACCOUNT_BALANCE),
        _scaled(account.max_net_worth, MAX_ACCOUNT_BALANCE),
        _scaled(account.shares, MAX_NUM_SHARES),
        _scaled(account.cost_basis, MAX_SHARE_PRICE),
        0.0,
        0.0,
    ]
    return np.array(market_features + account_features, dtype=np.float32)


def _scaled(value: float, scale: float) -> float:
    return float(np.clip(float(value) / scale, -1.0, 1.0))
