from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

INITIAL_ACCOUNT_BALANCE = 10_000.0
MAX_ACCOUNT_BALANCE = 2_147_483_647.0
MAX_NUM_SHARES = 2_147_483_647.0
MAX_SHARE_PRICE = 5_000.0
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10

FEATURE_SCALES = {
    "open": MAX_SHARE_PRICE,
    "high": MAX_SHARE_PRICE,
    "low": MAX_SHARE_PRICE,
    "close": MAX_SHARE_PRICE,
    "preclose": MAX_SHARE_PRICE,
    "volume": MAX_VOLUME,
    "amount": MAX_AMOUNT,
    "adjustflag": 10.0,
    "turn": 100.0,
    "tradestatus": 1.0,
    "pctChg": 100.0,
    "peTTM": 1e4,
    "pbMRQ": 100.0,
    "psTTM": 100.0,
    "pcfNcfTTM": 100.0,
    "isST": 1.0,
    "news_count": 50.0,
    "sentiment_mean": 1.0,
    "sentiment_max": 1.0,
    "sentiment_min": 1.0,
    "positive_count": 50.0,
    "negative_count": 50.0,
    "announcement_score": 5.0,
    "social_heat": 10.0,
    "pre_3d_return": 1.0,
    "pre_5d_return": 1.0,
    "pre_20d_return": 1.0,
    "volume_spike": 10.0,
    "event_count": 50.0,
    "event_positive_count": 50.0,
    "event_negative_count": 50.0,
    "event_risk_count": 50.0,
    "event_confidence_mean": 1.0,
    "event_duration_mean": 30.0,
    "regime_code": 10.0,
    "regime_0": 1.0,
    "regime_1": 1.0,
    "regime_2": 1.0,
    "regime_3": 1.0,
    "return_1d": 1.0,
    "return_5d": 1.0,
    "volatility_5d": 1.0,
    "volatility_20d": 1.0,
    "drawdown_20d": 1.0,
    "event_strength_mean": 1.0,
    "event_strength_max": 1.0,
    "event_strength_min": 1.0,
    "event_super_positive_count": 50.0,
    "event_super_negative_count": 50.0,
    "event_priced_in_count": 50.0,
    "priced_in_score": 1.0,
    "is_priced_in": 1.0,
}

FEATURE_SETS = {
    "full": list(FEATURE_SCALES.keys()),
    "minimal_signal": [
        "pre_5d_return",
        "volume_spike",
        "social_heat",
        "event_risk_count",
        "regime_code",
    ],
    "price_only": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turn",
        "pctChg",
        "pre_3d_return",
        "pre_5d_return",
        "pre_20d_return",
        "volume_spike",
    ],
}

ACCOUNT_FEATURE_COUNT = 6


class StockTradingEnv(gym.Env):
    """Single-stock trading environment compatible with stable-baselines3."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = INITIAL_ACCOUNT_BALANCE,
        random_price: bool = False,
        reward_strategy: str = "profit_sign",
        fee_rate: float = 0.0003,
        min_fee: float = 5.0,
        slippage_rate: float = 0.0005,
        lot_size: int = 100,
        max_position_ratio: float = 0.8,
        min_trade_ratio: float = 0.1,
        feature_set: str = "full",
        opportunity_cost_penalty: float = 0.0,
        opportunity_return_threshold: float = 0.015,
    ) -> None:
        super().__init__()
        self.initial_balance = float(initial_balance)
        self.random_price = random_price
        self.reward_strategy = reward_strategy
        self.fee_rate = fee_rate
        self.min_fee = min_fee
        self.slippage_rate = slippage_rate
        self.lot_size = lot_size
        self.max_position_ratio = max_position_ratio
        self.min_trade_ratio = min_trade_ratio
        self.feature_set = feature_set
        self.feature_columns = FEATURE_SETS.get(feature_set, FEATURE_SETS["full"])
        self.opportunity_cost_penalty = opportunity_cost_penalty
        self.opportunity_return_threshold = opportunity_return_threshold
        self.df = self._prepare_dataframe(df)

        # action[0]: 1=buy, 2=sell, 3=hold; action[1]: position percentage.
        self.action_space = spaces.Box(
            low=np.array([1.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_size = len(self.feature_columns) + ACCOUNT_FEATURE_COUNT
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "date" in df.columns:
            df = df.sort_values("date")

        required_columns = {"open", "high", "low", "close"}
        missing_required = required_columns - set(df.columns)
        if missing_required:
            raise ValueError(f"Missing required columns: {sorted(missing_required)}")

        for column in FEATURE_SCALES:
            if column not in df.columns:
                df[column] = 0
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

        df = df.reset_index(drop=True)
        if len(df) < 2:
            raise ValueError("StockTradingEnv needs at least 2 rows of stock data.")
        return df

    @staticmethod
    def _scaled(value: Any, scale: float) -> float:
        value = float(value) if np.isfinite(value) else 0.0
        return float(np.clip(value / scale, -1.0, 1.0))

    def _current_close(self) -> float:
        price = float(self.df.loc[self.current_step, "close"])
        return max(price, 0.01)

    def _next_observation(self) -> np.ndarray:
        row = self.df.loc[self.current_step]
        market_features = [
            self._scaled(row[column], FEATURE_SCALES[column])
            for column in self.feature_columns
        ]
        account_features = [
            self._scaled(self.balance, MAX_ACCOUNT_BALANCE),
            self._scaled(self.max_net_worth, MAX_ACCOUNT_BALANCE),
            self._scaled(self.shares_held, MAX_NUM_SHARES),
            self._scaled(self.cost_basis, MAX_SHARE_PRICE),
            self._scaled(self.total_shares_sold, MAX_NUM_SHARES),
            self._scaled(
                self.total_sales_value,
                MAX_NUM_SHARES * MAX_SHARE_PRICE,
            ),
        ]
        return np.array(market_features + account_features, dtype=np.float32)

    def _take_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).flatten()
        action_type = float(np.clip(action[0], 1.0, 3.0))
        amount = float(np.clip(action[1], 0.0, 1.0))
        self.last_action_type = action_type
        if action_type < 2.5:
            amount = max(amount, self.min_trade_ratio)

        if self.random_price:
            current_price = random.uniform(
                float(self.df.loc[self.current_step, "open"]),
                float(self.df.loc[self.current_step, "close"]),
            )
            current_price = max(current_price, 0.01)
        else:
            current_price = self._current_close()

        self.last_fees = 0.0
        if action_type < 1.5:
            fill_price = current_price * (1 + self.slippage_rate)
            current_position_value = self.shares_held * current_price
            max_position_value = self.net_worth * self.max_position_ratio
            allowed_position_value = max(0.0, max_position_value - current_position_value)
            budget = min(self.balance * amount, allowed_position_value)
            total_possible = int(budget / fill_price)
            shares_bought = self._round_lot(total_possible)
            additional_cost = shares_bought * fill_price
            fees = self._fee(additional_cost)
            while shares_bought > 0 and additional_cost + fees > self.balance:
                shares_bought = self._round_lot(shares_bought - self.lot_size)
                additional_cost = shares_bought * fill_price
                fees = self._fee(additional_cost)
            if shares_bought > 0:
                previous_cost = self.cost_basis * self.shares_held
                self.balance -= additional_cost + fees
                self.shares_held += shares_bought
                self.cost_basis = (
                    previous_cost + additional_cost
                ) / self.shares_held
                self.last_fees = fees
        elif action_type < 2.5:
            fill_price = current_price * (1 - self.slippage_rate)
            shares_sold = self._round_lot(int(self.shares_held * amount))
            shares_sold = min(self.shares_held, shares_sold)
            sales_value = shares_sold * fill_price
            fees = self._fee(sales_value)
            if shares_sold > 0:
                self.balance += sales_value - fees
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += sales_value
                self.last_fees = fees

        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        if self.shares_held == 0:
            self.cost_basis = 0.0

    def _round_lot(self, shares: int) -> int:
        if self.lot_size <= 1:
            return max(0, shares)
        return max(0, shares // self.lot_size * self.lot_size)

    def _fee(self, trade_value: float) -> float:
        if trade_value <= 0:
            return 0.0
        return max(self.min_fee, trade_value * self.fee_rate)

    def _calculate_reward(self, previous_net_worth: float) -> float:
        profit = self.net_worth - self.initial_balance
        if self.reward_strategy == "profit_delta":
            reward = float((self.net_worth - previous_net_worth) / self.initial_balance)
        else:
            reward = 1.0 if profit > 0 else -100.0

        if self._missed_upside():
            reward -= self.opportunity_cost_penalty
        return reward

    def _missed_upside(self) -> bool:
        if self.opportunity_cost_penalty <= 0:
            return False
        if getattr(self, "last_action_type", 3.0) < 2.5:
            return False
        if self.current_step <= 0:
            return False
        previous_close = float(self.df.loc[self.current_step - 1, "close"])
        current_close = float(self.df.loc[self.current_step, "close"])
        if previous_close <= 0:
            return False
        next_return = current_close / previous_close - 1
        return next_return >= self.opportunity_return_threshold

    def step(self, action: np.ndarray):
        previous_net_worth = self.net_worth
        self._take_action(action)

        self.current_step += 1
        terminated = self.net_worth <= 0
        truncated = self.current_step >= len(self.df) - 1
        if truncated:
            self.current_step = len(self.df) - 1

        reward = self._calculate_reward(previous_net_worth)
        obs = self._next_observation()
        info = {
            "balance": self.balance,
            "shares_held": self.shares_held,
            "net_worth": self.net_worth,
            "profit": self.net_worth - self.initial_balance,
            "fees": self.last_fees,
        }
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0.0
        self.total_shares_sold = 0
        self.total_sales_value = 0.0
        self.last_fees = 0.0
        self.last_action_type = 3.0
        self.current_step = 0
        return self._next_observation(), {}

    def render(self):
        profit = self.net_worth - self.initial_balance
        print("-" * 30)
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Shares held: {self.shares_held}")
        print(f"Avg cost for held shares: {self.cost_basis:.2f}")
        print(f"Net worth: {self.net_worth:.2f}")
        print(f"Profit: {profit:.2f}")
        return profit
