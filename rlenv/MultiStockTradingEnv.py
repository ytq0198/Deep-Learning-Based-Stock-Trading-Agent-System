from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import pandas as pd

from .StockTradingEnv0 import StockTradingEnv


class MultiStockTradingEnv(gym.Env):
    """Sample a random stock and contiguous date window for each episode.

    This expands effective training samples from ~60 single-stock days to
    train_days * n_stocks cross-sectional observations.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        panel_df: pd.DataFrame,
        episode_length: int = 30,
        seed: int | None = None,
        **env_kwargs,
    ) -> None:
        super().__init__()
        if "code" not in panel_df.columns:
            raise ValueError("MultiStockTradingEnv requires a code column.")
        self.panel = panel_df.sort_values(["code", "date"]).reset_index(drop=True)
        self.episode_length = episode_length
        self.env_kwargs = env_kwargs
        self._rng = random.Random(seed)
        self._eligible: dict[str, pd.DataFrame] = {}
        self._build_eligible()
        bootstrap = StockTradingEnv(self._sample_slice(), **self.env_kwargs)
        self.action_space = bootstrap.action_space
        self.observation_space = bootstrap.observation_space
        self._env: StockTradingEnv | None = None

    def _build_eligible(self) -> None:
        min_rows = self.episode_length + 2
        for code, group in self.panel.groupby("code"):
            frame = group.reset_index(drop=True)
            if len(frame) >= min_rows:
                self._eligible[str(code)] = frame
        if not self._eligible:
            raise ValueError("No stock has enough rows for the requested episode_length.")

    def _sample_slice(self) -> pd.DataFrame:
        code = self._rng.choice(list(self._eligible.keys()))
        frame = self._eligible[code]
        max_start = len(frame) - self.episode_length - 2
        start = self._rng.randint(0, max_start)
        end = start + self.episode_length + 2
        return frame.iloc[start:end].reset_index(drop=True)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        slice_df = self._sample_slice()
        self._env = StockTradingEnv(slice_df, **self.env_kwargs)
        obs, info = self._env.reset(seed=seed, options=options)
        info["sample_code"] = str(slice_df["code"].iloc[0])
        info["sample_start"] = str(slice_df["date"].iloc[0])
        info["sample_end"] = str(slice_df["date"].iloc[-1])
        return obs, info

    def step(self, action):
        if self._env is None:
            raise RuntimeError("Environment not reset.")
        return self._env.step(action)

    def render(self):
        if self._env is not None:
            self._env.render()
