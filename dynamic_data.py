from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class DynamicDataProvider:
    """Creates rolling train/test windows from one chronological stock dataframe."""

    def __init__(
        self,
        df: pd.DataFrame,
        train_window: int = 60,
        test_window: int = 1,
        step_size: int = 1,
    ) -> None:
        if "date" not in df.columns:
            raise ValueError("DynamicDataProvider requires a date column.")
        self.df = df.sort_values("date").drop_duplicates(subset=["date", "code"]).reset_index(drop=True)
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def windows(self):
        start = 0
        while start + self.train_window + self.test_window <= len(self.df):
            train = self.df.iloc[start : start + self.train_window].reset_index(drop=True)
            test = self.df.iloc[
                start + self.train_window : start + self.train_window + self.test_window
            ].reset_index(drop=True)
            if len(test) == 1 and start + self.train_window + 1 < len(self.df):
                # Gym environments need at least two rows to advance one step.
                test = self.df.iloc[start + self.train_window : start + self.train_window + 2].reset_index(drop=True)
            yield WalkForwardWindow(
                train=train,
                test=test,
                train_start=str(train["date"].iloc[0]),
                train_end=str(train["date"].iloc[-1]),
                test_start=str(test["date"].iloc[0]),
                test_end=str(test["date"].iloc[-1]),
            )
            start += self.step_size
