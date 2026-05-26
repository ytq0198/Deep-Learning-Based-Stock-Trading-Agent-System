from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CalendarWalkForwardWindow:
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_days: int
    test_days: int


class CalendarPanelProvider:
    """Rolling walk-forward on trading dates across a multi-stock panel."""

    def __init__(
        self,
        panel: pd.DataFrame,
        train_days: int = 120,
        test_days: int = 5,
        step_days: int = 5,
    ) -> None:
        if "date" not in panel.columns:
            raise ValueError("panel must include date")
        self.panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
        self.dates = sorted(self.panel["date"].astype(str).unique())
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

    def windows(self):
        start = 0
        while start + self.train_days + self.test_days <= len(self.dates):
            train_dates = self.dates[start : start + self.train_days]
            test_dates = self.dates[start + self.train_days : start + self.train_days + self.test_days]
            train = self.panel[self.panel["date"].isin(train_dates)].reset_index(drop=True)
            test = self.panel[self.panel["date"].isin(test_dates)].reset_index(drop=True)
            yield CalendarWalkForwardWindow(
                train=train,
                test=test,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                train_days=len(train_dates),
                test_days=len(test_dates),
            )
            start += self.step_days
