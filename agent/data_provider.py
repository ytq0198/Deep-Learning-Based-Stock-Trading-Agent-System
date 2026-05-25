from __future__ import annotations

from pathlib import Path

import pandas as pd

from .interfaces import MarketState


class CSVDataProvider:
    """Loads local CSV market data produced by get_stock_data.py."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def find_file(self, code: str) -> Path:
        for file_path in self.data_dir.glob("*.csv"):
            if code in file_path.name:
                return file_path
        raise FileNotFoundError(f"No CSV found for {code} in {self.data_dir}")

    def load_history(self, code: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        df = pd.read_csv(self.find_file(code))
        if "date" in df.columns:
            df = df.sort_values("date")
            if start:
                df = df[df["date"] >= start]
            if end:
                df = df[df["date"] <= end]
        return df.reset_index(drop=True)

    def get_latest_state(self, code: str) -> MarketState:
        df = self.load_history(code)
        return row_to_market_state(df.iloc[-1], code)


def row_to_market_state(row: pd.Series, default_code: str) -> MarketState:
    indicators: dict[str, float] = {}
    for key, value in row.items():
        if key in {"date", "code", "open", "high", "low", "close", "volume"}:
            continue
        indicators[key] = _to_float(value)

    return MarketState(
        date=str(row.get("date", "")),
        code=str(row.get("code", default_code)),
        open=_to_float(row.get("open", 0)),
        high=_to_float(row.get("high", 0)),
        low=_to_float(row.get("low", 0)),
        close=_to_float(row.get("close", 0)),
        volume=_to_float(row.get("volume", 0)),
        indicators=indicators,
    )


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
