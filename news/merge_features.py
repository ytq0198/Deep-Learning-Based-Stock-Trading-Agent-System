from __future__ import annotations

from pathlib import Path

import pandas as pd


SENTIMENT_COLUMNS = [
    "news_count",
    "sentiment_mean",
    "sentiment_max",
    "sentiment_min",
    "positive_count",
    "negative_count",
    "announcement_score",
    "social_heat",
    "event_count",
    "event_positive_count",
    "event_negative_count",
    "event_risk_count",
    "event_confidence_mean",
    "event_duration_mean",
    "event_strength_mean",
    "event_strength_max",
    "event_strength_min",
    "event_super_positive_count",
    "event_super_negative_count",
    "event_priced_in_count",
    "priced_in_score",
    "is_priced_in",
]


def merge_sentiment_features(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    stock_df = stock_df.copy()
    if "date" not in stock_df.columns:
        raise ValueError("stock_df must include a date column")

    if sentiment_df.empty:
        for column in SENTIMENT_COLUMNS:
            stock_df[column] = 0.0
        return _add_price_context(stock_df)

    merge_columns = ["date"]
    if "code" in stock_df.columns and "code" in sentiment_df.columns:
        merge_columns.append("code")

    merged = stock_df.merge(sentiment_df, how="left", on=merge_columns)
    for column in SENTIMENT_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return _add_price_context(merged)


def merge_directory(
    stock_dir: str | Path,
    sentiment_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    stock_dir = Path(stock_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stock_file in stock_dir.glob("*.csv"):
        stock_df = pd.read_csv(stock_file)
        merged = merge_sentiment_features(stock_df, sentiment_df)
        merged.to_csv(output_dir / stock_file.name, index=False, encoding="utf-8-sig")


def _add_price_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "close" not in df.columns:
        return df

    close = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0.0)
    for window in [3, 5, 20]:
        df[f"pre_{window}d_return"] = close.pct_change(window).fillna(0.0)
    volume = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
    rolling_volume = volume.rolling(window=5, min_periods=1).mean().replace(0, 1.0)
    df["volume_spike"] = (volume / rolling_volume).fillna(0.0)
    return df
