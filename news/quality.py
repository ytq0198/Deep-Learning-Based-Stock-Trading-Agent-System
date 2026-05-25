from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_sentiment_quality_report(
    stock_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> dict[str, object]:
    if stock_df.empty:
        return {
            "trading_days": 0,
            "covered_days": 0,
            "coverage_ratio": 0.0,
            "news_count_total": 0,
            "warning": "stock data is empty",
        }

    stock_dates = set(stock_df["date"].astype(str))
    if sentiment_df.empty:
        covered_dates: set[str] = set()
        news_count_total = 0
        positive_days = 0
        negative_days = 0
    else:
        sentiment_df = sentiment_df.copy()
        sentiment_df["date"] = sentiment_df["date"].astype(str)
        covered_dates = set(sentiment_df.loc[sentiment_df["news_count"] > 0, "date"])
        covered_dates &= stock_dates
        news_count_total = int(sentiment_df.get("news_count", pd.Series(dtype=float)).sum())
        positive_days = int((sentiment_df.get("positive_count", 0) > 0).sum())
        negative_days = int((sentiment_df.get("negative_count", 0) > 0).sum())

    trading_days = len(stock_dates)
    covered_days = len(covered_dates)
    coverage_ratio = covered_days / trading_days if trading_days else 0.0
    warning = ""
    if coverage_ratio < 0.1:
        warning = "news coverage is very low; sentiment features may not affect training"
    elif coverage_ratio < 0.3:
        warning = "news coverage is limited; use results cautiously"

    return {
        "trading_days": trading_days,
        "covered_days": covered_days,
        "coverage_ratio": coverage_ratio,
        "news_count_total": news_count_total,
        "positive_days": positive_days,
        "negative_days": negative_days,
        "warning": warning,
    }


def save_quality_report(report: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
