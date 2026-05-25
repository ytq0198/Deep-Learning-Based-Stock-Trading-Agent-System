from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from .event_extractor import EventResult
from .sentiment import SentimentResult


def build_daily_sentiment_features(
    results: list[SentimentResult],
    shift_to_next_day: bool = True,
    event_results: list[EventResult] | None = None,
) -> pd.DataFrame:
    """Aggregate scored news into daily stock features.

    `shift_to_next_day=True` avoids look-ahead bias: today's collected news becomes
    available to the trading model on the next calendar day.
    """

    if not results:
        sentiment_features = _empty_features()
    else:
        rows = []
        for result in results:
            feature_date = _next_day(result.date) if shift_to_next_day else result.date
            rows.append(
                {
                    "date": feature_date,
                    "code": result.code,
                    "score": result.score,
                    "positive": 1 if result.impact == "positive" else 0,
                    "negative": 1 if result.impact == "negative" else 0,
                    "announcement_score": result.score
                    if _looks_like_announcement(result.title)
                    else 0.0,
                }
            )

        frame = pd.DataFrame(rows)
        sentiment_features = frame.groupby(["date", "code"], as_index=False).agg(
            news_count=("score", "count"),
            sentiment_mean=("score", "mean"),
            sentiment_max=("score", "max"),
            sentiment_min=("score", "min"),
            positive_count=("positive", "sum"),
            negative_count=("negative", "sum"),
            announcement_score=("announcement_score", "sum"),
        )
    sentiment_features["social_heat"] = sentiment_features["news_count"] / sentiment_features["news_count"].rolling(
        window=5,
        min_periods=1,
    ).mean()

    event_features = build_daily_event_features(event_results or [], shift_to_next_day)
    merged = sentiment_features.merge(event_features, how="outer", on=["date", "code"])
    return _ensure_feature_columns(merged).fillna(0)


def build_daily_event_features(
    event_results: list[EventResult],
    shift_to_next_day: bool = True,
) -> pd.DataFrame:
    if not event_results:
        return _empty_event_features()

    rows = []
    for event in event_results:
        feature_date = _next_day(event.date) if shift_to_next_day else event.date
        rows.append(
            {
                "date": feature_date,
                "code": event.code,
                "event_count": 1,
                "event_positive_count": 1 if event.impact_direction == "positive" else 0,
                "event_negative_count": 1 if event.impact_direction == "negative" else 0,
                "event_risk_count": 1
                if event.event_type in {"investigation_penalty", "industry_risk", "market_panic"}
                else 0,
                "event_confidence": event.confidence_score,
                "event_duration": event.impact_duration_days,
                "event_strength": event.event_strength,
                "event_super_positive": 1
                if event.is_super_event and event.impact_direction == "positive"
                else 0,
                "event_super_negative": 1
                if event.is_super_event and event.impact_direction == "negative"
                else 0,
                "event_priced_in": 1 if event.is_priced_in else 0,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.groupby(["date", "code"], as_index=False).agg(
        event_count=("event_count", "sum"),
        event_positive_count=("event_positive_count", "sum"),
        event_negative_count=("event_negative_count", "sum"),
        event_risk_count=("event_risk_count", "sum"),
        event_confidence_mean=("event_confidence", "mean"),
        event_duration_mean=("event_duration", "mean"),
        event_strength_mean=("event_strength", "mean"),
        event_strength_max=("event_strength", "max"),
        event_strength_min=("event_strength", "min"),
        event_super_positive_count=("event_super_positive", "sum"),
        event_super_negative_count=("event_super_negative", "sum"),
        event_priced_in_count=("event_priced_in", "sum"),
    )


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "code",
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
        ]
    )


def _empty_event_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "code",
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
        ]
    )


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    empty = _empty_features()
    for column in empty.columns:
        if column not in df.columns:
            df[column] = 0
    return df[empty.columns]


def _next_day(value: str) -> str:
    parsed = datetime.fromisoformat(value).date()
    return (parsed + timedelta(days=1)).isoformat()


def _looks_like_announcement(title: str) -> bool:
    return any(keyword in title for keyword in ["公告", "年报", "季报", "业绩", "分红", "回购", "重组"])
