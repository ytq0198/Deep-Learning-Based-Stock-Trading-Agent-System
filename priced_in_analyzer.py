from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PricedInResult:
    is_priced_in: bool
    priced_in_score: float
    reason: str


class EventPricedInAnalyzer:
    """Rule-based price-in analyzer for event-driven trading."""

    def __init__(
        self,
        strong_pre_5d_return: float = 0.04,
        strong_pre_20d_return: float = 0.08,
        high_volume_spike: float = 1.8,
        high_turn: float = 1.0,
    ) -> None:
        self.strong_pre_5d_return = strong_pre_5d_return
        self.strong_pre_20d_return = strong_pre_20d_return
        self.high_volume_spike = high_volume_spike
        self.high_turn = high_turn

    def analyze(self, row: pd.Series) -> PricedInResult:
        pre_5d_return = _float(row.get("pre_5d_return", row.get("return_5d", 0.0)))
        pre_20d_return = _float(row.get("pre_20d_return", 0.0))
        volume_spike = _float(row.get("volume_spike", 0.0))
        turn = _float(row.get("turn", 0.0))
        event_strength = _float(row.get("event_strength_mean", row.get("event_strength", 0.0)))

        score = 0.0
        reasons: list[str] = []
        if event_strength > 0:
            if pre_5d_return >= self.strong_pre_5d_return:
                score += 0.35
                reasons.append("strong_pre_5d_rise")
            if pre_20d_return >= self.strong_pre_20d_return:
                score += 0.35
                reasons.append("strong_pre_20d_rise")
            if volume_spike >= self.high_volume_spike:
                score += 0.2
                reasons.append("volume_spike")
            if turn >= self.high_turn:
                score += 0.1
                reasons.append("high_turn")
        elif event_strength < 0:
            if pre_5d_return <= -self.strong_pre_5d_return:
                score += 0.25
                reasons.append("negative_event_after_drop")

        score = max(0.0, min(1.0, score))
        return PricedInResult(
            is_priced_in=score >= 0.5,
            priced_in_score=score,
            reason="|".join(reasons) if reasons else "not_priced_in",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate event features with price-in scores.")
    parser.add_argument("--feature-csv", default="newsdata/guba_strong_event_features.csv")
    parser.add_argument("--stock-csv", action="append", default=[])
    parser.add_argument("--output", default="newsdata/guba_strong_event_features_priced.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.feature_csv)
    stock_paths = args.stock_csv or ["stockdata_recent_stage10_event/train/sh.600036.sh.600036.csv"]
    stock = pd.concat([pd.read_csv(path) for path in stock_paths], ignore_index=True)
    stock = stock.drop_duplicates(subset=["date", "code"])
    merged = features.merge(
        stock[["date", "code", "pre_5d_return", "pre_20d_return", "volume_spike", "turn"]],
        how="left",
        on=["date", "code"],
    )
    analyzer = EventPricedInAnalyzer()
    rows = []
    for _, row in merged.iterrows():
        result = analyzer.analyze(row)
        rows.append(
            {
                "date": row["date"],
                "code": row["code"],
                "is_priced_in": int(result.is_priced_in),
                "priced_in_score": result.priced_in_score,
                "priced_in_reason": result.reason,
            }
        )
    priced = pd.DataFrame(rows)
    output = features.merge(priced, how="left", on=["date", "code"])
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved priced-in features to {args.output}")


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
