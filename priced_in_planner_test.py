from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from priced_in_analyzer import EventPricedInAnalyzer
from walk_forward import apply_planner_amplifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify priced-in positive events do not trigger Planner amplification.")
    parser.add_argument("--output", default="reports/priced_in_planner_test/result.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = pd.Series(
        {
            "event_super_positive_count": 1,
            "event_super_negative_count": 0,
            "event_strength_mean": 0.9,
            "pre_5d_return": 0.065,
            "pre_20d_return": 0.12,
            "volume_spike": 2.2,
            "turn": 1.3,
        }
    )
    priced = EventPricedInAnalyzer().analyze(row)
    row["priced_in_score"] = priced.priced_in_score
    row["event_priced_in_count"] = 1 if priced.is_priced_in else 0

    weak_buy = np.array([1.0, 0.08], dtype=np.float32)
    planned, reason = apply_planner_amplifier(
        action=weak_buy,
        market_row=row,
        enabled=True,
        min_buy_amount=0.5,
    )
    result = {
        "priced_in": {
            "is_priced_in": priced.is_priced_in,
            "priced_in_score": priced.priced_in_score,
            "reason": priced.reason,
        },
        "input_action": weak_buy.tolist(),
        "planned_action": np.asarray(planned).reshape(-1).tolist(),
        "planner_reason": reason,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
