from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from walk_forward import apply_planner_amplifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unit test Planner event amplification logic.")
    parser.add_argument("--output", default="reports/planner_event_test/planner_event_test.json")
    parser.add_argument("--min-buy", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weak_buy = np.array([1.0, 0.08], dtype=np.float32)
    cases = [
        (
            "super_positive_amplifies",
            weak_buy,
            pd.Series(
                {
                    "event_super_positive_count": 1,
                    "event_super_negative_count": 0,
                    "event_priced_in_count": 0,
                    "event_strength_mean": 0.9,
                }
            ),
        ),
        (
            "priced_in_positive_not_amplified",
            weak_buy,
            pd.Series(
                {
                    "event_super_positive_count": 1,
                    "event_super_negative_count": 0,
                    "event_priced_in_count": 1,
                    "event_strength_mean": 0.9,
                }
            ),
        ),
        (
            "super_negative_holds",
            weak_buy,
            pd.Series(
                {
                    "event_super_positive_count": 0,
                    "event_super_negative_count": 1,
                    "event_priced_in_count": 0,
                    "event_strength_mean": -0.95,
                }
            ),
        ),
    ]

    results = []
    for name, action, row in cases:
        planned, reason = apply_planner_amplifier(
            action=action,
            market_row=row,
            enabled=True,
            min_buy_amount=args.min_buy,
        )
        flat = np.asarray(planned).reshape(-1)
        results.append(
            {
                "case": name,
                "input_action": action.tolist(),
                "planned_action": flat.tolist(),
                "planner_reason": reason,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in results:
        print(f"{item['case']}: {item['planner_reason']} -> {item['planned_action']}")


if __name__ == "__main__":
    main()
