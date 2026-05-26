from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from news.merge_features import _add_price_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-sectional panel dataset from per-stock enriched CSVs.")
    parser.add_argument("--input-dir", default="stockdata_bank_enriched/per_stock")
    parser.add_argument("--output-root", default="stockdata_bank_panel")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    frames = [pd.read_csv(path) for path in sorted(input_dir.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No enriched CSV files found in {input_dir}")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = panel["date"].astype(str)
    if "code" not in panel.columns:
        raise ValueError("Panel requires a code column in per-stock files.")

    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    panel = _add_price_context_by_code(panel)

    unique_dates = sorted(panel["date"].unique())
    split_index = max(1, int(len(unique_dates) * (1 - args.test_ratio)))
    train_dates = set(unique_dates[:split_index])
    test_dates = set(unique_dates[split_index:])

    train = panel[panel["date"].isin(train_dates)].reset_index(drop=True)
    test = panel[panel["date"].isin(test_dates)].reset_index(drop=True)

    output_root.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_root / "panel_all.csv", index=False, encoding="utf-8-sig")
    train.to_csv(output_root / "panel_train.csv", index=False, encoding="utf-8-sig")
    test.to_csv(output_root / "panel_test.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rows_total": int(len(panel)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "stocks": int(panel["code"].nunique()),
        "train_days": len(train_dates),
        "test_days": len(test_dates),
        "train_start": min(train_dates),
        "train_end": max(train_dates),
        "test_start": min(test_dates) if test_dates else "",
        "test_end": max(test_dates) if test_dates else "",
        "strong_event_rows": int(
            (
                (panel.get("event_super_positive_count", 0) > 0)
                | (panel.get("event_super_negative_count", 0) > 0)
            ).sum()
        ),
    }
    (output_root / "panel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _add_price_context_by_code(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("code", sort=False):
        parts.append(_add_price_context(group.reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    main()
