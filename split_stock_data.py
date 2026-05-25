from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split stock CSV files by date range.")
    parser.add_argument("--source-train-dir", default="stockdata_recent/train")
    parser.add_argument("--source-test-dir", default="stockdata_recent/test")
    parser.add_argument("--output-root", default="stockdata_recent_balanced")
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_files = list(Path(args.source_train_dir).glob("*.csv")) + list(
        Path(args.source_test_dir).glob("*.csv")
    )
    if not source_files:
        raise FileNotFoundError("No source CSV files found.")

    grouped: dict[str, list[pd.DataFrame]] = {}
    for file_path in source_files:
        grouped.setdefault(file_path.name, []).append(pd.read_csv(file_path))

    output_root = Path(args.output_root)
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for file_name, frames in grouped.items():
        df = pd.concat(frames, ignore_index=True)
        df["date"] = df["date"].astype(str)
        df = df.drop_duplicates(subset=["date", "code"]).sort_values("date")

        train_df = df[(df["date"] >= args.train_start) & (df["date"] <= args.train_end)]
        test_df = df[(df["date"] >= args.test_start) & (df["date"] <= args.test_end)]
        if train_df.empty or test_df.empty:
            print(f"skip {file_name}: empty train or test split")
            continue

        train_df.to_csv(train_dir / file_name, index=False, encoding="utf-8-sig")
        test_df.to_csv(test_dir / file_name, index=False, encoding="utf-8-sig")
        print(
            f"{file_name}: train {len(train_df)} rows, test {len(test_df)} rows"
        )


if __name__ == "__main__":
    main()
