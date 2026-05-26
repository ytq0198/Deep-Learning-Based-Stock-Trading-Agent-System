from __future__ import annotations

import argparse
import json
from pathlib import Path

from bank_universe import BANK_STOCKS
from get_stock_data import Downloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download recent K-line data for the bank stock pool.")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-05-25")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Hold out last N%% trading days as test.")
    parser.add_argument("--output", default="stockdata_bank")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--codes", nargs="*", help="Optional subset of baostock codes.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    selected = [stock for stock in BANK_STOCKS if not args.codes or stock.code in args.codes]
    manifest = []

    for stock in selected:
        output_file = raw_dir / f"{stock.code}.{stock.name}.csv"
        if args.skip_existing and output_file.exists():
            manifest.append({"code": stock.code, "name": stock.name, "file": str(output_file), "skipped": True})
            continue
        Downloader(
            raw_dir,
            args.start_date,
            args.end_date,
            stock.code,
            args.timeout,
        ).run()
        manifest.append({"code": stock.code, "name": stock.name, "file": str(output_file), "skipped": False})

    split_by_date(output_root, args.test_ratio)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Downloaded bank pool to {output_root}")


def split_by_date(output_root: Path, test_ratio: float) -> None:
    import pandas as pd

    raw_dir = output_root / "raw"
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    all_dates: list[str] = []
    for csv_file in raw_dir.glob("*.csv"):
        frame = pd.read_csv(csv_file, usecols=["date"])
        all_dates.extend(frame["date"].astype(str).tolist())
    if not all_dates:
        return

    unique_dates = sorted(set(all_dates))
    split_index = max(1, int(len(unique_dates) * (1 - test_ratio)))
    train_dates = set(unique_dates[:split_index])
    test_dates = set(unique_dates[split_index:])

    for csv_file in raw_dir.glob("*.csv"):
        frame = pd.read_csv(csv_file)
        frame["date"] = frame["date"].astype(str)
        train = frame[frame["date"].isin(train_dates)]
        test = frame[frame["date"].isin(test_dates)]
        train.to_csv(train_dir / csv_file.name, index=False, encoding="utf-8-sig")
        test.to_csv(test_dir / csv_file.name, index=False, encoding="utf-8-sig")

    (output_root / "split_summary.json").write_text(
        json.dumps(
            {
                "train_start": min(train_dates),
                "train_end": max(train_dates),
                "test_start": min(test_dates) if test_dates else "",
                "test_end": max(test_dates) if test_dates else "",
                "train_days": len(train_dates),
                "test_days": len(test_dates),
                "stocks": len(list(raw_dir.glob("*.csv"))),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
