from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end bank panel alpha pipeline.")
    parser.add_argument("--start-date", default="2024-01-01", help="Recent window start (baostock supports through 2026-05).")
    parser.add_argument("--end-date", default="2026-05-25")
    parser.add_argument("--quick", action="store_true", help="Use 3 stocks and fewer walk-forward windows.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-announcements", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--timesteps", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    codes = ["sh.600036", "sh.601398", "sh.601939"] if args.quick else []
    code_args = ["--codes", *codes] if codes else []

    steps = []
    if not args.skip_download:
        steps.append(
            [
                python,
                "batch_get_stock_data.py",
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
                "--skip-existing",
                *code_args,
            ]
        )
    if not args.skip_announcements:
        steps.append(
            [
                python,
                "batch_announcement_pipeline.py",
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
                "--skip-existing",
                *code_args,
            ]
        )
    steps.extend(
        [
            [python, "build_panel_dataset.py"],
            [python, "panel_feature_analysis.py"],
        ]
    )
    if not args.skip_train:
        steps.extend(
            [
                [
                    python,
                    "main_panel.py",
                    "--timesteps",
                    str(args.timesteps),
                    "--opportunity-cost-penalty",
                    "2.0",
                ],
                [
                    python,
                    "walk_forward_panel.py",
                    "--max-windows",
                    "3" if args.quick else "6",
                    "--timesteps",
                    str(max(2048, args.timesteps // 4)),
                    "--opportunity-cost-penalty",
                    "2.0",
                ],
            ]
        )

    for command in steps:
        print(f"\n>>> {' '.join(command)}")
        subprocess.run(command, check=True, cwd=Path(__file__).parent)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
