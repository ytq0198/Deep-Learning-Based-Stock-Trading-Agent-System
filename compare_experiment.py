from __future__ import annotations

import argparse
import json
from pathlib import Path

from main import find_stock_file, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare price-only PPO and sentiment-enhanced PPO.")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--price-train-dir", default="stockdata/train")
    parser.add_argument("--price-test-dir", default="stockdata/test")
    parser.add_argument("--sentiment-train-dir", default="stockdata_sentiment/train")
    parser.add_argument("--sentiment-test-dir", default="stockdata_sentiment/test")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="reports/compare_experiment")
    parser.add_argument("--reward-strategy", choices=["profit_sign", "profit_delta"], default="profit_sign")
    parser.add_argument("--lot-size", type=int, default=1)
    parser.add_argument("--min-trade-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_result = _run_case(
        name="price_only",
        train_dir=Path(args.price_train_dir),
        test_dir=Path(args.price_test_dir),
        stock_code=args.stock_code,
        args=args,
        output_dir=output_dir,
    )
    sentiment_result = _run_case(
        name="sentiment",
        train_dir=Path(args.sentiment_train_dir),
        test_dir=Path(args.sentiment_test_dir),
        stock_code=args.stock_code,
        args=args,
        output_dir=output_dir,
    )

    summary = {
        "price_only": price_result,
        "sentiment": sentiment_result,
        "sentiment_profit_lift": sentiment_result["final_profit"] - price_result["final_profit"],
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "comparison_report.md", summary)

    print(f"Price-only final profit: {price_result['final_profit']:.2f}")
    print(f"Sentiment final profit: {sentiment_result['final_profit']:.2f}")
    print(f"Sentiment lift: {summary['sentiment_profit_lift']:.2f}")
    print(f"Saved comparison report to {output_dir}")


def _run_case(
    name: str,
    train_dir: Path,
    test_dir: Path,
    stock_code: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    train_file = find_stock_file(train_dir, stock_code)
    test_file = find_stock_file(test_dir, stock_code)
    if train_file is None or test_file is None:
        raise FileNotFoundError(f"Missing data for {name}: {train_dir}, {test_dir}")

    profits = train_and_evaluate(
        train_file=train_file,
        test_file=test_file,
        timesteps=args.timesteps,
        reward_strategy=args.reward_strategy,
        model_path=output_dir / f"{name}_model",
        plot_path=output_dir / f"{name}_{stock_code}.png",
        lot_size=args.lot_size,
        min_trade_ratio=args.min_trade_ratio,
        seed=args.seed,
    )
    return {
        "train_file": str(train_file),
        "test_file": str(test_file),
        "final_profit": float(profits[-1] if profits else 0.0),
        "steps": len(profits),
        "plot": str(output_dir / f"{name}_{stock_code}.png"),
    }


def _write_markdown(path: Path, summary: dict[str, object]) -> None:
    price = summary["price_only"]
    sentiment = summary["sentiment"]
    content = f"""# 新闻融合对比实验

## 实验结果

- 无新闻最终收益：{price["final_profit"]:.2f}
- 新闻融合最终收益：{sentiment["final_profit"]:.2f}
- 新闻融合增量：{summary["sentiment_profit_lift"]:.2f}

## 数据文件

- 无新闻训练集：`{price["train_file"]}`
- 无新闻测试集：`{price["test_file"]}`
- 新闻训练集：`{sentiment["train_file"]}`
- 新闻测试集：`{sentiment["test_file"]}`

## 说明

该结果只表示当前数据、当前随机种子和当前参数下的对比。要证明新闻特征稳定有效，需要更多股票、更多时间段和真实新闻数据覆盖。
"""
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
