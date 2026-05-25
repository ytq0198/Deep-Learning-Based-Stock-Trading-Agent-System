from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FEATURE_COLUMNS = [
    "news_count",
    "sentiment_mean",
    "sentiment_max",
    "sentiment_min",
    "positive_count",
    "negative_count",
    "announcement_score",
    "social_heat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sentiment feature correlation with future returns.")
    parser.add_argument("--data-dir", default="stockdata_recent_balanced_guba_sentiment")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--output-dir", default="reports/sentiment_correlation")
    parser.add_argument("--horizons", default="1,3,5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined_data(Path(args.data_dir), args.stock_code)
    analysis_df = add_future_returns(df, horizons)
    correlations = calculate_correlations(analysis_df, horizons)
    grouped = calculate_grouped_returns(analysis_df, horizons)

    analysis_df.to_csv(output_dir / "analysis_dataset.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(output_dir / "correlations.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(output_dir / "grouped_returns.csv", index=False, encoding="utf-8-sig")
    plot_sentiment_scatter(analysis_df, output_dir)
    plot_grouped_returns(grouped, output_dir)
    write_report(output_dir / "correlation_report.md", correlations, grouped, analysis_df)

    summary = {
        "rows": int(len(analysis_df)),
        "news_covered_rows": int((analysis_df["news_count"] > 0).sum()),
        "coverage_ratio": float((analysis_df["news_count"] > 0).mean()) if len(analysis_df) else 0.0,
        "best_abs_correlation": _best_abs_correlation(correlations),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Rows: {summary['rows']}")
    print(f"News coverage: {summary['coverage_ratio']:.2%}")
    print(f"Best abs correlation: {summary['best_abs_correlation']:.4f}")
    print(f"Saved report to {output_dir}")


def load_combined_data(data_dir: Path, stock_code: str) -> pd.DataFrame:
    frames = []
    for split in ["train", "test"]:
        split_dir = data_dir / split
        for file_path in split_dir.glob("*.csv"):
            if stock_code in file_path.name:
                frame = pd.read_csv(file_path)
                frame["split"] = split
                frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No data found for {stock_code} in {data_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["date"].astype(str)
    df = df.drop_duplicates(subset=["date", "code"]).sort_values("date").reset_index(drop=True)
    for column in FEATURE_COLUMNS + ["close"]:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def add_future_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = df.copy()
    close = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0.0)
    for horizon in horizons:
        df[f"future_{horizon}d_return"] = close.shift(-horizon) / close - 1
    return df


def calculate_correlations(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_COLUMNS:
        for horizon in horizons:
            target = f"future_{horizon}d_return"
            valid = df[[feature, target]].dropna()
            correlation = valid[feature].corr(valid[target]) if len(valid) >= 3 else 0.0
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "correlation": 0.0 if pd.isna(correlation) else float(correlation),
                    "sample_size": int(len(valid)),
                }
            )
    return pd.DataFrame(rows)


def calculate_grouped_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for feature in ["sentiment_mean", "news_count", "negative_count", "social_heat"]:
        labeled = df.copy()
        if feature == "sentiment_mean":
            labeled["group"] = pd.cut(
                labeled[feature],
                bins=[-float("inf"), -0.05, 0.05, float("inf")],
                labels=["negative", "neutral", "positive"],
            )
        else:
            median = float(labeled[feature].median())
            if median <= 0:
                labeled["group"] = labeled[feature].apply(
                    lambda value: "zero" if value <= 0 else "positive"
                )
            else:
                labeled["group"] = pd.cut(
                    labeled[feature],
                    bins=[-float("inf"), 0, median, float("inf")],
                    labels=["zero", "low", "high"],
                )
        for horizon in horizons:
            target = f"future_{horizon}d_return"
            for group, group_df in labeled.groupby("group", observed=True):
                returns = group_df[target].dropna()
                rows.append(
                    {
                        "feature": feature,
                        "group": str(group),
                        "target": target,
                        "mean_return": float(returns.mean()) if len(returns) else 0.0,
                        "sample_size": int(len(returns)),
                    }
                )
    return pd.DataFrame(rows)


def plot_sentiment_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    if "future_1d_return" not in df:
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(df["sentiment_mean"], df["future_1d_return"], alpha=0.7)
    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(0, color="gray", linewidth=1)
    plt.xlabel("sentiment_mean")
    plt.ylabel("future_1d_return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_vs_future_1d.png")
    plt.close()


def plot_grouped_returns(grouped: pd.DataFrame, output_dir: Path) -> None:
    subset = grouped[
        (grouped["feature"] == "sentiment_mean")
        & (grouped["target"] == "future_1d_return")
    ]
    if subset.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(subset["group"], subset["mean_return"])
    plt.axhline(0, color="gray", linewidth=1)
    plt.xlabel("sentiment group")
    plt.ylabel("mean future_1d_return")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_group_returns.png")
    plt.close()


def write_report(
    path: Path,
    correlations: pd.DataFrame,
    grouped: pd.DataFrame,
    analysis_df: pd.DataFrame,
) -> None:
    best = correlations.iloc[correlations["correlation"].abs().idxmax()] if not correlations.empty else None
    coverage = (analysis_df["news_count"] > 0).mean() if len(analysis_df) else 0.0
    content = [
        "# 新闻特征相关性分析",
        "",
        f"- 样本行数：{len(analysis_df)}",
        f"- 新闻覆盖率：{coverage:.2%}",
    ]
    if best is not None:
        content.extend(
            [
                f"- 最高绝对相关：`{best['feature']}` vs `{best['target']}` = {best['correlation']:.4f}",
                "",
            ]
        )
    content.extend(
        [
            "## 解读原则",
            "",
            "- 如果相关系数接近 0，说明当前新闻特征与未来收益关系很弱。",
            "- 如果某些特征在多个 horizon 上方向一致，才值得继续加入强化学习。",
            "- 样本过少时，相关性可能只是偶然现象。",
            "",
            "## 输出文件",
            "",
            "- `correlations.csv`：各新闻特征与未来收益的相关系数。",
            "- `grouped_returns.csv`：按情绪/热度分组后的平均未来收益。",
            "- `sentiment_vs_future_1d.png`：情绪与次日收益散点图。",
            "- `sentiment_group_returns.png`：情绪分组后的次日平均收益。",
        ]
    )
    path.write_text("\n".join(content), encoding="utf-8")


def _best_abs_correlation(correlations: pd.DataFrame) -> float:
    if correlations.empty:
        return 0.0
    return float(correlations["correlation"].abs().max())


if __name__ == "__main__":
    main()
