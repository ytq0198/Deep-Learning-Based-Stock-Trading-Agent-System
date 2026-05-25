from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from feature_baseline import (
    EVENT_FEATURES,
    PRICE_FEATURES,
    SENTIMENT_LIGHT_FEATURES,
    ensure_features,
    load_split,
    prepare_targets,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REGIME_FEATURES = ["regime_code", "regime_0", "regime_1", "regime_2", "regime_3"]

FEATURE_SETS = {
    "price_only": PRICE_FEATURES,
    "sentiment_light": PRICE_FEATURES + SENTIMENT_LIGHT_FEATURES,
    "event": PRICE_FEATURES + EVENT_FEATURES,
    "regime": PRICE_FEATURES + REGIME_FEATURES,
    "sentiment_event": PRICE_FEATURES + SENTIMENT_LIGHT_FEATURES + EVENT_FEATURES,
    "sentiment_regime": PRICE_FEATURES + SENTIMENT_LIGHT_FEATURES + REGIME_FEATURES,
    "event_regime": PRICE_FEATURES + EVENT_FEATURES + REGIME_FEATURES,
    "full": PRICE_FEATURES + SENTIMENT_LIGHT_FEATURES + EVENT_FEATURES + REGIME_FEATURES,
    "minimal_signal": ["pre_5d_return", "volume_spike", "social_heat", "event_risk_count", "regime_code"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search feature sets for next-day direction prediction.")
    parser.add_argument("--data-dir", default="stockdata_recent_regime")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--output-dir", default="reports/feature_selection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = prepare_targets(load_split(Path(args.data_dir) / "train", args.stock_code))
    test_df = prepare_targets(load_split(Path(args.data_dir) / "test", args.stock_code))
    rows = []

    for set_name, features in FEATURE_SETS.items():
        x_train = ensure_features(train_df, features)
        y_train = train_df["target_up"]
        x_test = ensure_features(test_df, features)
        y_test = test_df["target_up"]
        for model_name, model in build_models().items():
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            probabilities = positive_probability(model, x_test)
            rows.append(
                {
                    "feature_set": set_name,
                    "model": model_name,
                    "accuracy": float(accuracy_score(y_test, predictions)),
                    "auc": safe_auc(y_test, probabilities),
                    "feature_count": len(features),
                    "features": ",".join(features),
                }
            )

    results = pd.DataFrame(rows).sort_values(["auc", "accuracy"], ascending=False)
    results.to_csv(output_dir / "feature_selection_results.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir / "feature_selection_report.md", results)
    (output_dir / "best_result.json").write_text(
        json.dumps(results.iloc[0].to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best = results.iloc[0]
    print(f"Best: {best['feature_set']} / {best['model']}")
    print(f"Accuracy: {best['accuracy']:.2%}, AUC: {best['auc']:.4f}")
    print(f"Saved feature selection report to {output_dir}")


def build_models():
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    }


def positive_probability(model, x_test: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        if proba.shape[1] > 1:
            return proba[:, 1]
    return model.predict(x_test)


def safe_auc(y_true, probabilities) -> float:
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return 0.0


def write_report(path: Path, results: pd.DataFrame) -> None:
    lines = [
        "# 特征组合筛选报告",
        "",
        "## 排名前十",
        "",
        "| rank | feature_set | model | accuracy | auc | feature_count |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(results.head(10).iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['feature_set']} | {row['model']} | "
            f"{row['accuracy']:.2%} | {row['auc']:.4f} | {int(row['feature_count'])} |"
        )
    best = results.iloc[0]
    lines.extend(
        [
            "",
            "## 最优组合",
            "",
            f"- 特征集：`{best['feature_set']}`",
            f"- 模型：`{best['model']}`",
            f"- Accuracy：{best['accuracy']:.2%}",
            f"- AUC：{best['auc']:.4f}",
            f"- 特征：`{best['features']}`",
            "",
            "## 解读",
            "",
            "如果最优组合仍低于 52% accuracy 或 AUC 接近 0.5，说明当前特征仍不足以支撑强化学习交易。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
