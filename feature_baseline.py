from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PRICE_FEATURES = [
    "pctChg",
    "turn",
    "volume_spike",
    "pre_3d_return",
    "pre_5d_return",
    "pre_20d_return",
]

SENTIMENT_LIGHT_FEATURES = [
    "sentiment_mean",
    "social_heat",
]

EVENT_FEATURES = [
    "event_count",
    "event_positive_count",
    "event_negative_count",
    "event_risk_count",
    "event_confidence_mean",
    "event_duration_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline classifiers for next-day stock direction.")
    parser.add_argument("--data-dir", default="stockdata_recent_balanced_guba_sentiment")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--output-dir", default="reports/feature_baseline")
    parser.add_argument(
        "--feature-set",
        choices=["price_only", "sentiment_light", "event", "full"],
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(Path(args.data_dir) / "train", args.stock_code)
    test_df = load_split(Path(args.data_dir) / "test", args.stock_code)
    features = select_features(args.feature_set)
    train_df, test_df = prepare_targets(train_df), prepare_targets(test_df)

    x_train = ensure_features(train_df, features)
    y_train = train_df["target_up"]
    x_test = ensure_features(test_df, features)
    y_test = test_df["target_up"]

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results: dict[str, object] = {
        "feature_set": args.feature_set,
        "features": features,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "models": {},
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = _positive_probability(model, x_test)
        report = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "auc": _safe_auc(y_test, probabilities),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }
        results["models"][name] = report

    (output_dir / "baseline_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(output_dir / "baseline_report.md", results)
    print(f"Saved baseline report to {output_dir}")
    for name, report in results["models"].items():
        print(f"{name}: accuracy={report['accuracy']:.2%}, auc={report['auc']:.4f}")


def load_split(split_dir: Path, stock_code: str) -> pd.DataFrame:
    candidates = sorted(path for path in split_dir.glob("*.csv") if stock_code in path.name)
    if not candidates:
        raise FileNotFoundError(f"No CSV found for {stock_code} in {split_dir}")
    df = pd.read_csv(candidates[0]).sort_values("date").reset_index(drop=True)
    return df


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0.0)
    df["future_1d_return"] = close.shift(-1) / close - 1
    df = df.dropna(subset=["future_1d_return"]).reset_index(drop=True)
    df["target_up"] = (df["future_1d_return"] > 0).astype(int)
    return df


def select_features(feature_set: str) -> list[str]:
    if feature_set == "price_only":
        return PRICE_FEATURES
    if feature_set == "sentiment_light":
        return PRICE_FEATURES + SENTIMENT_LIGHT_FEATURES
    if feature_set == "event":
        return PRICE_FEATURES + EVENT_FEATURES
    return PRICE_FEATURES + SENTIMENT_LIGHT_FEATURES + EVENT_FEATURES


def ensure_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = df.copy()
    for feature in features:
        if feature not in frame.columns:
            frame[feature] = 0.0
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce").fillna(0.0)
    return frame[features]


def _positive_probability(model, x_test: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        if proba.shape[1] > 1:
            return proba[:, 1]
    return model.predict(x_test)


def _safe_auc(y_true, probabilities) -> float:
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return 0.0


def write_markdown_report(path: Path, results: dict[str, object]) -> None:
    lines = [
        "# 特征冷启动基线报告",
        "",
        f"- 特征集：`{results['feature_set']}`",
        f"- 训练样本：{results['train_rows']}",
        f"- 测试样本：{results['test_rows']}",
        f"- 特征：`{', '.join(results['features'])}`",
        "",
        "## 模型结果",
        "",
    ]
    for name, report in results["models"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Accuracy：{report['accuracy']:.2%}",
                f"- AUC：{report['auc']:.4f}",
                f"- Confusion Matrix：`{report['confusion_matrix']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 解读",
            "",
            "如果传统模型准确率无法稳定超过 52%，说明当前特征对次日涨跌的预测力较弱，暂不适合直接投入 PPO。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
