from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_baseline import (
    EVENT_FEATURES,
    PRICE_FEATURES,
    SENTIMENT_LIGHT_FEATURES,
    ensure_features,
    prepare_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Panel-level feature baseline and correlation check.")
    parser.add_argument("--panel-train", default="stockdata_bank_panel/panel_train.csv")
    parser.add_argument("--panel-test", default="stockdata_bank_panel/panel_test.csv")
    parser.add_argument("--output-dir", default="reports/panel_feature_analysis")
    parser.add_argument("--feature-set", default="full", choices=["price_only", "sentiment_light", "event", "full"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = prepare_targets(pd.read_csv(args.panel_train))
    test_df = prepare_targets(pd.read_csv(args.panel_test))
    features = select_features(args.feature_set)
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
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
    }
    results = {
        "feature_set": args.feature_set,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "stocks": int(train_df["code"].nunique()),
        "features": features,
        "models": {},
        "correlations": correlation_table(train_df),
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else pred
        results["models"][name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "auc": _safe_auc(y_test, proba),
        }

    (output_dir / "panel_baseline.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir / "panel_baseline_report.md", results)
    print(json.dumps(results["models"], ensure_ascii=False, indent=2))


def select_features(feature_set: str) -> list[str]:
    if feature_set == "price_only":
        return PRICE_FEATURES
    if feature_set == "sentiment_light":
        return PRICE_FEATURES + SENTIMENT_LIGHT_FEATURES
    if feature_set == "event":
        return PRICE_FEATURES + EVENT_FEATURES + [
            "event_super_positive_count",
            "event_super_negative_count",
            "has_strong_event",
        ]
    return (
        PRICE_FEATURES
        + SENTIMENT_LIGHT_FEATURES
        + EVENT_FEATURES
        + ["event_super_positive_count", "event_super_negative_count", "has_strong_event"]
    )


def correlation_table(df: pd.DataFrame) -> dict[str, float]:
    frame = prepare_targets(df.copy())
    correlations = {}
    for column in [
        "event_super_positive_count",
        "event_super_negative_count",
        "event_strength_mean",
        "announcement_score",
        "pre_5d_return",
        "volume_spike",
    ]:
        if column in frame.columns:
            correlations[column] = float(frame[column].corr(frame["future_1d_return"]))
    return correlations


def _safe_auc(y_true, y_score) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def write_report(path: Path, results: dict) -> None:
    lines = [
        "# 银行板块 Panel 特征验证",
        "",
        f"- 训练样本：{results['train_rows']}（约 {results['train_rows'] / max(results['stocks'], 1):.0f} 行/股）",
        f"- 测试样本：{results['test_rows']}",
        f"- 股票数：{results['stocks']}",
        "",
        "## 基线模型",
        "",
    ]
    for name, metrics in results["models"].items():
        lines.append(f"- {name}: accuracy={metrics['accuracy']:.2%}, auc={metrics['auc']:.4f}")
    lines.extend(["", "## 特征相关性", ""])
    for key, value in results["correlations"].items():
        lines.append(f"- {key}: {value:.4f}")
    lines.extend(
        [
            "",
            "## 判读",
            "",
            "- AUC 稳定 > 0.55：存在初步 edge，可进入 PPO panel 训练。",
            "- AUC ≈ 0.5：特征仍接近随机，应继续优化公告规则或延长样本。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
