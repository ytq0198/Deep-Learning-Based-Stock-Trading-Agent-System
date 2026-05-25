from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REGIME_FEATURES = [
    "return_1d",
    "return_5d",
    "volatility_5d",
    "volatility_20d",
    "turn",
    "volume_spike",
    "drawdown_20d",
    "sentiment_mean",
    "social_heat",
    "event_risk_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a market regime classifier and annotate stock data.")
    parser.add_argument("--source-train-dir", default="stockdata_recent_balanced_guba_event/train")
    parser.add_argument("--source-test-dir", default="stockdata_recent_balanced_guba_event/test")
    parser.add_argument("--output-root", default="stockdata_recent_regime")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = load_split(Path(args.source_train_dir), args.stock_code)
    test_df = load_split(Path(args.source_test_dir), args.stock_code)
    combined = pd.concat(
        [train_df.assign(split="train"), test_df.assign(split="test")],
        ignore_index=True,
    ).sort_values("date")
    combined = add_regime_features(combined)

    model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=args.n_regimes, random_state=args.seed, n_init=20),
    )
    x = combined[REGIME_FEATURES].fillna(0.0)
    combined["regime_code"] = model.fit_predict(x)
    combined = add_regime_one_hot(combined, args.n_regimes)

    output_root = Path(args.output_root)
    write_split(combined[combined["split"] == "train"], output_root / "train", args.stock_code)
    write_split(combined[combined["split"] == "test"], output_root / "test", args.stock_code)
    write_summary(combined, output_root / "regime_summary.json", args.n_regimes)

    print(f"Saved regime data to {output_root}")
    print(combined.groupby(["split", "regime_code"]).size())


def load_split(split_dir: Path, stock_code: str) -> pd.DataFrame:
    candidates = sorted(path for path in split_dir.glob("*.csv") if stock_code in path.name)
    if not candidates:
        raise FileNotFoundError(f"No CSV found for {stock_code} in {split_dir}")
    df = pd.read_csv(candidates[0]).sort_values("date").reset_index(drop=True)
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0.0)
    volume = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
    rolling_volume = volume.rolling(window=5, min_periods=1).mean().replace(0, 1.0)
    rolling_max = close.rolling(window=20, min_periods=1).max().replace(0, 1.0)

    df["return_1d"] = close.pct_change().fillna(0.0)
    df["return_5d"] = close.pct_change(5).fillna(0.0)
    df["volatility_5d"] = df["return_1d"].rolling(window=5, min_periods=2).std().fillna(0.0)
    df["volatility_20d"] = df["return_1d"].rolling(window=20, min_periods=2).std().fillna(0.0)
    df["volume_spike"] = (volume / rolling_volume).fillna(0.0)
    df["drawdown_20d"] = (close / rolling_max - 1).fillna(0.0)

    for column in REGIME_FEATURES:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def add_regime_one_hot(df: pd.DataFrame, n_regimes: int) -> pd.DataFrame:
    df = df.copy()
    for regime in range(n_regimes):
        df[f"regime_{regime}"] = (df["regime_code"] == regime).astype(float)
    return df


def write_split(df: pd.DataFrame, output_dir: Path, stock_code: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_df = df.drop(columns=["split"], errors="ignore")
    output_df.to_csv(output_dir / f"{stock_code}.{stock_code}.csv", index=False, encoding="utf-8-sig")


def write_summary(df: pd.DataFrame, path: Path, n_regimes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_regimes": n_regimes,
        "features": REGIME_FEATURES,
        "counts": {
            split: {
                str(regime): int(count)
                for regime, count in group["regime_code"].value_counts().sort_index().items()
            }
            for split, group in df.groupby("split")
        },
        "profile": {},
    }
    for regime, group in df.groupby("regime_code"):
        summary["profile"][str(regime)] = {
            feature: float(group[feature].mean()) for feature in REGIME_FEATURES
        }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
