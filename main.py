from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rlenv import StockTradingEnv


def find_stock_file(directory: Path, stock_code: str) -> Path | None:
    candidates = sorted(
        file_path for file_path in directory.glob("*.csv") if stock_code in file_path.name
    )
    if not candidates:
        return None

    # Prefer real downloaded data over synthetic smoke-test data when both exist.
    candidates.sort(key=lambda path: ("demo" in path.name.lower(), len(path.name)))
    return candidates[0]


def load_stock_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "date" in df.columns:
        df = df.sort_values("date")
    return df.reset_index(drop=True)


def make_env(
    df: pd.DataFrame,
    reward_strategy: str,
    lot_size: int,
    min_trade_ratio: float,
    feature_set: str,
    opportunity_cost_penalty: float,
):
    return lambda: StockTradingEnv(
        df,
        reward_strategy=reward_strategy,
        lot_size=lot_size,
        min_trade_ratio=min_trade_ratio,
        feature_set=feature_set,
        opportunity_cost_penalty=opportunity_cost_penalty,
    )


def create_demo_data(output_root: Path, stock_code: str = "sh.600036") -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", periods=260)
    returns = rng.normal(loc=0.0008, scale=0.018, size=len(dates))
    close = 35 * np.cumprod(1 + returns)
    open_price = close * (1 + rng.normal(0, 0.006, size=len(dates)))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0, 0.012, len(dates)))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0, 0.012, len(dates)))

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "code": stock_code,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "preclose": pd.Series(close).shift(1).fillna(close[0]),
            "volume": rng.integers(5_000_000, 50_000_000, len(dates)),
            "amount": rng.integers(100_000_000, 1_000_000_000, len(dates)),
            "adjustflag": 2,
            "turn": rng.uniform(0.1, 3.0, len(dates)),
            "tradestatus": 1,
            "pctChg": pd.Series(close).pct_change().fillna(0) * 100,
            "peTTM": rng.uniform(5, 25, len(dates)),
            "pbMRQ": rng.uniform(0.5, 3.0, len(dates)),
            "psTTM": rng.uniform(0.5, 8.0, len(dates)),
            "pcfNcfTTM": rng.uniform(0.5, 12.0, len(dates)),
            "isST": 0,
        }
    )

    train_dir = output_root / "train"
    test_dir = output_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    df.iloc[:-22].to_csv(train_dir / f"{stock_code}.demo.csv", index=False)
    df.iloc[-22:].to_csv(test_dir / f"{stock_code}.demo.csv", index=False)


def train_and_evaluate(
    train_file: Path,
    test_file: Path,
    timesteps: int,
    reward_strategy: str,
    model_path: Path,
    plot_path: Path,
    lot_size: int,
    min_trade_ratio: float,
    seed: int,
    feature_set: str,
    opportunity_cost_penalty: float,
) -> list[float]:
    np.random.seed(seed)
    train_df = load_stock_csv(train_file)
    test_df = load_stock_csv(test_file)

    env = DummyVecEnv([make_env(train_df, reward_strategy, lot_size, min_trade_ratio, feature_set, opportunity_cost_penalty)])
    model = PPO("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=timesteps)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    test_env = DummyVecEnv([make_env(test_df, reward_strategy, lot_size, min_trade_ratio, feature_set, opportunity_cost_penalty)])
    obs = test_env.reset()
    day_profits: list[float] = []
    trade_count = 0

    for _ in range(len(test_df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = test_env.step(action)
        day_profits.append(float(info[0]["profit"]))
        if float(info[0].get("fees", 0.0)) > 0:
            trade_count += 1
        if done[0]:
            break

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(day_profits, "-o", label=train_file.stem)
    plt.grid(True)
    plt.xlabel("step")
    plt.ylabel("profit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Evaluation trades: {trade_count}")
    return day_profits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent for stock trading.")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--train-dir", default="stockdata/train")
    parser.add_argument("--test-dir", default="stockdata/test")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument(
        "--reward-strategy",
        choices=["profit_sign", "profit_delta"],
        default="profit_sign",
    )
    parser.add_argument("--model-path", default="models/ppo_stock")
    parser.add_argument("--plot-dir", default="img")
    parser.add_argument(
        "--lot-size",
        type=int,
        default=1,
        help="Training/evaluation lot size. Use 100 for stricter A-share simulation.",
    )
    parser.add_argument(
        "--min-trade-ratio",
        type=float,
        default=0.1,
        help="Minimum cash/position ratio used when the model chooses buy or sell.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-set",
        choices=["full", "minimal_signal", "price_only"],
        default="full",
    )
    parser.add_argument("--opportunity-cost-penalty", type=float, default=0.0)
    parser.add_argument(
        "--make-demo-data",
        action="store_true",
        help="Create synthetic stock data for a quick smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path("stockdata")

    if args.make_demo_data:
        create_demo_data(data_root, args.stock_code)

    train_file = find_stock_file(Path(args.train_dir), args.stock_code)
    test_file = find_stock_file(Path(args.test_dir), args.stock_code)
    if not train_file or not test_file:
        raise FileNotFoundError(
            "Stock CSV not found. Run `python get_stock_data.py --code sh.600036` "
            "or use `python main.py --make-demo-data` first."
        )
    print(f"Using train file: {train_file}")
    print(f"Using test file: {test_file}")

    plot_path = Path(args.plot_dir) / f"{args.stock_code}.png"
    profits = train_and_evaluate(
        train_file=train_file,
        test_file=test_file,
        timesteps=args.timesteps,
        reward_strategy=args.reward_strategy,
        model_path=Path(args.model_path),
        plot_path=plot_path,
        lot_size=args.lot_size,
        min_trade_ratio=args.min_trade_ratio,
        seed=args.seed,
        feature_set=args.feature_set,
        opportunity_cost_penalty=args.opportunity_cost_penalty,
    )

    final_profit = profits[-1] if profits else 0.0
    print(f"Final profit: {final_profit:.2f}")
    print(f"Saved model to {args.model_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
