from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rlenv import MultiStockTradingEnv, StockTradingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on a multi-stock bank panel.")
    parser.add_argument("--panel-train", default="stockdata_bank_panel/panel_train.csv")
    parser.add_argument("--panel-test", default="stockdata_bank_panel/panel_test.csv")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--episode-length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="models/ppo_bank_panel.zip")
    parser.add_argument("--plot-dir", default="img_bank_panel")
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--min-trade-ratio", type=float, default=0.1)
    parser.add_argument("--feature-set", default="full", choices=["full", "minimal_signal", "price_only"])
    parser.add_argument("--opportunity-cost-penalty", type=float, default=2.0)
    parser.add_argument("--opportunity-return-threshold", type=float, default=0.015)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_panel = pd.read_csv(args.panel_train)
    test_panel = pd.read_csv(args.panel_test)

    env_kwargs = dict(
        lot_size=args.lot_size,
        min_trade_ratio=args.min_trade_ratio,
        reward_strategy="profit_delta",
        feature_set=args.feature_set,
        opportunity_cost_penalty=args.opportunity_cost_penalty,
        opportunity_return_threshold=args.opportunity_return_threshold,
    )

    train_env = DummyVecEnv([
        lambda: MultiStockTradingEnv(
            train_panel,
            episode_length=args.episode_length,
            seed=args.seed,
            **env_kwargs,
        )
    ])
    model = PPO("MlpPolicy", train_env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    profits = []
    trades = 0
    for code, group in test_panel.groupby("code"):
        stock_df = group.sort_values("date").reset_index(drop=True)
        if len(stock_df) < 2:
            continue
        test_env = DummyVecEnv([lambda stock_df=stock_df: StockTradingEnv(stock_df, **env_kwargs)])
        obs = test_env.reset()
        done = False
        start_worth = test_env.envs[0].unwrapped.net_worth
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, info = test_env.step(action)
            done = bool(done_arr[0])
        end_worth = float(info[0].get("net_worth", start_worth))
        if float(info[0].get("fees", 0.0)) > 0:
            trades += 1
        profits.append(end_worth - start_worth)

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(profits)), profits)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Per-stock test profit (panel model)")
    plt.xlabel("stock index")
    plt.ylabel("profit")
    plt.tight_layout()
    plt.savefig(plot_dir / "panel_test_profits.png")
    plt.close()

    print(f"Saved model to {model_path}")
    print(f"Test stocks: {len(profits)}, total profit: {sum(profits):.2f}, stocks with fees>0: {trades}")


if __name__ == "__main__":
    main()
