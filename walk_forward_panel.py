from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from panel_dynamic_data import CalendarPanelProvider
from rlenv import MultiStockTradingEnv, StockTradingEnv
from rule_event_strategy import RuleEventStrategy
from walk_forward import filter_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calendar walk-forward on multi-stock panel.")
    parser.add_argument("--panel", default="stockdata_bank_panel/panel_all.csv")
    parser.add_argument("--train-days", type=int, default=120, help="~6 months of trading days.")
    parser.add_argument("--test-days", type=int, default=5)
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=8192)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--output-dir", default="reports/walk_forward_panel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--min-trade-ratio", type=float, default=0.1)
    parser.add_argument("--feature-set", default="full", choices=["full", "minimal_signal", "price_only"])
    parser.add_argument("--opportunity-cost-penalty", type=float, default=2.0)
    parser.add_argument("--action-threshold", type=float, default=0.2)
    parser.add_argument("--strategies", default="rule_event,ppo_panel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel)
    provider = CalendarPanelProvider(
        panel,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    all_rows = []
    summaries = {}

    for strategy in strategies:
        rows, summary = run_panel_strategy(strategy, args, provider)
        summaries[strategy] = summary
        for row in rows:
            row["strategy"] = strategy
        all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    result.to_csv(output_dir / "walk_forward_panel_results.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(summaries, output_dir / "walk_forward_panel_report.md")
    write_plot(result, output_dir / "walk_forward_panel_equity.png")
    print(f"Saved panel walk-forward report to {output_dir}")


def run_panel_strategy(strategy: str, args: argparse.Namespace, provider: CalendarPanelProvider):
    rows = []
    net_worth = 10_000.0
    rule_strategy = RuleEventStrategy(use_planner_priced_in=False)
    rule_planner_strategy = RuleEventStrategy(use_planner_priced_in=True)
    env_kwargs = dict(
        lot_size=args.lot_size,
        min_trade_ratio=args.min_trade_ratio,
        reward_strategy="profit_delta",
        feature_set=args.feature_set,
        opportunity_cost_penalty=args.opportunity_cost_penalty,
    )

    for idx, window in enumerate(provider.windows()):
        if args.max_windows and idx >= args.max_windows:
            break
        test_profit = 0.0
        trade_count = 0

        if strategy == "ppo_panel":
            train_env = DummyVecEnv([
                lambda window=window: MultiStockTradingEnv(
                    window.train,
                    episode_length=args.episode_length,
                    seed=args.seed + idx,
                    **env_kwargs,
                )
            ])
            model = PPO("MlpPolicy", train_env, verbose=0, seed=args.seed + idx)
            model.learn(total_timesteps=args.timesteps)
            for code, group in window.test.groupby("code"):
                stock_df = group.sort_values("date").reset_index(drop=True)
                if len(stock_df) < 2:
                    continue
                profit, traded = evaluate_stock(model, stock_df, env_kwargs, args)
                test_profit += profit
                trade_count += int(traded)
        elif strategy in {"rule_event", "rule_event_planner"}:
            active_rule = rule_planner_strategy if strategy == "rule_event_planner" else rule_strategy
            for code, group in window.test.groupby("code"):
                stock_df = group.sort_values("date").reset_index(drop=True)
                if len(stock_df) < 2:
                    continue
                profit, traded = evaluate_rule_stock(active_rule, stock_df, env_kwargs)
                test_profit += profit
                trade_count += int(traded)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        net_worth += test_profit
        rows.append(
            {
                "window": idx,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "train_rows": len(window.train),
                "test_rows": len(window.test),
                "profit": test_profit,
                "trade_count": trade_count,
                "end_net_worth": net_worth,
            }
        )

    summary = {
        "windows": len(rows),
        "final_net_worth": net_worth,
        "total_profit": net_worth - 10_000.0,
        "trade_windows": int(sum(row["trade_count"] > 0 for row in rows)),
    }
    return rows, summary


def evaluate_stock(model, stock_df: pd.DataFrame, env_kwargs: dict, args: argparse.Namespace):
    test_env = DummyVecEnv([lambda stock_df=stock_df: StockTradingEnv(stock_df, **env_kwargs)])
    obs = test_env.reset()
    done = False
    start_worth = test_env.envs[0].unwrapped.net_worth
    traded = False
    step_idx = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action, _, _, _, _ = filter_action(
            action=action,
            market_row=stock_df.iloc[min(step_idx, len(stock_df) - 1)],
            action_threshold=args.action_threshold,
            boundary_margin=0.08,
            blocked_regimes=set(),
        )
        obs, _, done_arr, info = test_env.step(action)
        done = bool(done_arr[0])
        traded = traded or float(info[0].get("fees", 0.0)) > 0
        step_idx += 1
    end_worth = float(info[0].get("net_worth", start_worth))
    return end_worth - start_worth, traded


def evaluate_rule_stock(rule_strategy: RuleEventStrategy, stock_df: pd.DataFrame, env_kwargs: dict):
    test_env = DummyVecEnv([lambda stock_df=stock_df: StockTradingEnv(stock_df, **env_kwargs)])
    obs = test_env.reset()
    done = False
    start_worth = test_env.envs[0].unwrapped.net_worth
    traded = False
    step_idx = 0
    while not done:
        row = stock_df.iloc[min(step_idx, len(stock_df) - 1)]
        action, _ = rule_strategy.decide(row)
        obs, _, done_arr, info = test_env.step(action)
        done = bool(done_arr[0])
        traded = traded or float(info[0].get("fees", 0.0)) > 0
        step_idx += 1
    end_worth = float(info[0].get("net_worth", start_worth))
    return end_worth - start_worth, traded


def write_report(summaries: dict, path: Path) -> None:
    lines = ["# 银行板块 Panel Walk-forward 报告", ""]
    for strategy, summary in summaries.items():
        lines.append(
            f"- **{strategy}**: final={summary['final_net_worth']:.2f}, "
            f"profit={summary['total_profit']:.2f}, active_windows={summary['trade_windows']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(result: pd.DataFrame, path: Path) -> None:
    if result.empty:
        return
    plt.figure(figsize=(10, 5))
    for strategy, group in result.groupby("strategy"):
        plt.plot(group["window"], group["end_net_worth"], "-o", label=strategy)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
