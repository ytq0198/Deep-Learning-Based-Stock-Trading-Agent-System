from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from dynamic_data import DynamicDataProvider
from main import find_stock_file, load_stock_csv
from rule_event_strategy import RuleEventStrategy
from rlenv import StockTradingEnv
from walk_forward import apply_planner_amplifier, filter_action, load_combined_data


STRATEGIES = ("buy_and_hold", "rule_event", "rule_event_planner", "ppo_only", "ppo_planner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare buy-and-hold, rule-event, and PPO strategies under walk-forward windows.",
    )
    parser.add_argument("--data-dir", default="stockdata_announcements_2019")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--train-window", type=int, default=30)
    parser.add_argument("--test-window", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=10)
    parser.add_argument(
        "--start-date",
        default="",
        help="Only use rows on/after this date when building walk-forward windows (e.g. 2019-01-01).",
    )
    parser.add_argument("--output-dir", default="reports/strategy_compare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--min-trade-ratio", type=float, default=0.1)
    parser.add_argument("--action-threshold", type=float, default=0.2)
    parser.add_argument("--boundary-margin", type=float, default=0.08)
    parser.add_argument("--rule-buy-amount", type=float, default=0.6)
    parser.add_argument("--feature-set", default="full", choices=["full", "minimal_signal", "price_only"])
    parser.add_argument("--opportunity-cost-penalty", type=float, default=0.0)
    parser.add_argument(
        "--strategies",
        default="buy_and_hold,rule_event,rule_event_planner,ppo_only",
        help=f"Comma-separated list from: {','.join(STRATEGIES)}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined_data(Path(args.data_dir), args.stock_code)
    if args.start_date:
        df = df[df["date"] >= args.start_date].reset_index(drop=True)
    provider = DynamicDataProvider(
        df,
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size,
    )
    selected = [name.strip() for name in args.strategies.split(",") if name.strip()]
    for name in selected:
        if name not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {name}")

    summaries = {}
    all_rows = []
    for strategy in selected:
        rows, summary = run_strategy(strategy, args, provider)
        summaries[strategy] = summary
        for row in rows:
            row["strategy"] = strategy
        all_rows.extend(rows)

    result = pd.DataFrame(all_rows)
    result.to_csv(output_dir / "strategy_compare_results.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(summaries, output_dir / "strategy_compare_report.md")
    write_plot(result, output_dir / "strategy_compare_equity.png")
    print(f"Saved comparison to {output_dir}")
    for strategy, summary in summaries.items():
        print(
            f"{strategy}: final={summary['final_net_worth']:.2f}, "
            f"profit={summary['total_profit']:.2f}, trades={summary['trades']}"
        )


def run_strategy(strategy: str, args: argparse.Namespace, provider: DynamicDataProvider):
    rows = []
    net_worth = 10_000.0
    holding = False
    rule_strategy = RuleEventStrategy(
        buy_amount=args.rule_buy_amount,
        use_planner_priced_in=strategy == "rule_event_planner",
    )

    for idx, window in enumerate(provider.windows()):
        if args.max_windows and idx >= args.max_windows:
            break
        market_row = window.test.iloc[0]
        decision_reason = ""

        if strategy == "buy_and_hold":
            if not holding:
                action = np.array([[1.0, 0.8]], dtype=np.float32)
                decision_reason = "initial_buy"
                holding = True
            else:
                action = np.array([[3.0, 0.0]], dtype=np.float32)
                decision_reason = "hold_position"
        elif strategy in {"rule_event", "rule_event_planner"}:
            action, decision_reason = rule_strategy.decide(market_row)
        else:
            train_env = DummyVecEnv([
                lambda window=window: StockTradingEnv(
                    window.train,
                    initial_balance=net_worth,
                    lot_size=args.lot_size,
                    min_trade_ratio=args.min_trade_ratio,
                    reward_strategy="profit_delta",
                    feature_set=args.feature_set,
                    opportunity_cost_penalty=args.opportunity_cost_penalty,
                )
            ])
            model = PPO("MlpPolicy", train_env, verbose=0, seed=args.seed + idx)
            model.learn(total_timesteps=args.timesteps)

            test_env = DummyVecEnv([
                lambda window=window: StockTradingEnv(
                    window.test,
                    initial_balance=net_worth,
                    lot_size=args.lot_size,
                    min_trade_ratio=args.min_trade_ratio,
                    reward_strategy="profit_delta",
                    feature_set=args.feature_set,
                    opportunity_cost_penalty=args.opportunity_cost_penalty,
                )
            ])
            obs = test_env.reset()
            action, _ = model.predict(obs, deterministic=True)
            planner_reason = "ppo_raw"
            if strategy == "ppo_planner":
                action, planner_reason = apply_planner_amplifier(
                    action=action,
                    market_row=market_row,
                    enabled=True,
                    min_buy_amount=args.rule_buy_amount,
                )
            action, filter_reason, _, _, _ = filter_action(
                action=action,
                market_row=market_row,
                action_threshold=args.action_threshold,
                boundary_margin=args.boundary_margin,
                blocked_regimes=set(),
            )
            decision_reason = f"{planner_reason}|{filter_reason}"

        end_net_worth, trade_happened, info = simulate_one_step(
            test_df=window.test,
            action=action,
            net_worth=net_worth,
            lot_size=args.lot_size,
            min_trade_ratio=args.min_trade_ratio,
            feature_set=args.feature_set,
        )
        profit = end_net_worth - net_worth
        rows.append(
            {
                "window": idx,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "start_net_worth": net_worth,
                "end_net_worth": end_net_worth,
                "profit": profit,
                "trade_happened": trade_happened,
                "decision_reason": decision_reason,
                "event_super_positive": float(market_row.get("event_super_positive_count", 0)),
                "event_super_negative": float(market_row.get("event_super_negative_count", 0)),
            }
        )
        net_worth = end_net_worth

    frame = pd.DataFrame(rows)
    summary = {
        "windows": int(len(frame)),
        "final_net_worth": float(frame["end_net_worth"].iloc[-1]) if len(frame) else 10_000.0,
        "total_profit": float(frame["profit"].sum()) if len(frame) else 0.0,
        "trades": int(frame["trade_happened"].sum()) if len(frame) else 0,
        "event_buy_signals": int((frame["decision_reason"].str.contains("buy", case=False)).sum())
        if len(frame)
        else 0,
    }
    return rows, summary


def simulate_one_step(
    test_df: pd.DataFrame,
    action: np.ndarray,
    net_worth: float,
    lot_size: int,
    min_trade_ratio: float,
    feature_set: str,
) -> tuple[float, bool, dict]:
    env = DummyVecEnv([
        lambda: StockTradingEnv(
            test_df,
            initial_balance=net_worth,
            lot_size=lot_size,
            min_trade_ratio=min_trade_ratio,
            reward_strategy="profit_delta",
            feature_set=feature_set,
        )
    ])
    obs = env.reset()
    _, _, _, info = env.step(np.asarray(action, dtype=np.float32))
    end_net_worth = float(info[0]["net_worth"])
    trade_happened = float(info[0].get("fees", 0.0)) > 0
    return end_net_worth, trade_happened, info[0]


def write_report(summaries: dict[str, dict], path: Path) -> None:
    lines = [
        "# 策略 Walk-forward 对比报告",
        "",
        "用于验证：在相同滚动窗口下，纯规则事件策略是否优于买入持有；"
        "若规则策略有 edge，再考虑 PPO 是否能学到该信号。",
        "",
        "| 策略 | 最终净值 | 总收益 | 成交窗口数 | 事件买入信号 |",
        "|------|----------|--------|------------|--------------|",
    ]
    for strategy, summary in summaries.items():
        lines.append(
            f"| {strategy} | {summary['final_net_worth']:.2f} | "
            f"{summary['total_profit']:.2f} | {summary['trades']} | "
            f"{summary.get('event_buy_signals', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 策略说明",
            "",
            "- `buy_and_hold`：首窗买入后持有。",
            "- `rule_event`：强正面事件且未 priced-in 时买入，强负面时卖出/空仓。",
            "- `rule_event_planner`：规则事件 + priced_in 分析器。",
            "- `ppo_only`：滚动训练 PPO + 弱信号过滤。",
            "- `ppo_planner`：PPO + Planner 强事件放大。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(result: pd.DataFrame, path: Path) -> None:
    if result.empty:
        return
    plt.figure(figsize=(10, 5))
    for strategy, group in result.groupby("strategy"):
        plt.plot(group["window"], group["end_net_worth"], "-o", label=strategy)
    plt.grid(True)
    plt.xlabel("window")
    plt.ylabel("net worth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
