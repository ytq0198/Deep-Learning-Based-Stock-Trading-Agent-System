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
from risk_signal import RiskSignalScorer
from rlenv import StockTradingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward PPO training with rolling windows.")
    parser.add_argument("--data-dir", default="stockdata_recent_regime")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--train-window", type=int, default=30)
    parser.add_argument("--test-window", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=10)
    parser.add_argument("--output-dir", default="reports/walk_forward")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lot-size", type=int, default=1)
    parser.add_argument("--min-trade-ratio", type=float, default=0.1)
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=0.2,
        help="Hold when model action amount is below this threshold.",
    )
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=0.08,
        help="Hold when action type is too close to buy/sell or sell/hold boundary.",
    )
    parser.add_argument(
        "--blocked-regimes",
        default="",
        help="Comma separated regime codes where buy actions are blocked.",
    )
    parser.add_argument(
        "--auto-block-risk-regimes",
        action="store_true",
        help="Infer risky regimes from negative return/drawdown/event-risk profile.",
    )
    parser.add_argument(
        "--use-risk-signal",
        action="store_true",
        help="Use minimal_signal risk score to block buy actions.",
    )
    parser.add_argument("--max-risk-score-to-buy", type=float, default=0.55)
    parser.add_argument(
        "--feature-set",
        choices=["full", "minimal_signal", "price_only"],
        default="full",
    )
    parser.add_argument("--planner-amplify", action="store_true")
    parser.add_argument("--planner-min-buy", type=float, default=0.5)
    parser.add_argument("--opportunity-cost-penalty", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined_data(Path(args.data_dir), args.stock_code)
    provider = DynamicDataProvider(
        df,
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size,
    )

    rows = []
    net_worth = 10_000.0
    blocked_regimes = parse_blocked_regimes(args.blocked_regimes)
    if args.auto_block_risk_regimes:
        blocked_regimes |= infer_risky_regimes(df)
    risk_scorer = RiskSignalScorer()

    for idx, window in enumerate(provider.windows()):
        if args.max_windows and idx >= args.max_windows:
            break
        env = DummyVecEnv([
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
        model = PPO("MlpPolicy", env, verbose=0, seed=args.seed + idx)
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
        planned_action, planner_reason = apply_planner_amplifier(
            action=action,
            market_row=window.test.iloc[0],
            enabled=args.planner_amplify,
            min_buy_amount=args.planner_min_buy,
        )
        filtered_action, filter_reason, raw_action_type, raw_amount, risk_score = filter_action(
            action=planned_action,
            market_row=window.test.iloc[0],
            action_threshold=args.action_threshold,
            boundary_margin=args.boundary_margin,
            blocked_regimes=blocked_regimes,
            risk_scorer=risk_scorer if args.use_risk_signal else None,
            max_risk_score_to_buy=args.max_risk_score_to_buy,
        )
        _, _, _, info = test_env.step(filtered_action)
        end_net_worth = float(info[0]["net_worth"])
        profit = end_net_worth - net_worth
        trade_happened = float(info[0].get("fees", 0.0)) > 0
        rows.append(
            {
                "window": idx,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "start_net_worth": net_worth,
                "end_net_worth": end_net_worth,
                "profit": profit,
                "trade_happened": trade_happened,
                "raw_action_type": raw_action_type,
                "raw_amount": raw_amount,
                "filter_reason": filter_reason,
                "planner_reason": planner_reason,
                "regime_code": int(window.test.iloc[0].get("regime_code", -1)),
                "risk_score": risk_score,
            }
        )
        net_worth = end_net_worth

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "walk_forward_results.csv", index=False, encoding="utf-8-sig")
    write_report(result, output_dir / "walk_forward_report.md")
    write_plot(result, output_dir / "walk_forward_equity.png")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "windows": int(len(result)),
                "final_net_worth": float(result["end_net_worth"].iloc[-1]) if len(result) else 10_000.0,
                "total_profit": float(result["profit"].sum()) if len(result) else 0.0,
                "trades": int(result["trade_happened"].sum()) if len(result) else 0,
                "filtered_holds": int((result["filter_reason"] != "pass").sum()) if len(result) else 0,
                "avg_risk_score": float(result["risk_score"].mean()) if len(result) else 0.0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Walk-forward windows: {len(result)}")
    print(f"Final net worth: {result['end_net_worth'].iloc[-1] if len(result) else 10000:.2f}")
    print(f"Saved report to {output_dir}")


def load_combined_data(data_dir: Path, stock_code: str) -> pd.DataFrame:
    frames = []
    for split in ["train", "test"]:
        file_path = find_stock_file(data_dir / split, stock_code)
        if file_path:
            frames.append(load_stock_csv(file_path))
    if not frames:
        raise FileNotFoundError(f"No train/test CSV found in {data_dir}")
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def parse_blocked_regimes(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def infer_risky_regimes(df: pd.DataFrame) -> set[int]:
    if "regime_code" not in df.columns:
        return set()
    risky: set[int] = set()
    grouped = df.groupby("regime_code")
    for regime, group in grouped:
        return_5d = float(pd.to_numeric(group.get("return_5d", 0), errors="coerce").fillna(0).mean())
        drawdown = float(pd.to_numeric(group.get("drawdown_20d", 0), errors="coerce").fillna(0).mean())
        event_risk = float(pd.to_numeric(group.get("event_risk_count", 0), errors="coerce").fillna(0).mean())
        if (return_5d < 0 and drawdown < -0.02) or event_risk > 0.1:
            risky.add(int(regime))
    return risky


def filter_action(
    action,
    market_row: pd.Series,
    action_threshold: float,
    boundary_margin: float,
    blocked_regimes: set[int],
    risk_scorer: RiskSignalScorer | None = None,
    max_risk_score_to_buy: float = 0.55,
):
    arr = np.asarray(action, dtype=np.float32).copy()
    flat = arr.reshape(-1)
    raw_action_type = float(flat[0])
    raw_amount = float(flat[1])
    regime_code = int(market_row.get("regime_code", -1))
    risk_score = 0.0

    reason = "pass"
    if raw_action_type >= 2.5:
        reason = "model_hold"
    elif raw_amount < action_threshold:
        reason = "weak_amount"
    elif abs(raw_action_type - 1.5) < boundary_margin or abs(raw_action_type - 2.5) < boundary_margin:
        reason = "boundary_uncertain"
    elif raw_action_type < 1.5 and regime_code in blocked_regimes:
        reason = f"blocked_regime_{regime_code}"
    elif raw_action_type < 1.5 and risk_scorer is not None:
        risk_signal = risk_scorer.score(market_row)
        risk_score = risk_signal.score
        if risk_score > max_risk_score_to_buy or not risk_signal.allow_buy:
            reason = risk_signal.reason

    if reason != "pass":
        flat[0] = 3.0
        flat[1] = 0.0
    return flat.reshape(arr.shape), reason, raw_action_type, raw_amount, risk_score


def apply_planner_amplifier(
    action,
    market_row: pd.Series,
    enabled: bool,
    min_buy_amount: float,
):
    arr = np.asarray(action, dtype=np.float32).copy()
    flat = arr.reshape(-1)
    if not enabled:
        return arr, "disabled"

    super_positive = float(market_row.get("event_super_positive_count", 0.0))
    super_negative = float(market_row.get("event_super_negative_count", 0.0))
    priced_in = float(market_row.get("event_priced_in_count", 0.0))
    priced_in_score = float(market_row.get("priced_in_score", 0.0))
    event_strength = float(market_row.get("event_strength_mean", 0.0))

    if super_negative > 0 or event_strength <= -0.75:
        flat[0] = 3.0
        flat[1] = 0.0
        return flat.reshape(arr.shape), "super_negative_hold"

    if flat[0] < 1.5 and super_positive > 0 and priced_in <= 0 and priced_in_score < 0.5:
        flat[1] = max(float(flat[1]), min_buy_amount)
        return flat.reshape(arr.shape), "super_positive_amplify"

    if flat[0] < 1.5 and super_positive > 0 and (priced_in > 0 or priced_in_score >= 0.5):
        return arr, "positive_priced_in_no_amplify"

    return arr, "no_planner_change"


def write_report(result: pd.DataFrame, path: Path) -> None:
    if result.empty:
        path.write_text("# Walk-forward 报告\n\n没有产生窗口。", encoding="utf-8")
        return
    content = f"""# Walk-forward 动态训练报告

- 窗口数：{len(result)}
- 最终净值：{result["end_net_worth"].iloc[-1]:.2f}
- 总收益：{result["profit"].sum():.2f}
- 有效交易窗口数：{int(result["trade_happened"].sum())}
- 过滤为空仓窗口数：{int((result["filter_reason"] != "pass").sum())}
- 平均风险分：{result["risk_score"].mean():.4f}
- Planner 放大/调整窗口数：{int((result["planner_reason"] != "disabled").sum())}

## 说明

每个窗口都只使用最近一段训练数据重新训练 PPO，然后测试下一段数据。这用于模拟非平稳市场下的动态更新。

交易过滤器会在信号过弱、动作边界不清晰或市场状态被识别为高风险时转为空仓，以减少手续费和滑点造成的无效交易。
"""
    path.write_text(content, encoding="utf-8")


def write_plot(result: pd.DataFrame, path: Path) -> None:
    if result.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(result["window"], result["end_net_worth"], "-o")
    plt.grid(True)
    plt.xlabel("window")
    plt.ylabel("net worth")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
