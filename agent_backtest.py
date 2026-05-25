from __future__ import annotations

import argparse
from pathlib import Path

from agent.data_provider import CSVDataProvider
from agent.executor import PaperExecutor
from agent.planner import RuleBasedPlanner
from agent.policy import PPOPolicyModel
from agent.risk import BasicRiskManager
from agent.trading_agent import TradingAgent
from agent.world_model import HistoricalBootstrapWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-layer trading Agent backtest.")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--train-dir", default="stockdata/train")
    parser.add_argument("--test-dir", default="stockdata/test")
    parser.add_argument("--model-path", default="models/ppo_stock.zip")
    parser.add_argument("--output-dir", default="reports/agent_backtest")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--world-horizon", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_provider = CSVDataProvider(args.train_dir)
    test_provider = CSVDataProvider(args.test_dir)
    train_history = train_provider.load_history(args.stock_code)
    test_history = test_provider.load_history(args.stock_code)

    agent = TradingAgent(
        policy=PPOPolicyModel(Path(args.model_path)),
        world_model=HistoricalBootstrapWorldModel(train_history),
        planner=RuleBasedPlanner(),
        risk_manager=BasicRiskManager(),
        executor=PaperExecutor(),
        initial_balance=args.initial_balance,
        world_horizon=args.world_horizon,
    )
    metrics = agent.run_backtest(test_history, args.stock_code, args.output_dir)

    print(f"Agent final net worth: {metrics['final_net_worth']:.2f}")
    print(f"Agent total return: {metrics['total_return']:.2%}")
    print(f"Saved report to {args.output_dir}")


if __name__ == "__main__":
    main()
