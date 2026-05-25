from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from agent.data_provider import CSVDataProvider
from agent.executor import PaperExecutor
from agent.interfaces import AccountState, PolicyAction
from agent.planner import RuleBasedPlanner
from agent.policy import PPOPolicyModel
from agent.risk import BasicRiskManager
from agent.world_model import HistoricalBootstrapWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper-trading decision.")
    parser.add_argument("--stock-code", default="sh.600036")
    parser.add_argument("--train-dir", default="stockdata/train")
    parser.add_argument("--paper-dir", default="stockdata/test")
    parser.add_argument("--model-path", default="models/ppo_stock.zip")
    parser.add_argument("--output-dir", default="paper_trading")
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--shares", type=int, default=0)
    parser.add_argument("--cost-basis", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_history = CSVDataProvider(args.train_dir).load_history(args.stock_code)
    market = CSVDataProvider(args.paper_dir).get_latest_state(args.stock_code)
    account = AccountState(
        cash=args.cash,
        shares=args.shares,
        net_worth=args.cash + args.shares * market.close,
        max_net_worth=args.cash + args.shares * market.close,
        cost_basis=args.cost_basis,
    )

    policy = PPOPolicyModel(args.model_path)
    world_model = HistoricalBootstrapWorldModel(train_history)
    planner = RuleBasedPlanner()
    risk_manager = BasicRiskManager()
    executor = PaperExecutor()

    policy_action = policy.predict(market, account)
    scenarios = world_model.simulate(
        market,
        account,
        [policy_action, PolicyAction(action_type="hold", amount=0.0, source="planner")],
        horizon=5,
    )
    decision = planner.decide(market, account, policy_action, scenarios)
    risk_decision = risk_manager.review(decision, market, account)
    result = executor.execute(risk_decision, market, account)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "market": asdict(market),
        "policy_action": asdict(policy_action),
        "planner_reason": decision.reason,
        "risk_decision": asdict(risk_decision),
        "execution": asdict(result),
    }
    with (output_dir / "orders.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Paper decision: {result.executed_action.action_type}")
    print(f"Net worth: {result.account.net_worth:.2f}")
    print(f"Saved order log to {output_dir / 'orders.jsonl'}")


if __name__ == "__main__":
    main()
