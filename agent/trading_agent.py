from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .data_provider import row_to_market_state
from .executor import PaperExecutor
from .interfaces import AccountState, ExecutionResult, PolicyAction
from .metrics import buy_and_hold_metrics, calculate_metrics
from .planner import RuleBasedPlanner
from .policy import PPOPolicyModel
from .risk import BasicRiskManager
from .world_model import HistoricalBootstrapWorldModel


class TradingAgent:
    """Coordinates policy, world model, planner, risk manager, and executor."""

    def __init__(
        self,
        policy: PPOPolicyModel,
        world_model: HistoricalBootstrapWorldModel,
        planner: RuleBasedPlanner,
        risk_manager: BasicRiskManager,
        executor: PaperExecutor,
        initial_balance: float = 10_000.0,
        world_horizon: int = 5,
    ) -> None:
        self.policy = policy
        self.world_model = world_model
        self.planner = planner
        self.risk_manager = risk_manager
        self.executor = executor
        self.initial_balance = initial_balance
        self.world_horizon = world_horizon

    def run_backtest(
        self,
        history: pd.DataFrame,
        code: str,
        output_dir: str | Path,
    ) -> dict[str, float]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        account = AccountState(
            cash=self.initial_balance,
            shares=0,
            net_worth=self.initial_balance,
            max_net_worth=self.initial_balance,
        )
        executions: list[ExecutionResult] = []
        equity_rows: list[dict[str, object]] = []

        for _, row in history.iterrows():
            market = row_to_market_state(row, code)
            policy_action = self.policy.predict(market, account)
            candidates = [
                policy_action,
                PolicyAction(action_type="hold", amount=0.0, source="planner"),
            ]
            scenarios = self.world_model.simulate(
                market=market,
                account=account,
                candidates=candidates,
                horizon=self.world_horizon,
            )
            decision = self.planner.decide(market, account, policy_action, scenarios)
            risk_decision = self.risk_manager.review(decision, market, account)
            result = self.executor.execute(risk_decision, market, account)
            account = result.account
            executions.append(result)
            equity_rows.append(
                {
                    "date": market.date,
                    "cash": account.cash,
                    "shares": account.shares,
                    "net_worth": account.net_worth,
                    "max_net_worth": account.max_net_worth,
                    "action": result.executed_action.action_type,
                    "status": result.metadata.get("status", ""),
                }
            )

        equity_curve = pd.DataFrame(equity_rows)
        trades = pd.DataFrame([_execution_to_row(item) for item in executions])
        metrics = calculate_metrics(equity_curve, self.initial_balance)
        benchmark = buy_and_hold_metrics(history, self.initial_balance)

        equity_curve.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        trades.to_csv(output_dir / "trades.csv", index=False, encoding="utf-8-sig")
        with (output_dir / "backtest_summary.json").open("w", encoding="utf-8") as file:
            json.dump(
                {"agent": metrics, "buy_and_hold": benchmark},
                file,
                ensure_ascii=False,
                indent=2,
            )
        _write_markdown_report(output_dir / "backtest_report.md", metrics, benchmark)
        return metrics


def _execution_to_row(result: ExecutionResult) -> dict[str, object]:
    return {
        "date": result.date,
        "code": result.code,
        "requested_action": result.requested_action.action_type,
        "requested_amount": result.requested_action.amount,
        "executed_action": result.executed_action.action_type,
        "executed_amount": result.executed_action.amount,
        "fill_price": result.fill_price,
        "fees": result.fees,
        "slippage": result.slippage,
        "cash": result.account.cash,
        "shares": result.account.shares,
        "net_worth": result.account.net_worth,
        "status": result.metadata.get("status", ""),
        "filled_shares": result.metadata.get("shares", 0),
    }


def _write_markdown_report(
    path: Path,
    metrics: dict[str, float],
    benchmark: dict[str, float],
) -> None:
    content = f"""# Agent 回测报告

## Agent 策略

- 最终净值：{metrics["final_net_worth"]:.2f}
- 总收益率：{metrics["total_return"]:.2%}
- 最大回撤：{metrics["max_drawdown"]:.2%}
- 夏普比率：{metrics["sharpe_ratio"]:.4f}

## 买入并持有基准

- 最终净值：{benchmark["final_net_worth"]:.2f}
- 总收益率：{benchmark["total_return"]:.2%}
- 最大回撤：{benchmark["max_drawdown"]:.2%}
- 夏普比率：{benchmark["sharpe_ratio"]:.4f}

## 说明

本报告来自本地回测或模拟盘流程，不代表真实交易收益。
"""
    path.write_text(content, encoding="utf-8")
