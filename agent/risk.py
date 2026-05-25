from __future__ import annotations

from dataclasses import dataclass

from .interfaces import AccountState, AgentDecision, MarketState, PolicyAction, RiskDecision


@dataclass(frozen=True)
class RiskConfig:
    max_position_ratio: float = 0.8
    max_single_buy_ratio: float = 0.3
    max_drawdown: float = 0.1


class BasicRiskManager:
    """Applies hard safety checks before an action reaches the executor."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def review(
        self,
        decision: AgentDecision,
        market: MarketState,
        account: AccountState,
    ) -> RiskDecision:
        action = decision.action
        triggered: list[str] = []

        if market.indicators.get("tradestatus", 1.0) != 1.0:
            triggered.append("market_not_trading")

        if account.max_net_worth > 0:
            drawdown = (account.max_net_worth - account.net_worth) / account.max_net_worth
            if drawdown > self.config.max_drawdown and action.action_type == "buy":
                triggered.append("max_drawdown_blocks_buy")

        if action.action_type == "buy":
            current_position = account.shares * market.close
            requested_buy_value = account.cash * action.amount
            future_position_ratio = (
                current_position + requested_buy_value
            ) / max(account.net_worth, 1e-9)

            if action.amount > self.config.max_single_buy_ratio:
                action = PolicyAction(
                    action_type="buy",
                    amount=self.config.max_single_buy_ratio,
                    source="risk_manager",
                    raw_action=decision.action.raw_action,
                )
                triggered.append("single_buy_ratio_clipped")

            if future_position_ratio > self.config.max_position_ratio:
                triggered.append("max_position_blocks_buy")

        if action.action_type == "sell" and account.shares <= 0:
            triggered.append("cannot_sell_without_position")

        hard_blocks = {
            "market_not_trading",
            "max_drawdown_blocks_buy",
            "max_position_blocks_buy",
            "cannot_sell_without_position",
        }
        approved = not any(rule in hard_blocks for rule in triggered)
        if not approved:
            action = PolicyAction(action_type="hold", amount=0.0, source="risk_manager")

        reason = "approved" if approved else f"rejected by risk rules: {', '.join(triggered)}"
        return RiskDecision(
            approved=approved,
            action=action,
            reason=reason,
            triggered_rules=triggered,
        )
