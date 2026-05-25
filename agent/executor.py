from __future__ import annotations

from dataclasses import dataclass

from .interfaces import AccountState, ExecutionResult, MarketState, PolicyAction, RiskDecision


@dataclass(frozen=True)
class ExecutionConfig:
    fee_rate: float = 0.0003
    min_fee: float = 5.0
    slippage_rate: float = 0.0005
    lot_size: int = 100


class PaperExecutor:
    """Executes actions against a local simulated account."""

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def execute(
        self,
        risk_decision: RiskDecision,
        market: MarketState,
        account: AccountState,
    ) -> ExecutionResult:
        requested = risk_decision.action
        if not risk_decision.approved or requested.action_type == "hold":
            new_account = _mark_to_market(account, market.close)
            return ExecutionResult(
                date=market.date,
                code=market.code,
                requested_action=requested,
                executed_action=PolicyAction(action_type="hold", amount=0.0, source="risk_manager"),
                account=new_account,
                fill_price=market.close,
                metadata={"status": "rejected" if not risk_decision.approved else "skipped"},
            )

        if requested.action_type == "buy":
            return self._buy(requested, market, account)
        return self._sell(requested, market, account)

    def _buy(
        self,
        action: PolicyAction,
        market: MarketState,
        account: AccountState,
    ) -> ExecutionResult:
        fill_price = market.close * (1 + self.config.slippage_rate)
        budget = account.cash * action.amount
        raw_shares = int(budget / fill_price)
        shares = _round_lot(raw_shares, self.config.lot_size)
        trade_value = shares * fill_price
        fees = _fee(trade_value, self.config)

        if shares <= 0 or trade_value + fees > account.cash:
            new_account = _mark_to_market(account, market.close)
            return _skipped_result(market, action, new_account, fill_price, "insufficient_cash")

        cash = account.cash - trade_value - fees
        previous_cost = account.cost_basis * account.shares
        total_shares = account.shares + shares
        cost_basis = (previous_cost + trade_value) / total_shares
        net_worth = cash + total_shares * market.close
        new_account = AccountState(
            cash=cash,
            shares=total_shares,
            net_worth=net_worth,
            max_net_worth=max(account.max_net_worth, net_worth),
            cost_basis=cost_basis,
        )
        return ExecutionResult(
            date=market.date,
            code=market.code,
            requested_action=action,
            executed_action=action,
            account=new_account,
            fill_price=fill_price,
            fees=fees,
            slippage=fill_price - market.close,
            metadata={"status": "filled", "shares": shares},
        )

    def _sell(
        self,
        action: PolicyAction,
        market: MarketState,
        account: AccountState,
    ) -> ExecutionResult:
        fill_price = market.close * (1 - self.config.slippage_rate)
        raw_shares = int(account.shares * action.amount)
        shares = min(account.shares, _round_lot(raw_shares, self.config.lot_size))
        trade_value = shares * fill_price
        fees = _fee(trade_value, self.config)

        if shares <= 0:
            new_account = _mark_to_market(account, market.close)
            return _skipped_result(market, action, new_account, fill_price, "no_sellable_lot")

        cash = account.cash + trade_value - fees
        remaining_shares = account.shares - shares
        cost_basis = account.cost_basis if remaining_shares > 0 else 0.0
        net_worth = cash + remaining_shares * market.close
        new_account = AccountState(
            cash=cash,
            shares=remaining_shares,
            net_worth=net_worth,
            max_net_worth=max(account.max_net_worth, net_worth),
            cost_basis=cost_basis,
        )
        return ExecutionResult(
            date=market.date,
            code=market.code,
            requested_action=action,
            executed_action=action,
            account=new_account,
            fill_price=fill_price,
            fees=fees,
            slippage=market.close - fill_price,
            metadata={"status": "filled", "shares": shares},
        )


def _round_lot(shares: int, lot_size: int) -> int:
    if lot_size <= 1:
        return max(0, shares)
    return max(0, shares // lot_size * lot_size)


def _fee(trade_value: float, config: ExecutionConfig) -> float:
    if trade_value <= 0:
        return 0.0
    return max(config.min_fee, trade_value * config.fee_rate)


def _mark_to_market(account: AccountState, close: float) -> AccountState:
    net_worth = account.cash + account.shares * close
    return AccountState(
        cash=account.cash,
        shares=account.shares,
        net_worth=net_worth,
        max_net_worth=max(account.max_net_worth, net_worth),
        cost_basis=account.cost_basis,
    )


def _skipped_result(
    market: MarketState,
    action: PolicyAction,
    account: AccountState,
    fill_price: float,
    status: str,
) -> ExecutionResult:
    return ExecutionResult(
        date=market.date,
        code=market.code,
        requested_action=action,
        executed_action=PolicyAction(action_type="hold", amount=0.0, source="risk_manager"),
        account=account,
        fill_price=fill_price,
        metadata={"status": status},
    )
