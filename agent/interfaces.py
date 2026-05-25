from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import pandas as pd

ActionType = Literal["buy", "sell", "hold"]
DecisionSource = Literal["ppo", "world_model", "planner", "risk_manager"]


@dataclass(frozen=True)
class MarketState:
    """Structured market snapshot consumed by the trading agent."""

    date: str
    code: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AccountState:
    """Current simulated account state."""

    cash: float
    shares: int
    net_worth: float
    max_net_worth: float
    cost_basis: float = 0.0


@dataclass(frozen=True)
class PolicyAction:
    """Low-level model action before risk checks."""

    action_type: ActionType
    amount: float
    confidence: float | None = None
    source: DecisionSource = "ppo"
    raw_action: Any | None = None


@dataclass(frozen=True)
class Scenario:
    """World-model scenario for upper-level planning."""

    name: str
    horizon: int
    expected_return: float
    worst_case_return: float
    max_drawdown: float
    path: list[MarketState] = field(default_factory=list)


@dataclass(frozen=True)
class AgentDecision:
    """Planner output before final risk approval."""

    action: PolicyAction
    reason: str
    scenarios: list[Scenario] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskDecision:
    """Risk manager decision that can approve, modify, or reject an action."""

    approved: bool
    action: PolicyAction
    reason: str
    triggered_rules: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionResult:
    """Result returned by the backtest or paper-trading executor."""

    date: str
    code: str
    requested_action: PolicyAction
    executed_action: PolicyAction
    account: AccountState
    fill_price: float
    fees: float = 0.0
    slippage: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DataProvider(Protocol):
    """Provides market data for training, backtesting, or paper trading."""

    def load_history(self, code: str, start: str, end: str) -> pd.DataFrame:
        """Load historical market data."""

    def get_latest_state(self, code: str) -> MarketState:
        """Return the latest available market state."""


class WorldModel(Protocol):
    """Simulates future market states under candidate actions."""

    def simulate(
        self,
        market: MarketState,
        account: AccountState,
        candidates: list[PolicyAction],
        horizon: int,
    ) -> list[Scenario]:
        """Return scenario rollouts for upper-level planning."""


class PolicyModel(Protocol):
    """Fast lower-level reaction model, such as the PPO policy."""

    def predict(self, market: MarketState, account: AccountState) -> PolicyAction:
        """Return the next low-level trading action."""


class Planner(Protocol):
    """Upper-level reasoning layer that compares policy and world-model outputs."""

    def decide(
        self,
        market: MarketState,
        account: AccountState,
        policy_action: PolicyAction,
        scenarios: list[Scenario],
    ) -> AgentDecision:
        """Return the planned action and explanation."""


class RiskManager(Protocol):
    """Final gatekeeper before an action reaches the executor."""

    def review(
        self,
        decision: AgentDecision,
        market: MarketState,
        account: AccountState,
    ) -> RiskDecision:
        """Approve, modify, or reject the planned action."""


class Executor(Protocol):
    """Executes approved actions in backtest or paper-trading mode."""

    def execute(
        self,
        risk_decision: RiskDecision,
        market: MarketState,
        account: AccountState,
    ) -> ExecutionResult:
        """Apply the approved action and return the new account state."""
