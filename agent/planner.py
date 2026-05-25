from __future__ import annotations

from .interfaces import AccountState, AgentDecision, MarketState, PolicyAction, Scenario


class RuleBasedPlanner:
    """Upper-level planner that explains PPO actions using world-model scenarios."""

    def decide(
        self,
        market: MarketState,
        account: AccountState,
        policy_action: PolicyAction,
        scenarios: list[Scenario],
    ) -> AgentDecision:
        selected = _find_matching_scenario(policy_action, scenarios)
        reason = f"PPO suggests {policy_action.action_type} with amount {policy_action.amount:.2f}."

        if selected:
            reason += (
                f" World model scenario expects {selected.expected_return:.2%} "
                f"return over {selected.horizon} step(s), with max drawdown "
                f"{selected.max_drawdown:.2%}."
            )
            if selected.expected_return < 0 and policy_action.action_type == "buy":
                policy_action = PolicyAction(
                    action_type="hold",
                    amount=0.0,
                    source="planner",
                    raw_action=policy_action.raw_action,
                )
                reason += " Planner changes action to hold because simulated return is negative."

        return AgentDecision(
            action=policy_action,
            reason=reason,
            scenarios=scenarios,
            metadata={
                "date": market.date,
                "code": market.code,
                "net_worth": account.net_worth,
            },
        )


def _find_matching_scenario(
    action: PolicyAction,
    scenarios: list[Scenario],
) -> Scenario | None:
    prefix = f"{action.action_type}_"
    for scenario in scenarios:
        if scenario.name.startswith(prefix):
            return scenario
    return scenarios[0] if scenarios else None
