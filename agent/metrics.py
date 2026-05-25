from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_metrics(equity_curve: pd.DataFrame, initial_balance: float) -> dict[str, float]:
    if equity_curve.empty:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "final_net_worth": initial_balance,
        }

    net_worth = equity_curve["net_worth"].astype(float)
    returns = net_worth.pct_change().dropna()
    running_max = net_worth.cummax()
    drawdown = (running_max - net_worth) / running_max.replace(0, np.nan)

    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = math.sqrt(252) * float(returns.mean() / returns.std())

    return {
        "total_return": float(net_worth.iloc[-1] / initial_balance - 1),
        "max_drawdown": float(drawdown.max() if not drawdown.empty else 0.0),
        "sharpe_ratio": sharpe,
        "final_net_worth": float(net_worth.iloc[-1]),
    }


def buy_and_hold_metrics(history: pd.DataFrame, initial_balance: float) -> dict[str, float]:
    if history.empty:
        return calculate_metrics(pd.DataFrame(), initial_balance)

    close = pd.to_numeric(history["close"], errors="coerce").ffill().fillna(0)
    first_close = float(close.iloc[0])
    shares = int(initial_balance / first_close) if first_close > 0 else 0
    cash = initial_balance - shares * first_close
    equity = pd.DataFrame(
        {
            "date": history["date"] if "date" in history else range(len(history)),
            "net_worth": cash + shares * close,
        }
    )
    return calculate_metrics(equity, initial_balance)
