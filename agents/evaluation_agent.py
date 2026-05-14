from typing import Any, Dict

from backtests.metrics import sharpe_ratio, max_drawdown, annualized_return
from backtests.validation import sub_period_check, has_minimum_history


def evaluate_strategy(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: compute performance metrics from backtest results (pure Python, no LLM)."""
    results = state.get("backtest_results")
    if results is None or results.empty:
        state["evaluation"] = {"error": "No backtest results available"}
        return state

    net_returns = results["net_returns"]

    evaluation: Dict[str, Any] = {
        "sharpe": sharpe_ratio(net_returns),
        "max_drawdown": max_drawdown(net_returns),
        "annualized_return": annualized_return(net_returns),
        "avg_turnover": float(results["turnover"].mean()),
        "n_obs": int(net_returns.dropna().__len__()),
        "has_minimum_history": has_minimum_history(net_returns),
    }

    evaluation.update(sub_period_check(net_returns))

    state["evaluation"] = evaluation
    return state