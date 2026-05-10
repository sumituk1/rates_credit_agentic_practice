from typing import Any, Dict

from langgraph.graph import StateGraph, END

from graph.state import ResearchState
from agents.hypothesis_agent import generate_hypothesis
from agents.evaluation_agent import evaluate_strategy
from agents.critic_agent import critic_agent


# ---------------------------------------------------------------------------
# Signal dispatch: map hypothesis → (signal Series, returns Series)
# ---------------------------------------------------------------------------

def _load_curve_signal(signal_name: str, start: str) -> tuple:
    from data.loaders.fred import load_us_yields
    from data.processing.yield_curve import add_curve_features, yield_to_approx_returns
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "settings.yaml"))
    window = cfg["backtest"]["zscore_window"]

    df = load_us_yields(start=start).ffill().dropna()
    df = add_curve_features(df, zscore_window=window)
    df = df.ffill().dropna()

    if "5s30" in signal_name:
        signal = df["us_5s30s_zscore"]
        returns = yield_to_approx_returns(df["us_30y"], duration_years=20.0).reindex(signal.index).ffill()
    else:
        signal = df["us_2s10s_zscore"]
        returns = yield_to_approx_returns(df["us_10y"], duration_years=9.0).reindex(signal.index).ffill()

    return signal.dropna(), returns.dropna()


def _load_fx_carry_signal(instruments: list, start: str) -> tuple:
    from data.loaders.yahoo import load_fx_returns, FX_TICKERS
    from data.loaders.fred import load_us_yields
    from data.processing.fx_carry import fx_carry_signal
    import pandas as pd
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "settings.yaml"))
    window = cfg["backtest"]["zscore_window"]

    yields = load_us_yields(start=start).ffill().dropna()
    us_rate = yields["us_2y"]

    ticker = None
    for inst in instruments:
        inst_upper = inst.upper().replace("/", "").replace("-", "")
        if inst_upper in FX_TICKERS:
            ticker = FX_TICKERS[inst_upper]
            break
        if "=X" in inst:
            ticker = inst
            break
    if ticker is None:
        ticker = "EURUSD=X"

    fx_ret = load_fx_returns(ticker, start=start)
    foreign_rate = pd.Series(0.0, index=us_rate.index)

    signal = fx_carry_signal(us_rate, foreign_rate, zscore_window=window)
    common_idx = signal.index.intersection(fx_ret.index)
    return signal.loc[common_idx].dropna(), fx_ret.loc[common_idx].dropna()


# ---------------------------------------------------------------------------
# Backtest node
# ---------------------------------------------------------------------------

def backtest_node(state: ResearchState) -> ResearchState:
    from backtests.engine import run_long_short_backtest
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "settings.yaml"))
    start = cfg["data"]["start_date"]
    tc_bps = cfg["backtest"]["transaction_cost_bps"]

    hypothesis = state.get("hypothesis", {})
    signal_name = (hypothesis.get("signal_name") or "").lower()
    instruments = hypothesis.get("instruments") or []
    asset_class = (hypothesis.get("asset_class") or "").lower()

    try:
        if "carry" in signal_name or "carry" in asset_class or "fx" in asset_class:
            signal, returns = _load_fx_carry_signal(instruments, start)
        else:
            signal, returns = _load_curve_signal(signal_name, start)

        common = signal.index.intersection(returns.index)
        results = run_long_short_backtest(
            signal.loc[common],
            returns.loc[common],
            transaction_cost_bps=tc_bps,
        )
        state["backtest_results"] = results
        state["error"] = None
    except Exception as exc:
        state["backtest_results"] = None
        state["error"] = str(exc)

    return state


# ---------------------------------------------------------------------------
# History node — appends completed iteration before routing
# ---------------------------------------------------------------------------

def record_iteration(state: ResearchState) -> ResearchState:
    """Append the current hypothesis + evaluation + critic decision to history."""
    history = list(state.get("history") or [])
    history.append({
        "hypothesis": state.get("hypothesis", {}),
        "evaluation": state.get("evaluation", {}),
        "critic": state.get("critic", {}),
        "reasoning_trace": state.get("reasoning_trace", ""),
        "critic_reasoning": state.get("critic_reasoning", ""),
    })
    state["history"] = history
    return state


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_record(state: ResearchState) -> str:
    decision = state.get("decision", "reject")
    iteration = state.get("iteration", 0)
    max_iter = 5

    if decision == "accept":
        return "end"

    if iteration >= max_iter:
        print(f"[workflow] max iterations ({max_iter}) reached — stopping.")
        return "end"

    state["iteration"] = iteration + 1  # type: ignore[index]
    return "retry"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("hypothesis", generate_hypothesis)
    graph.add_node("backtest", backtest_node)
    graph.add_node("evaluate", evaluate_strategy)
    graph.add_node("critic", critic_agent)
    graph.add_node("record", record_iteration)

    graph.set_entry_point("hypothesis")

    graph.add_edge("hypothesis", "backtest")
    graph.add_edge("backtest", "evaluate")
    graph.add_edge("evaluate", "critic")
    graph.add_edge("critic", "record")

    graph.add_conditional_edges(
        "record",
        route_after_record,
        {
            "retry": "hypothesis",
            "end": END,
        },
    )

    return graph.compile()
