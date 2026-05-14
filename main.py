import json
import logging
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from graph.workflow import build_graph

_cfg = yaml.safe_load(open(Path(__file__).parent / "config" / "settings.yaml"))
logging.basicConfig(
    level=getattr(logging, _cfg["logging"]["level"].upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _log_section(title: str, content: str) -> None:
    logger.info("\n%s\n  %s\n%s\n%s", "─" * 60, title, "─" * 60, content)


def main() -> None:
    app = build_graph()
    initial_state = {"iteration": 0, "history": []}

    logger.info("=" * 60)
    logger.info("  Quant Research Agent — LangGraph + Llama")
    logger.info("=" * 60)
    logger.info("[run] invoking graph...")
    t0 = time.time()

    result = app.invoke(initial_state)
    logger.info("[run] graph finished in %.1fs", time.time() - t0)

    history = result.get("history", [])
    logger.info("[Loop complete] %d iteration(s), decision='%s'", len(history), result.get("decision"))

    for i, entry in enumerate(history, 1):
        h = entry.get("hypothesis", {})
        e = entry.get("evaluation", {})
        c = entry.get("critic", {})
        reasoning = entry.get("reasoning_trace", "")
        critic_reasoning = entry.get("critic_reasoning", "")

        logger.info("\n%s\n  ITERATION %d — %s  (%s)\n%s", "═" * 60, i, h.get("signal_name", "?"), c.get("decision", "?").upper(), "═" * 60)

        if reasoning:
            _log_section("HYPOTHESIS REASONING (CoT)", reasoning)

        _log_section("HYPOTHESIS", json.dumps(h, indent=2, default=str))

        _log_section(
            "EVALUATION",
            f"  Sharpe            : {e.get('sharpe', '?'):.3f}\n"
            f"  Max drawdown      : {e.get('max_drawdown', '?'):.3f}\n"
            f"  Ann. return       : {e.get('annualized_return', '?'):.3f}\n"
            f"  Avg turnover      : {e.get('avg_turnover', '?'):.4f}\n"
            f"  First-half Sharpe : {e.get('first_half_sharpe', '?'):.3f}\n"
            f"  Second-half Sharpe: {e.get('second_half_sharpe', '?'):.3f}\n"
            f"  Observations      : {e.get('n_obs', '?')}",
        )

        if critic_reasoning:
            _log_section("CRITIC REASONING (CoT)", critic_reasoning)

        _log_section(
            "CRITIC DECISION",
            f"  Decision  : {c.get('decision', '?').upper()}\n"
            f"  Reason    : {c.get('reason', '?')}\n"
            f"  Suggestion: {c.get('suggestion', '?')}",
        )

    _log_final_report(history, result.get("decision", "?"))

    final_error = result.get("error")
    if final_error:
        logger.error("[WARNING] Last error: %s", final_error)


def _log_final_report(history: list, final_decision: str) -> None:
    thresholds = {"sharpe": 0.5, "max_drawdown": -0.25, "avg_turnover": 0.5}

    def _tick(val, key) -> str:
        if val is None or val == "?":
            return "?"
        try:
            v = float(val)
        except (TypeError, ValueError):
            return "?"
        if key == "sharpe":
            return "✅" if v >= thresholds["sharpe"] else "❌"
        if key == "max_drawdown":
            return "✅" if v >= thresholds["max_drawdown"] else "❌"
        if key == "avg_turnover":
            return "✅" if v < thresholds["avg_turnover"] else "❌"
        if key in ("first_half_sharpe", "second_half_sharpe"):
            return "✅" if v > 0 else "❌"
        return ""

    def _fmt(val, fmt=".3f") -> str:
        try:
            return format(float(val), fmt)
        except (TypeError, ValueError):
            return "?"

    col_w = [5, 22, 8, 9, 9, 9, 9, 11, 11, 6]
    headers = ["Iter", "Signal", "Decision", "Sharpe", "MaxDD", "AnnRet", "Turnover", "H1 Sharpe", "H2 Sharpe", "Obs"]
    sep = "─" * sum(col_w + [len(col_w) * 3 - 1])

    rows = []
    for i, entry in enumerate(history, 1):
        e = entry.get("evaluation", {})
        c = entry.get("critic", {})
        h = entry.get("hypothesis", {})
        sharpe = e.get("sharpe")
        maxdd = e.get("max_drawdown")
        turnover = e.get("avg_turnover")
        h1 = e.get("first_half_sharpe")
        h2 = e.get("second_half_sharpe")
        rows.append([
            str(i),
            (h.get("signal_name") or "?")[:20],
            c.get("decision", "?").upper(),
            f"{_tick(sharpe, 'sharpe')} {_fmt(sharpe)}",
            f"{_tick(maxdd, 'max_drawdown')} {_fmt(maxdd)}",
            _fmt(e.get("annualized_return")),
            f"{_tick(turnover, 'avg_turnover')} {_fmt(turnover, '.4f')}",
            f"{_tick(h1, 'first_half_sharpe')} {_fmt(h1)}",
            f"{_tick(h2, 'second_half_sharpe')} {_fmt(h2)}",
            str(e.get("n_obs", "?")),
        ])

    def _row(cells) -> str:
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    lines = [
        "",
        "═" * len(sep),
        "  FINAL RESEARCH REPORT",
        f"  Outcome: {final_decision.upper()} after {len(history)} iteration(s)",
        f"  Thresholds — Sharpe≥{thresholds['sharpe']} | MaxDD≥{thresholds['max_drawdown']} | Turnover<{thresholds['avg_turnover']} | H1&H2 Sharpe>0",
        "═" * len(sep),
        _row(headers),
        sep,
        *[_row(r) for r in rows],
        sep,
    ]
    logger.info("\n".join(lines))


if __name__ == "__main__":
    main()
