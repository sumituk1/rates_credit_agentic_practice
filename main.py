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

    final_error = result.get("error")
    if final_error:
        logger.error("[WARNING] Last error: %s", final_error)


if __name__ == "__main__":
    main()
