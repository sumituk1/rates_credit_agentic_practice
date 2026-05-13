import json
import time
from dotenv import load_dotenv

load_dotenv()

from graph.workflow import build_graph


def _print_section(title: str, content: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(content)


def main() -> None:
    app = build_graph()
    initial_state = {"iteration": 0, "history": []}

    print("=" * 60)
    print("  Quant Research Agent — LangGraph + Llama")
    print("=" * 60)
    print("[run] invoking graph...", flush=True)
    t0 = time.time()

    result = app.invoke(initial_state)
    print(f"[run] graph finished in {time.time() - t0:.1f}s", flush=True)

    history = result.get("history", [])
    print(f"\n[Loop complete] {len(history)} iteration(s), decision='{result.get('decision')}'")

    for i, entry in enumerate(history, 1):
        h = entry.get("hypothesis", {})
        e = entry.get("evaluation", {})
        c = entry.get("critic", {})
        reasoning = entry.get("reasoning_trace", "")
        critic_reasoning = entry.get("critic_reasoning", "")

        print(f"\n{'═' * 60}")
        print(f"  ITERATION {i} — {h.get('signal_name', '?')}  ({c.get('decision', '?').upper()})")
        print(f"{'═' * 60}")

        if reasoning:
            _print_section("HYPOTHESIS REASONING (CoT)", reasoning)

        _print_section("HYPOTHESIS", json.dumps(h, indent=2, default=str))

        _print_section(
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
            _print_section("CRITIC REASONING (CoT)", critic_reasoning)

        _print_section(
            "CRITIC DECISION",
            f"  Decision  : {c.get('decision', '?').upper()}\n"
            f"  Reason    : {c.get('reason', '?')}\n"
            f"  Suggestion: {c.get('suggestion', '?')}",
        )

    final_error = result.get("error")
    if final_error:
        print(f"\n[WARNING] Last error: {final_error}")


if __name__ == "__main__":
    main()
