import json
from dotenv import load_dotenv

load_dotenv()

from graph.workflow import build_graph


def main() -> None:
    app = build_graph()

    initial_state = {"iteration": 0}

    print("=" * 60)
    print("Quant Research Agent — LangGraph + Llama")
    print("=" * 60)

    result = app.invoke(initial_state)

    print("\n--- Final State ---")
    for key in ("hypothesis", "evaluation", "critic", "decision", "iteration", "error"):
        if key in result:
            value = result[key]
            if isinstance(value, dict):
                print(f"\n[{key}]")
                print(json.dumps(value, indent=2, default=str))
            else:
                print(f"\n[{key}] {value}")


if __name__ == "__main__":
    main()