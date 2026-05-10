import json
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from models.llm import get_llm


_PROMPT_TEMPLATE = """You are a senior quant research critic. Be rigorous and concise.

Hypothesis submitted:
{hypothesis}

Backtest evaluation:
{evaluation}

Thresholds for acceptance:
- Sharpe >= 0.5
- Max drawdown > -0.25 (i.e. no worse than -25%)
- Avg daily turnover < 0.5
- Strategy must hold in both sub-periods (first_half_sharpe and second_half_sharpe both > 0)

Rules for rejection:
- Lookahead bias suspected
- Sharpe < 0 in either sub-period
- Max drawdown < -0.40

Return ONLY a valid JSON object — no prose, no markdown fences:
{{
  "decision": "accept|reject|refine",
  "reason": "<one sentence>",
  "suggestion": "<one concrete change to improve the signal, or 'none' if accepting/rejecting>"
}}"""


def _extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def critic_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: LLM-based critic decides accept / reject / refine."""
    llm = get_llm()

    prompt = _PROMPT_TEMPLATE.format(
        hypothesis=json.dumps(state.get("hypothesis", {}), indent=2),
        evaluation=json.dumps(state.get("evaluation", {}), indent=2),
    )

    for attempt in range(3):
        raw = llm.invoke([HumanMessage(content=prompt)])
        text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            decision = json.loads(_extract_json(text))
            assert "decision" in decision and decision["decision"] in {"accept", "reject", "refine"}
            state["critic"] = decision
            state["decision"] = decision["decision"]
            return state
        except (json.JSONDecodeError, AssertionError, TypeError):
            if attempt == 2:
                state["critic"] = {"decision": "reject", "reason": "Critic agent parse failure", "suggestion": "none"}
                state["decision"] = "reject"

    return state