import json
import re
from typing import Any, Dict

from pydantic import BaseModel, ValidationError
from langchain_core.messages import HumanMessage

from models.llm import get_llm


class Hypothesis(BaseModel):
    hypothesis: str
    asset_class: str
    instruments: list[str]
    signal_name: str
    signal_definition: str
    trade_rule: str
    holding_period_days: int
    rationale: str


_PROMPT = """You are a quant researcher specialising in rates and FX.

Generate exactly ONE trading hypothesis. Choose from these signal families:
- yield curve steepener / flattener (2s10s or 5s30s z-score)
- FX carry (rate differential z-score)
- rates differential
- COT positioning as a slow-moving leading indicator

Return ONLY a valid JSON object — no prose, no markdown fences — with exactly these fields:
{
  "hypothesis": "<one sentence describing the trade>",
  "asset_class": "<Rates | FX | Cross-Asset>",
  "instruments": ["<ticker or series id>", ...],
  "signal_name": "<snake_case signal name, e.g. us_2s10s_zscore>",
  "signal_definition": "<how to compute the signal in plain English>",
  "trade_rule": "<when to go long / short>",
  "holding_period_days": <integer>,
  "rationale": "<1-2 sentence economic rationale>"
}"""


def _extract_json(text: str) -> str:
    """Extract the first JSON object from a string, stripping markdown fences."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def generate_hypothesis(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: call Llama to generate a structured trading hypothesis."""
    llm = get_llm()

    for attempt in range(3):
        raw = llm.invoke([HumanMessage(content=_PROMPT)])
        text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            parsed = json.loads(_extract_json(text))
            hypothesis = Hypothesis(**parsed)
            state["hypothesis"] = hypothesis.model_dump()
            return state
        except (json.JSONDecodeError, ValidationError, TypeError):
            if attempt == 2:
                raise ValueError(f"Hypothesis agent failed to produce valid JSON after 3 attempts. Last output:\n{text}")

    return state