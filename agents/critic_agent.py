"""
Critic Agent — Chain-of-Thought with full iteration memory.

Flow:
  1. Receives the current hypothesis, evaluation metrics, AND the full history
     of every previous attempt in this research loop.
  2. Asks Llama to reason through 4 explicit steps:
       - Statistical validity (numbers vs. thresholds)
       - Robustness (sub-period consistency, cross-regime)
       - Bias check (lookahead, data-mining, overfitting flags)
       - Decision with a specific, surgical refinement suggestion
  3. Parses reasoning trace and decision JSON separately.
  4. Falls back to rule-based reject on parse failure.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from models.llm import get_llm


def _format_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "None."
    lines = []
    for i, entry in enumerate(history, 1):
        h = entry.get("hypothesis", {})
        e = entry.get("evaluation", {})
        c = entry.get("critic", {})
        lines.append(
            f"[Iteration {i}]\n"
            f"  Signal      : {h.get('signal_name', '?')}\n"
            f"  Hypothesis  : {h.get('hypothesis', '?')}\n"
            f"  Sharpe      : {e.get('sharpe', '?')}\n"
            f"  MaxDD       : {e.get('max_drawdown', '?')}\n"
            f"  Ann. Return : {e.get('annualized_return', '?')}\n"
            f"  Turnover    : {e.get('avg_turnover', '?')}\n"
            f"  H1 Sharpe   : {e.get('first_half_sharpe', '?')}\n"
            f"  H2 Sharpe   : {e.get('second_half_sharpe', '?')}\n"
            f"  Decision    : {c.get('decision', '?')}\n"
            f"  Reason      : {c.get('reason', '?')}\n"
            f"  Suggestion  : {c.get('suggestion', '?')}"
        )
    return "\n\n".join(lines)


_PROMPT_TEMPLATE = """You are a rigorous quant research critic at a top macro hedge fund.
Your job is NOT to be kind — it is to ensure every strategy that passes is genuinely robust.

=== CURRENT HYPOTHESIS ===
Signal      : {signal_name}
Description : {hypothesis}
Rationale   : {rationale}

=== CURRENT BACKTEST RESULTS ===
Sharpe ratio        : {sharpe}
Max drawdown        : {max_drawdown}
Annualised return   : {annualized_return}
Avg daily turnover  : {avg_turnover}
Observations        : {n_obs}
Sufficient history  : {has_min_history}
First-half Sharpe   : {first_half_sharpe}
Second-half Sharpe  : {second_half_sharpe}
First-half MaxDD    : {first_half_drawdown}
Second-half MaxDD   : {second_half_drawdown}

=== HISTORY OF ALL PREVIOUS ITERATIONS ===
{history}

Acceptance thresholds:
  - Sharpe >= 0.5
  - Max drawdown >= -0.25 (no worse than -25%)
  - Avg daily turnover < 0.5
  - Both sub-period Sharpes > 0 (strategy must work in both halves)
  - Sufficient history (>= 252 observations)

Grounds for immediate rejection:
  - Either sub-period Sharpe < 0
  - Max drawdown < -0.40
  - Fewer than 252 observations
  - Signal appears to have lookahead bias

Reason carefully through each step before deciding:

STEP 1 — STATISTICAL VALIDITY:
Does the strategy pass all acceptance thresholds?
Be explicit: state each threshold and whether it passes or fails.
Is the Sharpe consistent between the two halves, or concentrated in one period?

STEP 2 — ROBUSTNESS:
Does the strategy show meaningful, consistent edge across both sub-periods?
Compare with previous iterations — is this genuinely better or is it minor variation?
Is the signal economically meaningful or does it look like a numerical artefact?

STEP 3 — BIAS & RISK FLAGS:
Any lookahead bias concerns? Is the holding period consistent with the signal frequency?
Is the turnover realistic given transaction costs?
Are there regime-dependency risks not captured in the backtest?

STEP 4 — DECISION & SPECIFIC SUGGESTION:
State your decision: accept, reject, or refine.
If refining: give ONE concrete, specific change (e.g. "increase z-score lookback from 60 to 120 days",
"switch from long-only to long-short", "use 5s30s instead of 2s10s given flat-curve regime").
Do not suggest vague changes like "improve the signal".

After completing all 4 steps, output your decision as a JSON block.
The JSON must start with ```json and end with ```.

```json
{{
  "decision": "accept|reject|refine",
  "reason": "<one precise sentence — cite specific metric values>",
  "suggestion": "<one concrete actionable change, or 'none' if accepting/rejecting without path forward>"
}}
```"""


def _extract_json(text: str) -> str:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _extract_reasoning(text: str) -> str:
    idx = text.find("```json")
    return text[:idx].strip() if idx != -1 else ""


def _rule_based_fallback(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Hard rules used when LLM parse fails — ensures the loop always continues."""
    sharpe = evaluation.get("sharpe", 0)
    mdd = evaluation.get("max_drawdown", -1)
    h1 = evaluation.get("first_half_sharpe", 0)
    h2 = evaluation.get("second_half_sharpe", 0)

    if sharpe >= 0.5 and mdd >= -0.25 and h1 > 0 and h2 > 0:
        return {"decision": "accept", "reason": "Rule-based fallback: all thresholds passed.", "suggestion": "none"}
    return {
        "decision": "refine",
        "reason": f"Rule-based fallback: Sharpe={sharpe:.2f}, MaxDD={mdd:.2f}, H1={h1:.2f}, H2={h2:.2f}.",
        "suggestion": "Review signal definition and lookback window.",
    }


def critic_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: CoT critic with full iteration memory."""
    llm = get_llm()
    hypothesis = state.get("hypothesis", {})
    evaluation = state.get("evaluation", {})
    history: List[Dict[str, Any]] = state.get("history", [])

    prompt = _PROMPT_TEMPLATE.format(
        signal_name=hypothesis.get("signal_name", "?"),
        hypothesis=hypothesis.get("hypothesis", "?"),
        rationale=hypothesis.get("rationale", "?"),
        sharpe=evaluation.get("sharpe", "?"),
        max_drawdown=evaluation.get("max_drawdown", "?"),
        annualized_return=evaluation.get("annualized_return", "?"),
        avg_turnover=evaluation.get("avg_turnover", "?"),
        n_obs=evaluation.get("n_obs", "?"),
        has_min_history=evaluation.get("has_minimum_history", "?"),
        first_half_sharpe=evaluation.get("first_half_sharpe", "?"),
        second_half_sharpe=evaluation.get("second_half_sharpe", "?"),
        first_half_drawdown=evaluation.get("first_half_drawdown", "?"),
        second_half_drawdown=evaluation.get("second_half_drawdown", "?"),
        history=_format_history(history),
    )

    for attempt in range(3):
        raw = llm.invoke([HumanMessage(content=prompt)])
        text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            decision = json.loads(_extract_json(text))
            assert decision.get("decision") in {"accept", "reject", "refine"}
            state["critic"] = decision
            state["decision"] = decision["decision"]
            state["critic_reasoning"] = _extract_reasoning(text)
            return state
        except (json.JSONDecodeError, AssertionError, TypeError):
            if attempt == 2:
                fallback = _rule_based_fallback(evaluation)
                state["critic"] = fallback
                state["decision"] = fallback["decision"]
                state["critic_reasoning"] = f"Parse failed after 3 attempts. Applied rule-based fallback.\nLast LLM output:\n{text}"

    return state
