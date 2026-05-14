"""
Hypothesis Agent — Chain-of-Thought with live macro context injection.

Flow:
  1. Fetch current yield curve from FRED (or fall back to a description).
  2. Summarise what previous iterations tried and why they failed.
  3. Ask Llama to reason through 5 explicit steps before committing to a hypothesis.
  4. Parse the reasoning trace and the JSON block separately.
  5. Validate with Pydantic; retry up to 3 times on parse failure.
"""

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError
from langchain_core.messages import HumanMessage

from models.llm import get_llm


SUPPORTED_SIGNAL_NAMES = ["us_2s10s_zscore", "us_5s30s_zscore", "fx_carry"]


class Hypothesis(BaseModel):
    hypothesis: str
    asset_class: str
    instruments: List[str]
    signal_name: str
    signal_definition: str
    trade_rule: str
    holding_period_days: int
    zscore_window: int
    rationale: str


# ---------------------------------------------------------------------------
# Macro context — injected into the prompt so the LLM reasons about real data
# ---------------------------------------------------------------------------

def _get_macro_context() -> str:
    """Fetch the latest FRED yield curve snapshot for the prompt."""
    try:
        from data.loaders.fred import load_us_yields
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        df = load_us_yields(start=start).ffill().dropna()
        if df.empty:
            raise ValueError("empty")
        latest = df.iloc[-1]
        spread_2s10s = latest["us_10y"] - latest["us_2y"]
        spread_5s30s = latest["us_30y"] - latest["us_5y"]
        shape = "inverted" if spread_2s10s < 0 else ("flat" if spread_2s10s < 0.5 else "steep")
        return (
            f"Date: {df.index[-1].date()}\n"
            f"  US 2y  = {latest['us_2y']:.2f}%\n"
            f"  US 5y  = {latest['us_5y']:.2f}%\n"
            f"  US 10y = {latest['us_10y']:.2f}%\n"
            f"  US 30y = {latest['us_30y']:.2f}%\n"
            f"  2s10s  = {spread_2s10s:+.2f} bps  → curve is {shape}\n"
            f"  5s30s  = {spread_5s30s:+.2f} bps"
        )
    except Exception:
        return "Live FRED data unavailable. Reason from general macro principles and recent historical norms."


def _history_summary(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "None — this is the first attempt."

    def _fmt_metric(value: Any) -> str:
        """Format metrics defensively for mixed runtime types."""
        if value is None:
            return "?"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    lines = []
    for i, entry in enumerate(history, 1):
        h = entry.get("hypothesis", {})
        e = entry.get("evaluation", {})
        c = entry.get("critic", {})
        lines.append(
            f"  Attempt {i}: signal='{h.get('signal_name', '?')}' | "
            f"zscore_window={h.get('zscore_window', '?')} | "
            f"Sharpe={_fmt_metric(e.get('sharpe'))} | "
            f"MaxDD={_fmt_metric(e.get('max_drawdown'))} | "
            f"Decision={c.get('decision', '?')} | "
            f"Critic said: '{c.get('reason', '?')}' | "
            f"Suggestion: '{c.get('suggestion', '?')}'"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt template — explicit 5-step CoT before JSON output
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are a senior macro quant researcher with 20 years of experience in rates and FX.

=== LIVE MACRO CONTEXT ===
{macro_context}

=== PREVIOUS FAILED ATTEMPTS ===
{history_summary}

Your task: propose ONE new trading hypothesis. Do NOT repeat a signal that has already been tried.
If previous attempts failed, your reasoning must explain specifically why this attempt is different.

Think step by step through each of the following before writing your hypothesis:

STEP 1 — REGIME ANALYSIS:
What does the current yield curve shape tell you? Is it steep, flat, or inverted?
What macro regime does this imply (tightening cycle, easing, late-cycle, etc.)?
How does the regime favour or disfavour each signal family?

STEP 2 — SIGNAL FAMILY SELECTION:
Given the regime, which signal family has the strongest edge?
You MUST choose signal_name from exactly one of these three supported values:
  - "us_2s10s_zscore"  → long/short 10Y bond based on 2s10s spread z-score
  - "us_5s30s_zscore"  → long/short 30Y bond based on 5s30s spread z-score
  - "fx_carry"         → long/short EURUSD based on USD carry signal
Do NOT invent other signal names — they will not be computed.
If previous attempts used a signal, you must pick a DIFFERENT one.

STEP 3 — INSTRUMENT & PARAMETER SELECTION:
Which specific instruments? What zscore_window (lookback in days) for the z-score?
Default is 60 days. If prior attempts failed, try a meaningfully different window (e.g. 30, 90, 120, 252).
Why these parameters over alternatives?

STEP 4 — ECONOMIC MECHANISM:
What is the transmission mechanism? Why should this signal predict forward returns?
Be specific — invoke carry, mean-reversion, momentum, or flow logic as appropriate.

STEP 5 — FAILURE MODES:
Under what conditions does this signal break down?
How is this attempt specifically addressing the weaknesses from previous iterations?

After completing all 5 steps, output your hypothesis as a JSON block.
The JSON must start with ```json and end with ```.

```json
{{
  "hypothesis": "<one crisp sentence describing the trade>",
  "asset_class": "<Rates | FX | Cross-Asset>",
  "instruments": ["<ticker or FRED series id>", "..."],
  "signal_name": "<must be one of: us_2s10s_zscore, us_5s30s_zscore, fx_carry>",
  "signal_definition": "<how to compute the signal — be precise>",
  "trade_rule": "<when to go long, when to go short, when flat>",
  "holding_period_days": <integer>,
  "zscore_window": <integer, lookback days for z-score, e.g. 30/60/90/120/252>,
  "rationale": "<1-2 sentences synthesising steps 1-4>"
}}
```"""


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _extract_reasoning(text: str) -> str:
    """Return everything before the JSON block as the reasoning trace."""
    idx = text.find("```json")
    return text[:idx].strip() if idx != -1 else ""


def _parse_hypothesis_json(text: str, llm: Any) -> Dict[str, Any]:
    """Parse hypothesis JSON and attempt one LLM-based repair if malformed."""
    candidate = _extract_json(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repair_prompt = (
            "Convert the following content into STRICT valid JSON.\n"
            "Return ONLY a single JSON object with keys exactly:\n"
            "hypothesis, asset_class, instruments, signal_name, signal_definition, trade_rule, holding_period_days, rationale.\n"
            "Rules:\n"
            "- trade_rule must be a plain string, not an object\n"
            "- holding_period_days must be an integer\n"
            "- instruments must be a list of strings\n\n"
            f"Input:\n{candidate}"
        )
        repaired_raw = llm.invoke([HumanMessage(content=repair_prompt)])
        repaired_text = repaired_raw.content if hasattr(repaired_raw, "content") else str(repaired_raw)
        return json.loads(_extract_json(repaired_text))


def _normalize_hypothesis_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model payload to match Hypothesis schema."""
    normalized = dict(payload)

    instruments = normalized.get("instruments")
    if isinstance(instruments, str):
        normalized["instruments"] = [instruments]
    elif isinstance(instruments, list):
        normalized["instruments"] = [str(item) for item in instruments]

    trade_rule = normalized.get("trade_rule")
    if isinstance(trade_rule, dict):
        normalized["trade_rule"] = json.dumps(trade_rule, ensure_ascii=True)
    elif trade_rule is not None:
        normalized["trade_rule"] = str(trade_rule)

    holding_period = normalized.get("holding_period_days")
    if holding_period is not None:
        try:
            normalized["holding_period_days"] = int(holding_period)
        except (TypeError, ValueError):
            pass

    return normalized


def _already_tried(hypothesis: "Hypothesis", history: List[Dict[str, Any]]) -> bool:
    """Return True if this (signal_name, zscore_window) combo was used in a prior iteration."""
    for entry in history:
        h = entry.get("hypothesis", {})
        if (
            h.get("signal_name") == hypothesis.signal_name
            and h.get("zscore_window") == hypothesis.zscore_window
        ):
            return True
    return False


def _dedupe_prompt(tried: List[Dict[str, Any]]) -> str:
    combos = ", ".join(
        f"{h.get('signal_name')}(window={h.get('zscore_window')})" for h in tried
    )
    remaining = [s for s in SUPPORTED_SIGNAL_NAMES if s not in {h.get("signal_name") for h in tried}]
    remaining_str = ", ".join(remaining) if remaining else "any signal with a different zscore_window"
    return (
        f"IMPORTANT: The following (signal, window) combinations have already been backtested: {combos}. "
        f"You MUST produce a hypothesis with a different combination. "
        f"Untried signals: {remaining_str}."
    )


def generate_hypothesis(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: CoT hypothesis generation with macro context + iteration memory."""
    llm = get_llm()
    history: List[Dict[str, Any]] = state.get("history", [])

    macro_context = _get_macro_context()
    history_str = _history_summary(history)

    prompt = _PROMPT_TEMPLATE.format(
        macro_context=macro_context,
        history_summary=history_str,
    )

    for attempt in range(3):
        # On retry after a duplicate, prepend an explicit deduplication instruction
        active_prompt = prompt
        if attempt > 0:
            tried = [e.get("hypothesis", {}) for e in history]
            active_prompt = _dedupe_prompt(tried) + "\n\n" + prompt

        raw = llm.invoke([HumanMessage(content=active_prompt)])
        text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            parsed = _parse_hypothesis_json(text, llm)
            normalized = _normalize_hypothesis_payload(parsed)
            hypothesis = Hypothesis(**normalized)

            if _already_tried(hypothesis, history):
                if attempt == 2:
                    # Accept it on final attempt rather than crashing — better than no hypothesis
                    pass
                else:
                    continue

            state["hypothesis"] = hypothesis.model_dump()
            state["reasoning_trace"] = _extract_reasoning(text)
            return state
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            if attempt == 2:
                raise ValueError(
                    f"Hypothesis agent failed after 3 attempts. Last error: {exc}\nLast output:\n{text}"
                )

    return state
