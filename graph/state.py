from typing import Any, Dict, List, Optional, TypedDict


class ResearchState(TypedDict, total=False):
    hypothesis: Dict[str, Any]
    data: Dict[str, Any]
    backtest_results: Any
    evaluation: Dict[str, Any]
    critic: Dict[str, Any]
    decision: str
    iteration: int
    history: List[Dict[str, Any]]      # all previous (hypothesis, evaluation, critic) triples
    reasoning_trace: str               # hypothesis agent CoT scratchpad
    critic_reasoning: str              # critic agent CoT scratchpad
    error: Optional[str]
