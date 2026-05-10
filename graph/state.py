from typing import Any, Dict, Optional, TypedDict


class ResearchState(TypedDict, total=False):
    hypothesis: Dict[str, Any]
    data: Dict[str, Any]
    backtest_results: Any
    evaluation: Dict[str, Any]
    critic: Dict[str, Any]
    decision: str
    iteration: int
    error: Optional[str]