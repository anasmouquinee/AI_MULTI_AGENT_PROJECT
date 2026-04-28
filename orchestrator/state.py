"""
orchestrator/state.py
=====================
Definition of the shared state for the multi-agent pipeline.
Uses TypedDict so LangGraph can manage transitions.
"""
from typing import Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    """
    Shared state between all agents in the pipeline.
    Each agent reads and writes specific keys:

    - query:       user input (immutable).
    - excerpts:    passages retrieved by the Collector.
    - analysis:    structured analysis from the Analyst.
    - draft:       review draft from the Writer.
    - final:       final validated document.
    - feedback:    feedback from the Validator (if there are issues).
    - retry_count: retry counter Writer ↔ Validator.
    - scores:      quality scores from the Validator.
    """
    query: str
    excerpts: Optional[str]
    analysis: Optional[str]
    draft: Optional[str]
    final: Optional[str]
    feedback: Optional[str]
    retry_count: int
    scores: Optional[dict]
