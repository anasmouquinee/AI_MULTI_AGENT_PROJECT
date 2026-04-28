"""
orchestrator/main.py
====================
Main pipeline with LangGraph.
Defines the graph: Collector → Analyst → Writer → Validator
with a conditional retry loop if the Validator rejects the draft.
"""
import logging

from langgraph.graph import END, START, StateGraph

from agents.analyst import run_analyst
from agents.collector import run_collector
from agents.validator import run_validator
from agents.writer import run_writer
from config.settings import MAX_RETRY_COUNT
from orchestrator.state import PipelineState

logger = logging.getLogger(__name__)


# -- Graph nodes ------------------------------------------

def collector_node(state: PipelineState) -> dict:
    """Node 1: retrieves relevant excerpts from the corpus."""
    logger.info("=" * 60)
    logger.info("NODE: Collector -- Retrieving passages from corpus")
    logger.info("=" * 60)

    excerpts = run_collector(state["query"])

    return {"excerpts": excerpts}


def analyst_node(state: PipelineState) -> dict:
    """Node 2: analyzes excerpts to identify trends and gaps."""
    logger.info("=" * 60)
    logger.info("NODE: Analyst -- Analyzing trends and gaps")
    logger.info("=" * 60)

    analysis = run_analyst(state["excerpts"])

    return {"analysis": analysis}


def writer_node(state: PipelineState) -> dict:
    """Node 3: generates the literature review in Markdown."""
    logger.info("=" * 60)
    logger.info("NODE: Writer -- Generating literature review")
    logger.info("=" * 60)

    feedback = state.get("feedback")
    draft = run_writer(
        analysis=state["analysis"],
        excerpts=state["excerpts"],
        feedback=feedback,
    )

    return {"draft": draft}


def validator_node(state: PipelineState) -> dict:
    """Node 4: validates the quality of the draft."""
    logger.info("=" * 60)
    logger.info("NODE: Validator -- Checking quality")
    logger.info("=" * 60)

    result = run_validator(
        draft=state["draft"],
        excerpts=state["excerpts"],
    )

    update = {
        "scores": result["scores"],
        "retry_count": state.get("retry_count", 0) + 1,
    }

    if result["approved"]:
        update["final"] = result["final"]
        update["feedback"] = None
        logger.info("[OK] Document APPROVED by the Validator")
    else:
        update["feedback"] = result["feedback"]
        update["final"] = None
        logger.info("[FAIL] Document REJECTED -- sending feedback to Writer")

    return update


# -- Conditional decision function --------------------------

def should_retry(state: PipelineState) -> str:
    """
    Decides whether the Writer should retry or if the pipeline ends.

    Returns:
        - "writer" if there is feedback and retries are not exceeded.
        - "end" if approved or the retry limit is reached.
    """
    # If it was approved (has final document), end
    if state.get("final"):
        return "end"

    # If it has feedback but can still retry
    retry_count = state.get("retry_count", 0)
    if state.get("feedback") and retry_count <= MAX_RETRY_COUNT:
        logger.info(
            "Retrying writing with feedback retry=%d max=%d",
            retry_count, MAX_RETRY_COUNT,
        )
        return "writer"

    # If retries are exceeded, end with the last draft
    logger.warning("Maximum retries reached, using last draft")
    return "end"


# -- Graph construction -----------------------------------

def build_pipeline() -> StateGraph:
    """
    Builds the LangGraph graph with the flow:

        START → Collector → Analyst → Writer → Validator
                                         ↑         |
                                         |←(retry)←|
                                                    |→ END
    """
    builder = StateGraph(PipelineState)

    # Add the 4 nodes
    builder.add_node("collector", collector_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("writer", writer_node)
    builder.add_node("validator", validator_node)

    # Main sequential flow
    builder.add_edge(START, "collector")
    builder.add_edge("collector", "analyst")
    builder.add_edge("analyst", "writer")
    builder.add_edge("writer", "validator")

    # Conditional edge after Validator
    builder.add_conditional_edges(
        "validator",
        should_retry,
        {
            "writer": "writer",  # Retry writing
            "end": END,          # End pipeline
        },
    )

    graph = builder.compile()

    return graph


def run_pipeline(query: str) -> PipelineState:
    """
    Runs the complete research pipeline.

    Args:
        query: user's research query.

    Returns:
        Final state with the generated document.
    """
    logger.info("Starting multi-agent pipeline query=%s", query[:80])

    graph = build_pipeline()

    # Initial state
    initial_state = {
        "query": query,
        "excerpts": None,
        "analysis": None,
        "draft": None,
        "final": None,
        "feedback": None,
        "retry_count": 0,
        "scores": None,
    }

    # Execute the graph
    final_state = graph.invoke(initial_state)

    logger.info("Pipeline completed")

    return final_state
