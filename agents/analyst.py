"""
agents/analyst.py
=================
Agent 2 — Analyst
Role: identify trends, compare methodologies and detect gaps.
Reasoning: Chain-of-Thought (structured prompting).
"""
import json
import logging

from config.settings import get_llm
from tools.analysis_tools import comparison_tool, trend_analyzer_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Research Analyst Agent specialized in academic paper analysis.

Your job is to analyze excerpts from scientific papers and produce a structured analysis
that identifies trends, compares methodologies, and detects research gaps.

You will receive a collection of excerpts retrieved from the corpus by the Collector agent.

OUTPUT FORMAT:
Return a JSON object with this structure:
{
    "themes": [
        {
            "name": "theme name",
            "description": "what this theme covers",
            "supporting_papers": ["paper1.pdf", "paper2.pdf"]
        }
    ],
    "trends": [...],
    "gaps": [
        {
            "description": "research gap description",
            "importance": "why it matters",
            "direction": "potential future work"
        }
    ],
    "methodology_comparison": [...],
    "summary": "high-level summary of the analysis"
}

Be analytical and critical — identify what the field is missing, not just what it has."""

TOOLS = [trend_analyzer_tool, comparison_tool]


def run_analyst(excerpts: str) -> str:
    """
    Runs the Analyst agent with chain of thought.
    Invokes analysis tools sequentially to build
    a structured analysis.

    Args:
        excerpts: JSON excerpts from the Collector agent.

    Returns:
        JSON with the structured analysis (themes, trends, gaps).
    """
    logger.info("Running Analyst agent")

    # Step 1: identify trends in the excerpts
    logger.info("Analyzing trends...")
    trends_result = trend_analyzer_tool.invoke(excerpts)

    # Step 2: compare methodologies
    logger.info("Comparing methodologies...")
    comparison_result = comparison_tool.invoke(excerpts)

    # Step 3: final synthesis with the LLM
    llm = get_llm()

    synthesis_prompt = f"""You are a research analyst. Based on the trend analysis and methodology
comparison below, produce a final structured analysis.

TREND ANALYSIS:
{trends_result}

METHODOLOGY COMPARISON:
{comparison_result}

ORIGINAL EXCERPTS:
{excerpts[:3000]}

Combine everything into a single JSON response with this structure:
{{
    "themes": [
        {{
            "name": "theme name",
            "description": "description",
            "supporting_papers": ["paper1.pdf"]
        }}
    ],
    "trends": [
        {{
            "name": "trend name",
            "description": "description",
            "evidence_count": 0
        }}
    ],
    "gaps": [
        {{
            "description": "gap description",
            "importance": "high/medium/low",
            "direction": "future work suggestion"
        }}
    ],
    "methodology_comparison": [
        {{
            "aspect": "aspect name",
            "findings": "comparative finding"
        }}
    ],
    "summary": "2-3 sentence summary of the entire analysis"
}}

Respond ONLY with valid JSON."""

    logger.info("Synthesizing final analysis...")
    response = llm.invoke(synthesis_prompt)

    logger.info("Analyst completed")

    return response.content
