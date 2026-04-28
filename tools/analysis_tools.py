"""
tools/analysis_tools.py
=======================
Analysis tools for the Analyst agent.
- TrendAnalyzerTool: identifies trends and recurring themes.
- ComparisonTool: compares methodologies between papers.
"""
import json

from langchain_core.tools import tool

from config.settings import get_llm


@tool
def trend_analyzer_tool(excerpts: str) -> str:
    """
    Analyzes a set of excerpts from scientific papers to identify
    trends, recurring themes, and research patterns.

    Args:
        excerpts: text with relevant excerpts from the corpus (JSON or plain text).

    Returns:
        Structured JSON with the identified trends.
    """
    llm = get_llm()

    prompt = f"""You are a research analyst. Analyze the following excerpts from scientific papers
and identify the main trends and recurring themes.

EXCERPTS:
{excerpts}

Respond ONLY with valid JSON in this exact format:
{{
    "trends": [
        {{
            "name": "trend name",
            "description": "brief description of this trend",
            "evidence_count": number_of_excerpts_supporting_this,
            "key_papers": ["paper1.pdf", "paper2.pdf"]
        }}
    ],
    "emerging_topics": ["topic1", "topic2"],
    "dominant_methodology": "description of the most common approach"
}}"""

    response = llm.invoke(prompt)
    return response.content


@tool
def comparison_tool(excerpts: str) -> str:
    """
    Compares the methodologies, approaches, and findings described in different
    excerpts from scientific papers. Detects differences and gaps.

    Args:
        excerpts: text with excerpts from multiple papers.

    Returns:
        JSON with structured comparison and detected gaps.
    """
    llm = get_llm()

    prompt = f"""You are a research analyst. Compare the methodologies and findings
described in the following excerpts from different scientific papers.

EXCERPTS:
{excerpts}

Respond ONLY with valid JSON in this exact format:
{{
    "comparisons": [
        {{
            "aspect": "aspect being compared",
            "approaches": [
                {{
                    "paper": "paper name",
                    "method": "description of their approach",
                    "strength": "key advantage",
                    "limitation": "key limitation"
                }}
            ]
        }}
    ],
    "research_gaps": [
        {{
            "gap": "description of the gap",
            "why_important": "why this gap matters",
            "potential_direction": "possible research direction"
        }}
    ],
    "consensus_points": ["point where papers agree"]
}}"""

    response = llm.invoke(prompt)
    return response.content
