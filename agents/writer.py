"""
agents/writer.py
================
Agent 3 — Writer
Role: generate a structured literature review in Markdown.
"""
import json
import logging

from config.settings import get_llm
from tools.writing_tools import citation_builder_tool, markdown_formatter_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Academic Writer Agent specialized in producing structured
literature reviews from research analysis.

Your job is to transform a structured analysis (themes, trends, gaps, comparisons)
into a well-written literature review in Markdown format.

Requirements:
- ~500 words
- Proper sections: Introduction, Body (by theme), Research Gaps, Conclusion
- Academic tone and formal language
- Cite sources using [filename] notation
- Smooth transitions between sections
- Highlight key findings and open challenges"""

TOOLS = [markdown_formatter_tool, citation_builder_tool]


def run_writer(analysis: str, excerpts: str, feedback: str = None) -> str:
    """
    Runs the Writer agent to generate the literature review.

    Args:
        analysis: JSON analysis from the Analyst agent.
        excerpts: original excerpts to generate citations.
        feedback: feedback from the Validator for revisions (optional).

    Returns:
        Markdown text with the literature review.
    """
    logger.info("Running Writer agent")

    # Step 1: generate citations from sources
    logger.info("Building citations...")
    citations = citation_builder_tool.invoke(excerpts)

    # Step 2: generate the draft with the formatter
    logger.info("Formatting literature review...")

    # If there is feedback, include it as additional instructions
    feedback_section = ""
    if feedback:
        feedback_section = f"""

REVISION INSTRUCTIONS (from quality reviewer):
{feedback}

Please address ALL the issues mentioned above in your revised draft."""

    # Prepare the complete input for the formatter
    full_input = f"""ANALYSIS:
{analysis}

CITATIONS:
{citations}
{feedback_section}"""

    draft = markdown_formatter_tool.invoke(full_input)

    # Step 3: post-processing — ensure clean Markdown format
    llm = get_llm()

    polish_prompt = f"""You are an academic editor. Polish the following literature review draft.
Ensure it:
1. Has proper Markdown headers (##, ###)
2. Includes a References section at the end using the citations below
3. Is approximately 500 words
4. Has smooth transitions between sections
5. Uses academic language consistently

DRAFT:
{draft}

AVAILABLE CITATIONS:
{citations}

Return the polished literature review in Markdown format. Nothing else."""

    logger.info("Polishing final draft...")
    polished = llm.invoke(polish_prompt)

    logger.info("Writer completed")

    return polished.content
