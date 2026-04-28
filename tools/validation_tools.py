"""
tools/validation_tools.py
=========================
Validation tools for the Validator agent.
- ConsistencyCheckerTool: verifies coherence between the draft and sources.
- GrammarTool: evaluates grammar and academic tone.
"""
import json

from langchain_core.tools import tool

from config.settings import get_llm


@tool
def consistency_checker_tool(draft_and_excerpts: str) -> str:
    """
    Verifies that the claims in the review draft are supported
    by the original excerpts. Detects inconsistencies and unsupported claims.

    Args:
        draft_and_excerpts: JSON with keys 'draft' (the review draft) and 'excerpts' (original excerpts).

    Returns:
        JSON with the result of the consistency check.
    """
    llm = get_llm()

    prompt = f"""You are a quality reviewer for academic papers. Check the consistency
between the literature review draft and the original source excerpts.

INPUT:
{draft_and_excerpts}

Analyze and respond ONLY with valid JSON in this exact format:
{{
    "is_consistent": true/false,
    "issues": [
        {{
            "type": "unsupported_claim" | "contradiction" | "misattribution",
            "description": "what the issue is",
            "location": "where in the draft",
            "suggestion": "how to fix it"
        }}
    ],
    "score": 0-10,
    "summary": "overall assessment"
}}"""

    response = llm.invoke(prompt)
    return response.content


@tool
def grammar_tool(text: str) -> str:
    """
    Evaluates the grammar, style, and academic tone of the draft.
    Suggests corrections to improve writing quality.

    Args:
        text: draft text to evaluate.

    Returns:
        JSON with grammar evaluation and suggestions.
    """
    llm = get_llm()

    prompt = f"""You are an academic writing reviewer. Evaluate the following text for:
1. Grammar and spelling errors
2. Academic tone and formality
3. Clarity and coherence
4. Proper use of transitions

TEXT:
{text}

Respond ONLY with valid JSON in this exact format:
{{
    "grammar_score": 0-10,
    "tone_score": 0-10,
    "clarity_score": 0-10,
    "overall_score": 0-10,
    "errors": [
        {{
            "type": "grammar" | "tone" | "clarity",
            "original": "problematic text",
            "suggestion": "corrected version",
            "explanation": "why this change"
        }}
    ],
    "is_acceptable": true/false,
    "summary": "overall quality assessment"
}}"""

    response = llm.invoke(prompt)
    return response.content
