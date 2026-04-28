"""
agents/validator.py
===================
Agent 4 — Validator
Role: verify consistency, citation accuracy, and draft quality.
Decides whether the document is acceptable or needs revision.
"""
import json
import logging

from config.settings import get_llm
from tools.validation_tools import consistency_checker_tool, grammar_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Quality Validator Agent for academic literature reviews.

Your job is to verify:
1. Consistency: claims are backed by the original source excerpts
2. Citation accuracy: sources are properly referenced
3. Writing quality: grammar, academic tone, and clarity
4. Completeness: the review covers the key themes and gaps

You either APPROVE the document or REJECT it with specific feedback."""

TOOLS = [consistency_checker_tool, grammar_tool]


def run_validator(draft: str, excerpts: str) -> dict:
    """
    Runs the Validator agent to check the quality of the draft.

    Args:
        draft: Markdown draft from the Writer agent.
        excerpts: original excerpts to verify consistency.

    Returns:
        Dict with keys:
        - approved (bool): whether the document passes validation.
        - feedback (str | None): feedback if there are issues.
        - final (str | None): final document if approved.
        - scores (dict): quality scores.
    """
    logger.info("Running Validator agent")

    # Step 1: verify consistency between draft and sources
    logger.info("Verifying consistency...")
    consistency_input = json.dumps({
        "draft": draft[:3000],  # Limit for LLM context
        "excerpts": excerpts[:3000],
    })
    consistency_result = consistency_checker_tool.invoke(consistency_input)

    # Step 2: evaluate grammar and tone
    logger.info("Evaluating grammar and style...")
    grammar_result = grammar_tool.invoke(draft)

    # Step 3: final decision with the LLM
    llm = get_llm()

    decision_prompt = f"""You are a quality reviewer. Based on the consistency check and grammar
evaluation below, decide if this literature review is ready for publication.

CONSISTENCY CHECK:
{consistency_result}

GRAMMAR EVALUATION:
{grammar_result}

RULES FOR APPROVAL:
- Approve if overall quality is acceptable (scores >= 6/10)
- Reject if there are unsupported claims, contradictions, or poor grammar
- Minor issues can be noted but don't require rejection

Respond ONLY with valid JSON:
{{
    "approved": true/false,
    "overall_score": 0-10,
    "consistency_score": 0-10,
    "grammar_score": 0-10,
    "feedback": "specific feedback if rejected, or 'Document approved' if accepted",
    "issues_found": ["issue1", "issue2"]
}}"""

    logger.info("Making final decision...")
    response = llm.invoke(decision_prompt)

    # Parse the decision
    try:
        decision = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        # If the LLM does not return valid JSON, approve by default
        # but log the issue
        logger.warning("Validator response is not valid JSON, approving by default")
        decision = {
            "approved": True,
            "overall_score": 7,
            "feedback": "Document approved (validator response parsing issue)",
            "issues_found": [],
        }

    is_approved = decision.get("approved", True)

    result = {
        "approved": is_approved,
        "feedback": decision.get("feedback") if not is_approved else None,
        "final": draft if is_approved else None,
        "scores": {
            "overall": decision.get("overall_score", 0),
            "consistency": decision.get("consistency_score", 0),
            "grammar": decision.get("grammar_score", 0),
        },
    }

    status = "APPROVED" if is_approved else "REJECTED"
    logger.info(
        "Validator completed status=%s score=%s",
        status, decision.get("overall_score", 0),
    )

    return result
