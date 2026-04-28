"""
agents/collector.py
===================
Agent 1 — Collector (RAG Agent)
Role: query the indexed corpus to retrieve relevant passages.
Reasoning: ReAct (tool loop).
"""
import json
import logging

from config.settings import get_llm
from tools.rag_tools import keyword_extractor_tool, rag_query_tool

logger = logging.getLogger(__name__)

# System prompt defining the role and behavior of the agent
SYSTEM_PROMPT = """You are a Research Collector Agent specialized in academic literature retrieval.

Your job is to find the most relevant passages from a corpus of indexed scientific papers
to answer a research query.

WORKFLOW:
1. First, use the keyword_extractor_tool to identify key terms from the query
2. Then, use the rag_query_tool to search the corpus with the original query
3. If needed, refine your search using extracted keywords as additional queries
4. Compile all relevant excerpts with their source metadata

OUTPUT FORMAT:
Return a JSON object with this structure:
{
    "query": "the original research query",
    "excerpts": [
        {
            "text": "relevant passage text",
            "source": "filename.pdf",
            "page": "page number",
            "relevance": "why this passage is relevant"
        }
    ],
    "total_found": number
}

Be thorough but selective — only include passages that directly relate to the query."""

# Available tools for this agent
TOOLS = [rag_query_tool, keyword_extractor_tool]


def create_collector_agent():
    """
    Creates the Collector agent with ReAct reasoning.
    Uses create_react_agent from LangGraph for the tool loop.
    """
    from langgraph.prebuilt import create_react_agent

    llm = get_llm()
    agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
    return agent


def run_collector(query: str) -> str:
    """
    Runs the Collector agent directly without ReAct (simplified mode).
    More reliable with local models that do not always support tool calling.

    Args:
        query: user's query about the research topic.

    Returns:
        Text with the relevant excerpts and their metadata.
    """
    logger.info("Running Collector agent query=%s", query[:80])

    # Step 1: extract keywords
    keywords_result = keyword_extractor_tool.invoke(query)
    logger.info("Keywords extracted")

    # Step 2: search in the corpus with the original query
    main_results = rag_query_tool.invoke(query)
    logger.info("Main search completed")

    # Step 3: try a complementary search with refined keywords
    try:
        keywords_data = json.loads(keywords_result)
        top_keywords = keywords_data.get("keywords", [])[:5]
        if top_keywords:
            refined_query = " ".join(top_keywords)
            refined_results = rag_query_tool.invoke(refined_query)
        else:
            refined_results = "[]"
    except (json.JSONDecodeError, TypeError):
        refined_results = "[]"

    # Step 4: combine results without duplicates
    try:
        main_list = json.loads(main_results)
        refined_list = json.loads(refined_results)
    except (json.JSONDecodeError, TypeError):
        main_list = []
        refined_list = []

    # Deduplicate by source + partial content
    seen_texts = set()
    combined = []

    for item in main_list + refined_list:
        text_key = item.get("text", "")[:100]
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            combined.append(item)

    output = {
        "query": query,
        "excerpts": combined,
        "total_found": len(combined),
    }

    logger.info("Collector completed total_excerpts=%d", len(combined))

    return json.dumps(output, indent=2, ensure_ascii=False)
