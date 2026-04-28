"""
tools/rag_tools.py
==================
Retrieval tools for the Collector agent.
- RAGQueryTool: semantic query over the indexed corpus.
- KeywordExtractorTool: extracts key terms to refine searches.
"""
import json
from typing import Optional

from langchain_core.tools import tool

# Global reference to the index — injected when starting the pipeline
_index = None


def set_index(index):
    """Injects the vector index so that tools can use it."""
    global _index
    _index = index


@tool
def rag_query_tool(query: str) -> str:
    """
    Searches for relevant passages in the indexed scientific paper corpus.
    Returns the most similar fragments along with source metadata.

    Args:
        query: natural language query about the research topic.

    Returns:
        JSON text with retrieved passages and their sources.
    """
    if _index is None:
        return json.dumps({"error": "Index not initialized. Run indexing first."})

    from rag.retriever import retrieve

    results = retrieve(_index, query)

    # Format results for the agent
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append({
            "rank": i,
            "text": result["text"][:800],  # Limit length for context
            "source": result["source"],
            "page": result["page"],
            "relevance_score": result["score"],
        })

    return json.dumps(formatted, indent=2, ensure_ascii=False)


@tool
def keyword_extractor_tool(text: str) -> str:
    """
    Extracts keywords and main concepts from a text.
    Useful for refining search queries in the corpus.

    Args:
        text: text from which to extract keywords.

    Returns:
        List of keywords in JSON format.
    """
    # Simple extraction based on frequency and heuristics
    # (does not require LLM, works offline)
    import re
    from collections import Counter

    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Basic English stopwords for research
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "been",
        "from", "that", "this", "with", "they", "will", "each", "make",
        "like", "which", "their", "would", "there", "what", "about",
        "than", "into", "them", "these", "other", "some", "also", "more",
        "very", "when", "such", "only", "could", "after", "where", "most",
        "should", "being", "does", "through", "between", "over", "using",
        "used", "based", "paper", "proposed", "method", "approach", "results",
        "show", "model", "models", "data", "however", "figure", "table",
    }

    filtered = [w for w in words if w not in stopwords]
    word_counts = Counter(filtered)

    # Take the top 15 most frequent words as keywords
    keywords = [word for word, _ in word_counts.most_common(15)]

    return json.dumps({
        "keywords": keywords,
        "total_unique_terms": len(set(filtered)),
    }, indent=2)
