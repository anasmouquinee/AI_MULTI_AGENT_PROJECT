"""
rag/retriever.py
================
Semantic query engine over the vector index.
Returns relevant passages with source metadata.
"""
import logging
from typing import Optional

from llama_index.core import VectorStoreIndex

from config.settings import TOP_K

logger = logging.getLogger(__name__)


def get_query_engine(index: VectorStoreIndex, top_k: int = TOP_K):
    """
    Creates a query engine configured for semantic retrieval.

    Args:
        index: VectorStoreIndex already built or loaded.
        top_k: number of most relevant fragments to retrieve.

    Returns:
        LlamaIndex QueryEngine.
    """
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="no_text",  # Only retrieve nodes, without generating response
    )
    return query_engine


def get_retriever(index: VectorStoreIndex, top_k: int = TOP_K):
    """
    Creates a retriever that returns nodes without text generation.

    Args:
        index: VectorStoreIndex.
        top_k: number of fragments to retrieve.

    Returns:
        LlamaIndex Retriever.
    """
    return index.as_retriever(similarity_top_k=top_k)


def retrieve(
    index: VectorStoreIndex,
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Executes a semantic query and returns structured results.

    Args:
        index: VectorStoreIndex.
        query: natural language query.
        top_k: number of fragments to retrieve.

    Returns:
        List of dicts with keys: text, source, score, page.
    """
    logger.info("Executing semantic query query=%s top_k=%d", query[:80], top_k)

    retriever = get_retriever(index, top_k)
    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        metadata = node.metadata or {}
        results.append(
            {
                "text": node.get_text(),
                "source": metadata.get("file_name", "unknown"),
                "score": round(node.score, 4) if node.score else None,
                "page": metadata.get("page_label", "N/A"),
            }
        )

    logger.info("Results retrieved result_count=%d", len(results))

    return results
