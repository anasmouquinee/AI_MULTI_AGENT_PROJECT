"""
tools/writing_tools.py
======================
Writing tools for the Writer agent.
- MarkdownFormatterTool: formats analysis as a literature review.
- CitationBuilderTool: generates academic citations in APA format.
"""
import json

from langchain_core.tools import tool

from config.settings import get_llm


@tool
def markdown_formatter_tool(analysis: str) -> str:
    """
    Takes a structured analysis and formats it as a literature review
    in Markdown format with clear sections.

    Args:
        analysis: JSON analysis with trends, comparisons, and gaps.

    Returns:
        Markdown-formatted text containing the literature review.
    """
    llm = get_llm()

    prompt = f"""You are an academic writer. Transform the following structured analysis
into a well-written literature review section in Markdown format.

ANALYSIS:
{analysis}

Requirements:
- Use proper Markdown formatting with headers (##, ###)
- Write approximately 500 words
- Include an Introduction, Main Body (organized by themes), and Conclusion
- Reference specific papers using [Author, Year] style placeholders
- Use academic language and formal tone
- Include transition sentences between sections
- Highlight research gaps and future directions

Write the literature review now:"""

    response = llm.invoke(prompt)
    return response.content


@tool
def citation_builder_tool(sources: str) -> str:
    """
    Generates bibliographic citations in APA format from retrieved source metadata.

    Args:
        sources: JSON with source metadata (file_name, page, etc.).

    Returns:
        List of formatted citations in APA style.
    """
    # Parse sources
    try:
        source_list = json.loads(sources)
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, try to extract file names
        source_list = [{"source": sources}]

    # Generate citations based on arXiv file names
    # Format: Author(s). (Year). Title. arXiv preprint arXiv:XXXX.XXXXX
    citations = []
    seen = set()

    if isinstance(source_list, list):
        items = source_list
    elif isinstance(source_list, dict):
        items = [source_list]
    else:
        items = []

    for item in items:
        source = item.get("source", item.get("file_name", "unknown"))
        if source in seen:
            continue
        seen.add(source)

        # Extract arXiv ID from filename if possible
        clean_name = source.replace(".pdf", "").replace("_", " ")
        citations.append({
            "source_file": source,
            "citation": f"[{clean_name}] — Retrieved from indexed corpus.",
            "reference_key": f"[{clean_name}]",
        })

    return json.dumps({
        "citations": citations,
        "total_sources": len(citations),
    }, indent=2, ensure_ascii=False)
