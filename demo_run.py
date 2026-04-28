"""
demo_run.py
===========
End-to-end demo script for the multi-agent system.
Executes the full pipeline: ingestion -> agents -> final result.
"""
import logging
import os
import sys
from pathlib import Path

# Force UTF-8 in Windows console to avoid encoding errors
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

console = Console(force_terminal=True)

# -- Constants ---
DEMO_QUERY = (
    "What are the main approaches to few-shot learning in NLP, "
    "and what are the open challenges?"
)


def check_prerequisites():
    """Verifies that everything is ready before execution."""
    from config.settings import DATA_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL

    errors = []

    # Verify that there are PDFs in the data directory
    pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    if not pdf_files:
        errors.append(
            f"No PDFs found in {DATA_DIR}.\n"
            f"Run first: python data/download_papers.py"
        )

    # Verify connection to Ollama
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass
        console.print(f"  [OK] Ollama accessible at {OLLAMA_BASE_URL}", style="green")
    except Exception:
        errors.append(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.\n"
            f"Make sure Ollama is running: ollama serve"
        )

    if errors:
        console.print("\n[bold red]ERROR: Prerequisites not met:[/bold red]")
        for err in errors:
            console.print(f"  - {err}", style="red")
        sys.exit(1)

    console.print(f"  [OK] {len(pdf_files)} PDFs found in {DATA_DIR}", style="green")
    console.print(f"  [OK] LLM Model: {OLLAMA_MODEL}", style="green")


def setup_index():
    """Creates or loads the vector index."""
    from config.settings import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, DATA_DIR
    from rag.indexer import create_index, load_existing_index
    from rag.ingestion import load_documents

    chroma_path = Path(CHROMA_PERSIST_DIR)
    force_reindex = "--reindex" in sys.argv

    if chroma_path.exists() and any(chroma_path.iterdir()) and not force_reindex:
        console.print("\nLoading existing index from ChromaDB...", style="cyan")
        index = load_existing_index()
        console.print("  [OK] Index loaded", style="green")
    else:
        if force_reindex:
            console.print("\nForcing new index creation...", style="cyan")
        else:
            console.print("\nCreating new index (first run)...", style="cyan")
        console.print("  This may take a few minutes...", style="dim")

        documents = load_documents(str(DATA_DIR))
        console.print(f"  [OK] {len(documents)} documents loaded", style="green")

        index = create_index(documents)
        console.print("  [OK] Index created and persisted", style="green")

    return index


def run_demo():
    """Executes the complete demo."""
    # Banner
    console.print(Panel.fit(
        "[bold cyan]Multi-Agent Academic Research Assistant[/bold cyan]\n"
        "[dim]Multi-agent system for scientific literature analysis[/dim]\n"
        "[dim]LlamaIndex (RAG) + LangGraph (orchestration) + Ollama (local LLM)[/dim]",
        border_style="cyan",
    ))

    # Step 1: verify prerequisites
    console.print("\n[bold]1. Verifying prerequisites...[/bold]")
    check_prerequisites()

    # Step 2: prepare index
    console.print("\n[bold]2. Preparing vector index...[/bold]")
    index = setup_index()

    # Inject the index into the RAG tools
    from tools.rag_tools import set_index
    set_index(index)

    # Step 3: run pipeline
    console.print("\n[bold]3. Executing multi-agent pipeline...[/bold]")
    console.print(Panel(
        f"[bold]Query:[/bold] {DEMO_QUERY}",
        title="Research Query",
        border_style="yellow",
    ))

    from orchestrator.main import run_pipeline

    console.print("\n[cyan]Pipeline: Collector -> Analyst -> Writer -> Validator[/cyan]\n")
    final_state = run_pipeline(DEMO_QUERY)

    # Step 4: show results
    console.print("\n" + "=" * 60)
    console.print("[bold green]PIPELINE COMPLETED[/bold green]\n")

    # Show scores
    scores = final_state.get("scores")
    if scores:
        console.print(Panel(
            f"Overall: {scores.get('overall', 'N/A')}/10\n"
            f"Consistency: {scores.get('consistency', 'N/A')}/10\n"
            f"Grammar: {scores.get('grammar', 'N/A')}/10",
            title="Quality Scores",
            border_style="green",
        ))

    # Show final document
    final_doc = final_state.get("final") or final_state.get("draft", "No output generated.")
    console.print(Panel(
        Markdown(final_doc),
        title="Literature Review",
        border_style="green",
    ))

    # Save result
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "literature_review.md"
    output_path.write_text(final_doc, encoding="utf-8")
    console.print(f"\nResult saved in: {output_path}", style="green")

    # Show the number of retries
    retry_count = final_state.get("retry_count", 0)
    if retry_count > 1:
        console.print(
            f"The Validator requested {retry_count - 1} revision(s)",
            style="yellow",
        )


if __name__ == "__main__":
    run_demo()
