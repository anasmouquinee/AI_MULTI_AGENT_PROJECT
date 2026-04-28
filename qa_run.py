"""
qa_run.py
=========
Interactive Q&A script to query the ingested research papers.
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

from config.settings import get_llm
from demo_run import check_prerequisites, setup_index
from rag.retriever import retrieve

# Configure logging to only show critical errors so it doesn't clutter the chat
logging.basicConfig(level=logging.WARNING)

console = Console(force_terminal=True)

QA_PROMPT = """You are a helpful academic research assistant.
Answer the user's question using ONLY the provided context from scientific papers.
Synthesize information from MULTIPLE sources if they are provided in the context.
If the context does not contain the answer, politely say that you don't know based on the provided documents.
Do not make up facts outside the context.
Include inline citations like [filename.pdf] when you use a specific piece of information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER IN MARKDOWN FORMAT:"""


def run_qa_loop():
    """Runs the interactive Q&A loop."""
    console.print(Panel.fit(
        "[bold cyan]Interactive Research Q&A[/bold cyan]\n"
        "[dim]Ask questions about your ingested PDFs![/dim]\n"
        "[dim]Type 'exit' or 'quit' to leave.[/dim]",
        border_style="cyan",
    ))

    # Re-use logic from demo_run.py
    console.print("\n[bold]1. Verifying prerequisites...[/bold]")
    check_prerequisites()

    console.print("\n[bold]2. Loading vector index...[/bold]")
    index = setup_index()

    llm = get_llm()

    console.print("\n" + "=" * 60)
    console.print("[bold green]READY! Ask your questions below.[/bold green]")
    console.print("=" * 60 + "\n")

    while True:
        try:
            # Get user input
            question = console.input("[bold yellow]You:[/bold yellow] ").strip()
            
            if question.lower() in ['exit', 'quit']:
                console.print("\n[cyan]Goodbye![/cyan]")
                break
            
            if not question:
                continue

            # Retrieve context
            console.print("[dim]Searching papers...[/dim]")
            results = retrieve(index, query=question, top_k=15)
            
            if not results:
                console.print("[red]No relevant passages found.[/red]\n")
                continue

            # Format context
            context_blocks = []
            for i, res in enumerate(results):
                source = res["source"]
                text = res["text"].replace('\n', ' ').strip()
                context_blocks.append(f"--- Excerpt {i+1} from [{source}] ---\n{text}")
            
            full_context = "\n\n".join(context_blocks)

            # Generate answer
            console.print("[dim]Generating answer...[/dim]")
            prompt = QA_PROMPT.format(context=full_context, question=question)
            
            response = llm.invoke(prompt)

            # Print answer
            console.print(Panel(
                Markdown(response.content),
                title="AI Assistant",
                border_style="green",
            ))

        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"\n[red]An error occurred: {e}[/red]\n")


if __name__ == "__main__":
    run_qa_loop()
