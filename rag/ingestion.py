"""
rag/ingestion.py
================
Loads PDF documents from a directory using LlamaIndex.
Uses PyMuPDFReader for more reliable text extraction.
"""
import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader

logger = logging.getLogger(__name__)


def load_documents(data_dir: str) -> list:
    """
    Reads all PDFs from *data_dir* and returns a list of Documents.

    Args:
        data_dir: path to the directory containing PDFs.

    Returns:
        List of LlamaIndex Document objects with attached metadata.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {data_path}")

    logger.info("Loading PDF documents pdf_count=%d", len(pdf_files))

    # PyMuPDFReader offers better extraction than the default reader
    try:
        from llama_index.readers.file import PyMuPDFReader

        file_extractor = {".pdf": PyMuPDFReader()}
    except ImportError:
        # Fallback to default reader if pymupdf is not installed
        logger.warning("PyMuPDFReader not available, using default reader")
        file_extractor = None

    reader = SimpleDirectoryReader(
        input_dir=str(data_path),
        file_extractor=file_extractor,
        recursive=False,
    )

    documents = reader.load_data()
    logger.info("Documents loaded successfully doc_count=%d", len(documents))

    return documents
