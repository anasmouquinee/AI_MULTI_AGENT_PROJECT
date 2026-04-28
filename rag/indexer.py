"""
rag/indexer.py
==============
Builds and persists the vector index with ChromaDB.
Supports both initial creation and loading of an existing index.
"""
import logging

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from config.settings import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    get_embed_model,
)

logger = logging.getLogger(__name__)


def _get_chroma_collection(persist_dir: str, collection_name: str):
    """Creates or retrieves a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(collection_name)
    return collection


def create_index(
    documents: list,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> VectorStoreIndex:
    """
    Builds a VectorStoreIndex from documents:
    1. Splits into chunks (512 tokens, 64 overlap)
    2. Generates embeddings with BGE-small
    3. Stores in ChromaDB

    Args:
        documents: list of LlamaIndex Documents.
        persist_dir: directory where ChromaDB persists the data.
        collection_name: collection name in ChromaDB.

    Returns:
        VectorStoreIndex ready for queries.
    """
    logger.info(
        "Creating vector index doc_count=%d collection=%s",
        len(documents), collection_name,
    )

    # Set up the embedding model globally
    Settings.embed_model = get_embed_model()

    # Set up ChromaDB as vector store
    collection = _get_chroma_collection(persist_dir, collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Set up the splitter for chunking
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Build the index (chunks + embeddings + storage)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    logger.info(
        "Index created and persisted successfully persist_dir=%s",
        persist_dir,
    )

    return index


def load_existing_index(
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> VectorStoreIndex:
    """
    Loads an existing index from ChromaDB.

    Args:
        persist_dir: directory where ChromaDB data is persisted.
        collection_name: collection name.

    Returns:
        VectorStoreIndex loaded from disk.
    """
    logger.info(
        "Loading existing index from ChromaDB collection=%s",
        collection_name,
    )

    Settings.embed_model = get_embed_model()

    collection = _get_chroma_collection(persist_dir, collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_vector_store(vector_store)

    logger.info("Index loaded successfully")

    return index
