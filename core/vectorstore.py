import os
import shutil
import time
import gc
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHROMA_PERSIST_DIR  = "codesage_chroma_db"
COLLECTION_NAME     = "codesage_chunks"
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and return the embedding model."""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBEDDING_MODEL,
        encode_kwargs = {"normalize_embeddings": True}
    )
    print("Embedding model ready.")
    return embeddings


def _safe_delete_chroma_dir(path: str) -> None:
    """Safely delete Chroma dir — handles Windows file locks."""
    if not os.path.exists(path):
        return

    gc.collect()

    for attempt in range(5):
        try:
            shutil.rmtree(path)
            print("Cleared existing vector store.")
            return
        except PermissionError:
            if attempt < 4:
                print(f"Waiting for lock release... ({attempt + 1}/5)")
                time.sleep(1)
            else:
                global COLLECTION_NAME
                import uuid
                COLLECTION_NAME = f"codesage_{uuid.uuid4().hex[:8]}"
                print(f"Using new collection: {COLLECTION_NAME}")


def build_vectorstore(
    chunks     : List[Document],
    embeddings : HuggingFaceEmbeddings,
    reset      : bool = False
) -> Chroma:
    """Build Chroma vector store from code chunks."""

    if reset:
        _safe_delete_chroma_dir(CHROMA_PERSIST_DIR)

    print(f"Building vector store with {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        persist_directory = CHROMA_PERSIST_DIR,
        collection_name   = COLLECTION_NAME
    )
    print(f"Vector store built. {vectorstore._collection.count()} vectors stored.")
    return vectorstore


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings
) -> Optional[Chroma]:
    """Load existing vector store if it exists."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None

    vectorstore = Chroma(
        persist_directory  = CHROMA_PERSIST_DIR,
        embedding_function = embeddings,
        collection_name    = COLLECTION_NAME
    )

    count = vectorstore._collection.count()
    if count == 0:
        return None

    print(f"Loaded existing store ({count} vectors).")
    return vectorstore


def close_vectorstore(vectorstore: Optional[Chroma]) -> None:
    """Close Chroma client to release Windows file locks."""
    if vectorstore is None:
        return
    try:
        vectorstore._client.close()
    except Exception:
        pass
    finally:
        gc.collect()
        time.sleep(0.5)