from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings

# Default path for persistent vector database
DEFAULT_DB_PATH = Path("./vector_db")
COLLECTION_NAME = "documind"


def get_client(persist_dir: str | Path = DEFAULT_DB_PATH) -> chromadb.PersistentClient:
    """
    Initializes and returns a persistent ChromaDB client.
    """
    db_path = Path(persist_dir).resolve()
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def get_collection(persist_dir: str | Path = DEFAULT_DB_PATH):
    """
    Retrieves or creates the 'documind' collection with cosine similarity distance metric.
    """
    client = get_client(persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_documents(
    chunks: List[Dict[str, Any]],
    embeddings: Any,
    source: str = "unknown",
    persist_dir: str | Path = DEFAULT_DB_PATH
) -> int:
    """
    Stores chunk texts, 384-dimensional embeddings, and metadata into ChromaDB.
    Uses upsert with deterministic IDs to avoid duplicate records on repeated processing.

    Parameters:
        chunks: List of chunk dictionaries with 'content', 'page_number', and 'chunk_id'.
        embeddings: 2D numpy array or list of 384-dimensional embeddings.
        source: Name of the source PDF document.
        persist_dir: Directory where ChromaDB files are persisted.

    Returns:
        int: Total number of documents currently stored in the collection.
    """
    if not chunks:
        return 0

    collection = get_collection(persist_dir)

    # Convert numpy array to list of lists if needed
    if isinstance(embeddings, np.ndarray):
        embeddings_list = embeddings.tolist()
    else:
        embeddings_list = [
            e.tolist() if isinstance(e, np.ndarray) else list(e)
            for e in embeddings
        ]

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            content = chunk.get("content", "")
            page_number = int(chunk.get("page_number", 1))
            chunk_id = int(chunk.get("chunk_id", i + 1))
            chunk_source = str(chunk.get("source", source))
        else:
            # Fallback if raw text string is passed
            content = str(chunk)
            page_number = 1
            chunk_id = i + 1
            chunk_source = source

        # Deterministic unique ID to prevent duplicates when re-processing
        unique_id = f"{chunk_source}_p{page_number}_c{chunk_id}"

        ids.append(unique_id)
        documents.append(content)
        metadatas.append({
            "source": chunk_source,
            "page_number": page_number,
            "chunk_id": chunk_id
        })

    # ChromaDB upsert avoids duplicates by updating existing IDs and inserting new ones
    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        documents=documents,
        metadatas=metadatas
    )

    total_count = collection.count()
    return total_count


def query_chunks(
    query_embedding: Any,
    top_k: int = 3,
    source: Optional[str] = None,
    persist_dir: str | Path = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Queries ChromaDB collection using a query embedding.

    Parameters:
        query_embedding: 384-dimensional query embedding (list or 1D/2D numpy array).
        top_k: Number of most similar chunks to retrieve.
        source: Optional source document name to filter by.
        persist_dir: Directory where ChromaDB files are persisted.

    Returns:
        Dict: Raw ChromaDB query results containing ids, documents, metadatas, and distances.
    """
    collection = get_collection(persist_dir)

    # Convert numpy array to list if needed
    if isinstance(query_embedding, np.ndarray):
        # Flatten if 2D with single row
        if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_list = query_embedding[0].tolist()
        else:
            query_list = query_embedding.tolist()
    else:
        query_list = list(query_embedding)

    # Wrap in list if 1D list
    if query_list and not isinstance(query_list[0], list):
        query_embeddings = [query_list]
    else:
        query_embeddings = query_list

    where_filter = {"source": source} if source else None

    # Retrieve at most collection.count() if fewer than top_k exist
    available_docs = collection.count()
    actual_k = min(top_k, available_docs) if available_docs > 0 else top_k

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=actual_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    return results


def count_documents(persist_dir: str | Path = DEFAULT_DB_PATH) -> int:
    """
    Returns the total number of documents in the documind collection.
    """
    collection = get_collection(persist_dir)
    return collection.count()


def reset_collection(persist_dir: str | Path = DEFAULT_DB_PATH) -> None:
    """
    Deletes the documind collection to allow a fresh start if needed.
    """
    client = get_client(persist_dir)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
