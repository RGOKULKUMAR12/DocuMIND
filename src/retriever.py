from typing import List, Dict, Any, Optional
import numpy as np
from src.embeddings import embed_query
from src.vector_store import query_chunks, count_documents, get_collection


def retrieve(
    query: str,
    top_k: int = 3,
    source: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Performs similarity search in ChromaDB using the user query.

    Parameters:
        query: User question or search query string.
        top_k: Number of relevant chunks to retrieve (default: 3).
        source: Optional source document name to filter results by.

    Returns:
        List of dictionaries containing chunk text, source, page number,
        chunk ID, and similarity score.
    """
    if not query or not query.strip():
        return []

    # Check if collection has documents stored
    total_docs = count_documents()
    if total_docs == 0:
        return []

    # 1. Generate 384-dimensional query embedding using existing function
    query_embedding = embed_query(query.strip())

    # 2. Query ChromaDB collection
    results = query_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        source=source
    )

    retrieved_chunks: List[Dict[str, Any]] = []

    # ChromaDB returns nested lists for batches: results['documents'][0], etc.
    documents_list = results.get("documents", [[]])[0]
    metadatas_list = results.get("metadatas", [[]])[0]
    distances_list = results.get("distances", [[]])[0]

    for i in range(len(documents_list)):
        doc_text = documents_list[i]
        meta = metadatas_list[i] if i < len(metadatas_list) else {}
        distance = distances_list[i] if i < len(distances_list) else 0.0

        # In ChromaDB with cosine metric (hnsw:space: 'cosine'):
        # distance = 1 - cosine_similarity
        # Therefore, cosine_similarity = 1 - distance
        similarity_score = max(0.0, 1.0 - float(distance))

        retrieved_chunks.append({
            "content": doc_text,
            "source": meta.get("source", "unknown"),
            "page_number": meta.get("page_number", 1),
            "chunk_id": meta.get("chunk_id", i + 1),
            "similarity_score": round(similarity_score, 4)
        })

    # Expand each match with adjacent chunks so headings and split definitions
    # are kept together for answer generation.
    if retrieved_chunks:
        collection = get_collection()
        expanded_by_key = {}
        for chunk in retrieved_chunks:
            key = (chunk["source"], chunk["chunk_id"])
            expanded_by_key[key] = chunk

            for neighbor_id in (chunk["chunk_id"] - 1, chunk["chunk_id"] + 1):
                neighbor = collection.get(
                    where={
                        "$and": [
                            {"source": chunk["source"]},
                            {"chunk_id": neighbor_id},
                        ]
                    },
                    include=["documents", "metadatas"],
                )
                documents = neighbor.get("documents", [])
                metadatas = neighbor.get("metadatas", [])
                if documents:
                    neighbor_meta = metadatas[0] if metadatas else {}
                    neighbor_key = (
                        neighbor_meta.get("source", chunk["source"]),
                        neighbor_meta.get("chunk_id", neighbor_id),
                    )
                    if neighbor_key not in expanded_by_key:
                        expanded_by_key[neighbor_key] = {
                            "content": documents[0],
                            "source": neighbor_key[0],
                            "page_number": neighbor_meta.get("page_number", chunk["page_number"]),
                            "chunk_id": neighbor_key[1],
                            "similarity_score": chunk["similarity_score"],
                        }

        retrieved_chunks = list(expanded_by_key.values())

    return retrieved_chunks


def retrieve_mmr(
    query: str,
    top_k: int = 3,
    candidates: int = 10,
    lambda_param: float = 0.7,
    source: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves chunks using Maximum Marginal Relevance (MMR).
    MMR balances relevance to the query with diversity among selected chunks,
    reducing redundant results.

    MMR formula for each candidate:
        score = lambda * similarity(query, candidate) - (1 - lambda) * max(similarity(candidate, already_selected))

    Parameters:
        query: User question or search query string.
        top_k: Number of final chunks to return (default: 3).
        candidates: Size of the initial candidate pool from ChromaDB (default: 10).
        lambda_param: Balance between relevance and diversity (0 to 1).
                      Higher values favor relevance, lower values favor diversity.
                      Default: 0.7 (slightly favoring relevance).
        source: Optional source document name to filter results by.

    Returns:
        List of dictionaries containing chunk text, source, page number,
        chunk ID, and similarity score.
    """
    if not query or not query.strip():
        return []

    total_docs = count_documents()
    if total_docs == 0:
        return []

    # Ensure we don't request more candidates than available documents
    actual_candidates = min(candidates, total_docs)
    actual_top_k = min(top_k, actual_candidates)

    # 1. Generate query embedding
    query_emb = embed_query(query.strip())
    if isinstance(query_emb, np.ndarray) and query_emb.ndim == 2:
        query_emb = query_emb[0]
    query_emb = np.array(query_emb, dtype=np.float32)

    # 2. Retrieve initial candidate set with their embeddings from ChromaDB
    collection = get_collection()
    where_filter = {"source": source} if source else None

    # Convert query embedding for ChromaDB
    if isinstance(query_emb, np.ndarray):
        query_list = [query_emb.tolist()]
    else:
        query_list = [list(query_emb)]

    results = collection.query(
        query_embeddings=query_list,
        n_results=actual_candidates,
        where=where_filter,
        include=["documents", "metadatas", "distances", "embeddings"]
    )

    documents_list = results.get("documents", [[]])[0]
    metadatas_list = results.get("metadatas", [[]])[0]
    distances_list = results.get("distances", [[]])[0]
    embeddings_list = results.get("embeddings", [[]])[0]

    if not documents_list:
        return []

    # Convert candidate embeddings to numpy array for efficient computation
    candidate_embeddings = np.array(embeddings_list, dtype=np.float32)

    # Convert ChromaDB cosine distances to similarity scores
    candidate_similarities = np.array([max(0.0, 1.0 - d) for d in distances_list], dtype=np.float32)

    # 3. MMR iterative selection
    selected_indices = []
    remaining_indices = list(range(len(documents_list)))

    for _ in range(actual_top_k):
        if not remaining_indices:
            break

        best_idx = None
        best_mmr_score = -float("inf")

        for idx in remaining_indices:
            # Relevance: similarity between query and this candidate
            relevance = candidate_similarities[idx]

            # Diversity penalty: max similarity between this candidate and already selected chunks
            if selected_indices:
                selected_embeddings = candidate_embeddings[selected_indices]
                # Cosine similarity between this candidate and each selected chunk
                # Embeddings are already normalized (from Sentence Transformers), so dot product = cosine similarity
                sim_to_selected = np.dot(selected_embeddings, candidate_embeddings[idx])
                max_sim_to_selected = float(np.max(sim_to_selected))
            else:
                # First selection has no diversity penalty
                max_sim_to_selected = 0.0

            # MMR score: balance relevance and diversity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    # 4. Build result list from selected indices
    retrieved_chunks: List[Dict[str, Any]] = []
    for idx in selected_indices:
        meta = metadatas_list[idx] if idx < len(metadatas_list) else {}
        retrieved_chunks.append({
            "content": documents_list[idx],
            "source": meta.get("source", "unknown"),
            "page_number": meta.get("page_number", 1),
            "chunk_id": meta.get("chunk_id", idx + 1),
            "similarity_score": round(float(candidate_similarities[idx]), 4)
        })

    return retrieved_chunks
