from sentence_transformers import SentenceTransformer

# Lazy-loaded model to avoid exhausting memory at import time
_model = None


def _get_model():
    """
    Lazily loads the SentenceTransformer model on first use.
    This prevents memory exhaustion when Streamlit and other heavy imports load simultaneously.
    """
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
                local_files_only=True
            )
        except Exception:
            _model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
    return _model


def embed(chunks, batch_size: int = 16):
    """
    Generates 384-dimensional normalized embeddings for document chunks.
    Accepts either a list of chunk dictionaries (with 'content' key) or a list of strings.
    """
    if not chunks:
        return []

    # Extract text strings if chunk dictionaries are provided
    texts = [c["content"] if isinstance(c, dict) else c for c in chunks]

    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings


def embed_query(query):
    """
    Generates a 384-dimensional normalized embedding for a user query.
    """
    model = _get_model()
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return query_embedding


def calculate_similarity(query_embedding, chunk_embeddings):
    """
    Calculates cosine similarity between a query embedding and chunk embeddings.
    """
    model = _get_model()
    similarities = model.similarity(
        query_embedding,
        chunk_embeddings
    )
    return similarities
