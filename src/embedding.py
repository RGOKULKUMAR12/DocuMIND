# Backward-compatibility alias for src.embeddings
from src.embeddings import (
    model,
    embed,
    embed_query,
    calculate_similarity
)

__all__ = ["model", "embed", "embed_query", "calculate_similarity"]
