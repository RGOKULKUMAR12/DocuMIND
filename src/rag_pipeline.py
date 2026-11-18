from typing import Dict, Any, Optional, List
from src.retriever import retrieve
from src.llm import generate_answer


def ask(
    question: str,
    source: Optional[str] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Complete Retrieval-Augmented Generation (RAG) pipeline:
    1. Validates user question.
    2. Retrieves top-k semantically relevant chunks from ChromaDB.
    3. Synthesizes a context-grounded answer using Gemini.
    4. Extracts verifiable source citations directly from retrieved chunks.

    Parameters:
        question: The user query or question.
        source: Optional source document name to filter retrieval by.
        top_k: Number of relevant chunks to retrieve (default: 3).

    Returns:
        Dict[str, Any]: Structured result containing:
            - 'question': Original user question
            - 'answer': Generated answer from Gemini
            - 'retrieved_chunks': Full list of retrieved chunk dictionaries
            - 'citations': Extracted source citations with page numbers and similarity scores
    """
    clean_question = question.strip() if question else ""
    if not clean_question:
        return {
            "question": question,
            "answer": "Please provide a valid question.",
            "retrieved_chunks": [],
            "citations": []
        }

    # Step 1: Retrieve relevant document chunks using the existing retriever
    retrieved_chunks = retrieve(
        query=clean_question,
        top_k=top_k,
        source=source
    )

    # Step 2: Handle cases where no document chunks could be found
    if not retrieved_chunks:
        return {
            "question": clean_question,
            "answer": "The provided documents do not contain sufficient information to answer this question.",
            "retrieved_chunks": [],
            "citations": []
        }

    # Step 3: Generate grounded answer using Gemini LLM
    answer = generate_answer(
        question=clean_question,
        context=retrieved_chunks
    )

    # Step 4: Construct verifiable source citations from retrieved context
    citations: List[Dict[str, Any]] = []
    for chunk in retrieved_chunks:
        citations.append({
            "source": chunk.get("source", "Unknown"),
            "page_number": chunk.get("page_number", 1),
            "chunk_id": chunk.get("chunk_id", 1),
            "similarity_score": chunk.get("similarity_score", 0.0),
            "content": chunk.get("content", "")
        })

    return {
        "question": clean_question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "citations": citations
    }
