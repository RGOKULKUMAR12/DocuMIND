import os
from pathlib import Path
import streamlit as st

from src.document_loader import load_pdf
from src.text_processing import clean_txt
from src.chunking import create_chunks
from src.embeddings import embed
from src.vector_store import add_documents, get_collection, count_documents
from src.rag_pipeline import ask

# Set page configuration
st.set_page_config(
    page_title="DocuMind – Document Q&A RAG",
    page_icon="📄",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploaded_pdfs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_stored_sources():
    """
    Retrieves the list of unique PDF source document names currently in ChromaDB.
    """
    try:
        collection = get_collection()
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        sources = {m.get("source") for m in metadatas if m and m.get("source")}
        return sorted(list(sources))
    except Exception:
        return []


def process_uploaded_file(uploaded_file) -> int:
    """
    Processes a single uploaded PDF file:
    Saves it, extracts text, cleans text, creates chunks, computes embeddings, and stores in ChromaDB.
    Returns the number of chunks added.
    """
    file_path = UPLOAD_DIR / uploaded_file.name

    # Save uploaded file to disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1. Load pages from PDF
    pages = load_pdf(file_path)
    if not pages:
        return 0

    # 2. Preprocess text
    cleaned_pages = []
    for page in pages:
        cleaned_text = clean_txt(page.get("content", ""))
        if cleaned_text:
            cleaned_pages.append({
                "page_number": page.get("page_number", 1),
                "content": cleaned_text
            })

    if not cleaned_pages:
        return 0

    # 3. Create overlapping chunks
    chunks = create_chunks(
        cleaned_pages,
        chunk_size=500,
        chunk_overlap=50
    )
    if not chunks:
        return 0

    # 4 & 5. Generate embeddings and store in ChromaDB in batches to prevent memory spikes
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        chunk_slice = chunks[i:i + batch_size]
        embeddings = embed(chunk_slice, batch_size=16)
        add_documents(
            chunks=chunk_slice,
            embeddings=embeddings,
            source=uploaded_file.name
        )

    return len(chunks)


# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "stored_sources" not in st.session_state:
    st.session_state.stored_sources = get_stored_sources()


# Sidebar: Document Management & Filters
with st.sidebar:
    st.title("📄 Document Management")
    st.markdown("Upload and manage documents for retrieval-augmented question answering.")

    # File uploader (supports multiple PDFs)
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Process documents button
    if uploaded_files:
        if st.button("⚡ Process Uploaded Documents", use_container_width=True):
            with st.spinner("Processing and indexing documents..."):
                total_new_chunks = 0
                for file in uploaded_files:
                    try:
                        chunks_count = process_uploaded_file(file)
                        total_new_chunks += chunks_count
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")

                st.session_state.stored_sources = get_stored_sources()
                st.success(f"Indexed {len(uploaded_files)} document(s) with {total_new_chunks} total chunks!")

    st.divider()

    # Document Filtering
    st.subheader("🎯 Retrieval Filter")
    stored_sources = st.session_state.stored_sources
    source_options = ["All Documents"] + stored_sources
    selected_source = st.selectbox(
        "Filter answers by document:",
        options=source_options,
        index=0
    )
    active_source = None if selected_source == "All Documents" else selected_source

    st.divider()

    # Database Statistics
    st.subheader("📊 Knowledge Base Stats")
    total_docs_count = count_documents()
    st.metric(label="Total Indexed Chunks", value=total_docs_count)
    st.write(f"**Available Documents ({len(stored_sources)}):**")
    if stored_sources:
        for s in stored_sources:
            st.caption(f"• {s}")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # Clear chat session
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main Page: Header & Chat Interface
st.title("DocuMind – Intelligent Document Q&A")
st.caption("Grounded Question Answering with PyMuPDF, Sentence Transformers, ChromaDB, and Gemini")

# Display previous conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display citations if present in assistant response
        citations = message.get("citations", [])
        if citations:
            with st.expander(f"🔍 Source Citations ({len(citations)} retrieved chunks)"):
                for i, citation in enumerate(citations, 1):
                    score = citation.get("similarity_score", 0.0)
                    src_doc = citation.get("source", "Unknown")
                    page_num = citation.get("page_number", "?")
                    chunk_id = citation.get("chunk_id", "?")
                    content = citation.get("content", "")

                    st.markdown(f"**[{i}] Source: `{src_doc}` | Page: `{page_num}` | Similarity Score: `{score:.4f}`**")
                    st.text(content)
                    if i < len(citations):
                        st.divider()

# User Query Input
user_question = st.chat_input("Ask a question based on your documents...")

if user_question:
    # 1. Display and record user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Check if knowledge base is populated
    if total_docs_count == 0:
        warning_msg = "Please upload and process at least one PDF document before asking questions."
        st.session_state.messages.append({"role": "assistant", "content": warning_msg, "citations": []})
        with st.chat_message("assistant"):
            st.warning(warning_msg)
    else:
        # 3. Execute complete RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating grounded answer..."):
                try:
                    result = ask(
                        question=user_question,
                        source=active_source,
                        top_k=4
                    )

                    answer = result.get("answer", "No response could be generated.")
                    citations = result.get("citations", [])

                    # Render generated answer
                    st.markdown(answer)

                    # Render expandable source inspection
                    if citations:
                        with st.expander(f"🔍 Source Citations ({len(citations)} retrieved chunks)"):
                            for i, citation in enumerate(citations, 1):
                                score = citation.get("similarity_score", 0.0)
                                src_doc = citation.get("source", "Unknown")
                                page_num = citation.get("page_number", "?")
                                content = citation.get("content", "")

                                st.markdown(f"**[{i}] Source: `{src_doc}` | Page: `{page_num}` | Similarity Score: `{score:.4f}`**")
                                st.text(content)
                                if i < len(citations):
                                    st.divider()

                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })

                except Exception as e:
                    error_msg = f"An error occurred while generating the answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "citations": []
                    })
