"""
app.py
=============================================================================
Zyro Intelligent RAG - Streamlit Web User Interface

Features:
- Secure Groq API Key management (secrets.toml -> environment variable -> sidebar input).
- Multi-PDF document uploader for Zyro HR Policies with handling for:
  * Empty or 0-page PDFs
  * Scanned/image-only PDFs (no extractable text)
  * Parsing warnings & chunk counts
- Configurable RAG calibration (Squared L2 Distance threshold & Top-k chunks).
- Interactive conversational chat history.
- Safe citation rendering using Streamlit native components (st.container, st.caption,
  st.code/st.text) rather than raw unsanitized HTML injection.
- Direct alignment between retrieved context chunks and citation cards.
- Comprehensive user-facing error reporting for invalid API keys and Groq failures.
=============================================================================
"""

import os
import tempfile
import streamlit as st
from rag_engine import (
    load_pdf_documents,
    split_documents_into_chunks,
    get_embedding_function,
    create_vector_store,
    retrieve_relevant_chunks,
    generate_grounded_answer,
    DEFAULT_GROQ_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_DISTANCE_THRESHOLD
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zyro Intelligent RAG - HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_docs_count" not in st.session_state:
    st.session_state.indexed_docs_count = 0

if "indexed_chunks_count" not in st.session_state:
    st.session_state.indexed_chunks_count = 0

if "processed_files_names" not in st.session_state:
    st.session_state.processed_files_names = []

if "ingestion_warnings" not in st.session_state:
    st.session_state.ingestion_warnings = []


# ---------------------------------------------------------------------------
# API Key Discovery (Streamlit Secrets -> Environment Variable -> UI Input)
# ---------------------------------------------------------------------------
def resolve_groq_api_key() -> str:
    """
    Safely retrieves the Groq API key from Streamlit secrets or OS environment,
    ensuring it is never hardcoded or exposed in frontend outputs.
    """
    # 1. Check Streamlit Secrets (.streamlit/secrets.toml)
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            return st.secrets["GROQ_API_KEY"].strip()
    except Exception:
        pass

    # 2. Check Environment Variables
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key

    return ""


# ---------------------------------------------------------------------------
# Citation Display Helper (Streamlit Native Components - Safe Rendering)
# ---------------------------------------------------------------------------
def display_citations(citations: list):
    """
    Safely renders citation metadata using native Streamlit components.
    Avoids injecting untrusted document text or filenames into raw HTML.
    """
    if not citations:
        return

    st.markdown("#### 📌 Verified Source Citations")
    for idx, cite in enumerate(citations):
        with st.container(border=True):
            col_doc, col_page, col_score = st.columns([3, 1, 1])
            with col_doc:
                st.markdown(f"**Document:** `{cite['source_file']}`")
            with col_page:
                st.markdown(f"**Page:** {cite['page_number']}")
            with col_score:
                st.markdown(f"**Dist (L2²):** {cite['score']}")

            st.caption("Excerpt from verified document chunk:")
            # Use st.text to safely render raw text content without HTML/script injection risks
            st.text(cite["snippet"])


# ---------------------------------------------------------------------------
# Sidebar: Knowledge Base & RAG Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/organization.png", width=64)
    st.title("Zyro Dynamics HR")
    st.caption("AI-Powered Employee Help Desk")
    st.markdown("---")

    # API Key Configuration
    st.subheader("🔑 Groq Credentials")
    discovered_key = resolve_groq_api_key()

    if discovered_key:
        groq_api_key = discovered_key
        st.success("✅ Groq API Key loaded securely from secrets/environment.")
    else:
        groq_api_key = st.text_input(
            "Enter Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Your API key is kept in memory during this session and never logged."
        )
        if not groq_api_key:
            st.info("💡 Tip: Add `GROQ_API_KEY = \"...\"` to `.streamlit/secrets.toml` to avoid manual entry.")

    st.markdown("---")

    # Document Upload Section
    st.subheader("📄 Upload HR Policies")
    uploaded_files = st.file_uploader(
        "Upload one or more HR Policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload company policy PDFs (e.g., Leave Policy, Code of Conduct, Attendance)."
    )

    # Retrieval Threshold & Model Calibration Settings
    with st.expander("⚙️ RAG Calibration & Settings", expanded=False):
        selected_model = st.text_input(
            "Groq LLM Model",
            value=DEFAULT_GROQ_MODEL,
            help="Supported production models: openai/gpt-oss-20b, llama-3.3-70b-versatile, llama-3.1-8b-instant."
        )

        st.caption(
            "**Distance Threshold (Squared L2):**\n"
            "FAISS returns squared Euclidean distance. Values range from 0 (identical) "
            "up to 4.0. Lower values are stricter. Calibrate against official benchmark."
        )
        distance_threshold = st.slider(
            "Distance Cutoff",
            min_value=0.2,
            max_value=2.5,
            value=DEFAULT_DISTANCE_THRESHOLD,
            step=0.05,
            help="Cutoff threshold for similarity. Chunks with distance > cutoff are excluded."
        )
        top_k = st.slider("Top Chunks (k)", min_value=1, max_value=8, value=4)

    # Indexing Action Button
    if st.button("🚀 Process & Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file before processing.")
        else:
            with st.spinner("Processing HR PDFs and generating embeddings..."):
                saved_temp_paths = []
                temp_dir = tempfile.mkdtemp()

                try:
                    for uploaded_file in uploaded_files:
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_temp_paths.append(temp_file_path)

                    # Step 1: Load PDF Documents with defensive checks
                    raw_docs, warnings = load_pdf_documents(saved_temp_paths)
                    st.session_state.ingestion_warnings = warnings

                    # Show any parsing warnings (empty pages, scanned PDFs, etc.)
                    for w in warnings:
                        st.warning(f"⚠️ {w}")

                    if not raw_docs:
                        st.error(
                            "No extractable text found across the uploaded PDFs! "
                            "Please check that the PDFs contain digital text and are not empty or scanned images."
                        )
                    else:
                        # Step 2: Split into non-empty chunks
                        chunks = split_documents_into_chunks(raw_docs)

                        if not chunks:
                            st.error("No valid text chunks could be produced from the documents.")
                        else:
                            # Step 3: Embed & Build FAISS Index
                            embed_fn = get_embedding_function()
                            vector_store = create_vector_store(chunks, embed_fn)

                            # Update Session State
                            st.session_state.vector_store = vector_store
                            st.session_state.indexed_docs_count = len(uploaded_files)
                            st.session_state.indexed_chunks_count = len(chunks)
                            st.session_state.processed_files_names = [f.name for f in uploaded_files]

                            st.success(f"Indexed {len(uploaded_files)} PDF(s) into {len(chunks)} text chunks!")

                except Exception as e:
                    st.error(f"Error during indexing pipeline: {str(e)}")
                finally:
                    # Clean up temporary local files
                    for p in saved_temp_paths:
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass

    # Knowledge Base Status Overview
    st.markdown("---")
    st.subheader("📊 Knowledge Base Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Indexed PDFs", st.session_state.indexed_docs_count)
    with col2:
        st.metric("Total Chunks", st.session_state.indexed_chunks_count)

    if st.session_state.processed_files_names:
        with st.expander("📋 Active Documents", expanded=False):
            for name in st.session_state.processed_files_names:
                st.write(f"• `{name}`")

    # Clear Chat History Button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main Chat Application Interface
# ---------------------------------------------------------------------------
st.title("Zyro Intelligent RAG")
st.subheader("Grounded Human Resources Policy Assistant for Zyro Dynamics Employees")
st.markdown(
    "Ask questions regarding Zyro Dynamics official HR policies. "
    "All responses are verified and directly grounded in the uploaded policy documents."
)
st.markdown("---")

# Display existing conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display verified citations using safe native components
        if "citations" in message and message["citations"]:
            display_citations(message["citations"])

# Chat Input Handler
user_query = st.chat_input("Ask a question about Zyro Dynamics HR policies (e.g., 'What is the annual leave entitlement?')...")

if user_query:
    # 1. Verify Vector Store exists
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and index HR policy documents in the sidebar first.")
        st.stop()

    # 2. Verify Groq API Key exists
    if not groq_api_key:
        st.warning("⚠️ Please provide a valid Groq API Key in the sidebar or `.streamlit/secrets.toml`.")
        st.stop()

    # Append and render user query in chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Searching HR documents and generating grounded response..."):
            # Step A: Semantic retrieval using configurable threshold and k
            retrieved = retrieve_relevant_chunks(
                vector_store=st.session_state.vector_store,
                query=user_query,
                top_k=top_k,
                distance_threshold=distance_threshold
            )

            # Step B: Grounded generation with dual-layer safeguard
            active_model = selected_model.strip() if selected_model else DEFAULT_GROQ_MODEL
            result = generate_grounded_answer(
                query=user_query,
                retrieved_docs_with_scores=retrieved,
                groq_api_key=groq_api_key,
                model_name=active_model
            )

            response_text = result["answer"]
            citations = result.get("citations", [])

            # Render response
            st.markdown(response_text)

            # Render safe citations when answer is grounded
            if citations:
                display_citations(citations)

    # Save assistant response to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "citations": citations
    })
