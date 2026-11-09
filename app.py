"""
Zyro Intelligent RAG - Streamlit Web User Interface

Automatically loads HR policy PDFs from data/hr_policies/ when the app starts.
"""

import os
from pathlib import Path

import streamlit as st

from rag_engine import (
    load_pdf_documents,
    split_documents_into_chunks,
    get_embedding_function,
    create_vector_store,
    retrieve_relevant_chunks,
    generate_grounded_answer,
    DEFAULT_GROQ_MODEL,
    DEFAULT_DISTANCE_THRESHOLD,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Zyro Intelligent RAG - HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# AUTOMATICALLY LOAD HR POLICY PDF FILES
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
POLICY_DIR = PROJECT_DIR / "data" / "hr_policies"


@st.cache_resource(show_spinner=False)
def build_knowledge_base():

    pdf_paths = sorted(
        str(path) for path in POLICY_DIR.glob("*.pdf")
    )

    if not pdf_paths:
        return None, 0, 0, [], [
            f"No PDF files were found in: {POLICY_DIR}"
        ]

    raw_docs, warnings = load_pdf_documents(pdf_paths)

    if not raw_docs:
        return None, len(pdf_paths), 0, [
            Path(p).name for p in pdf_paths
        ], warnings + [
            "No extractable text was found in the HR policy PDFs."
        ]

    chunks = split_documents_into_chunks(raw_docs)

    if not chunks:
        return None, len(pdf_paths), 0, [
            Path(p).name for p in pdf_paths
        ], warnings + [
            "No text chunks could be created from the HR policy PDFs."
        ]

    embedding_function = get_embedding_function()

    vector_store = create_vector_store(
        chunks,
        embedding_function
    )

    return (
        vector_store,
        len(pdf_paths),
        len(chunks),
        [Path(p).name for p in pdf_paths],
        warnings,
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "knowledge_base_loaded" not in st.session_state:
    st.session_state.knowledge_base_loaded = False
    st.session_state.vector_store = None
    st.session_state.indexed_docs_count = 0
    st.session_state.indexed_chunks_count = 0
    st.session_state.processed_files_names = []
    st.session_state.ingestion_warnings = []


if not st.session_state.knowledge_base_loaded:

    try:
        with st.spinner(
            "Loading HR policy PDFs and preparing the knowledge base..."
        ):

            (
                st.session_state.vector_store,
                st.session_state.indexed_docs_count,
                st.session_state.indexed_chunks_count,
                st.session_state.processed_files_names,
                st.session_state.ingestion_warnings,
            ) = build_knowledge_base()

        st.session_state.knowledge_base_loaded = True

    except Exception as exc:

        st.session_state.ingestion_warnings = [
            f"Knowledge-base initialization failed: {exc}"
        ]

        st.session_state.knowledge_base_loaded = True


# --------------------------------------------------
# GROQ API KEY
# --------------------------------------------------

def resolve_groq_api_key():

    try:
        if (
            "GROQ_API_KEY" in st.secrets
            and st.secrets["GROQ_API_KEY"]
        ):
            return str(
                st.secrets["GROQ_API_KEY"]
            ).strip()

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "").strip()


# --------------------------------------------------
# DISPLAY CITATIONS
# --------------------------------------------------

def display_citations(citations):

    if not citations:
        return

    st.markdown("#### 📌 Verified Source Citations")

    for cite in citations:

        with st.container(border=True):

            col_doc, col_page, col_score = st.columns(
                [3, 1, 1]
            )

            with col_doc:
                st.markdown(
                    f"**Document:** `{cite['source_file']}`"
                )

            with col_page:
                st.markdown(
                    f"**Page:** {cite['page_number']}"
                )

            with col_score:
                st.markdown(
                    f"**Dist (L2²):** {cite['score']}"
                )

            st.caption(
                "Excerpt from verified document chunk:"
            )

            st.text(cite["snippet"])


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/organization.png",
        width=64
    )

    st.title("Zyro Dynamics HR")
    st.caption("AI-Powered Employee Help Desk")

    st.markdown("---")

    st.subheader("🔑 Groq Credentials")

    discovered_key = resolve_groq_api_key()

    if discovered_key:

        groq_api_key = discovered_key

        st.success(
            "✅ Groq API Key loaded securely from secrets/environment."
        )

    else:

        groq_api_key = st.text_input(
            "Enter Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Your API key is kept in memory during this session and never logged."
        )

        if not groq_api_key:

            st.info(
                '💡 Add `GROQ_API_KEY = "..."` '
                "to Streamlit Cloud Secrets."
            )

    st.markdown("---")

    st.subheader("📚 HR Policy Knowledge Base")

    if st.session_state.vector_store is not None:

        st.success(
            "Knowledge base loaded automatically."
        )

    else:

        st.error(
            "Knowledge base could not be loaded."
        )

    st.caption("PDF folder: `data/hr_policies/`")

    for warning in st.session_state.ingestion_warnings:
        st.warning(warning)

    with st.expander(
        "⚙️ RAG Calibration & Settings",
        expanded=False
    ):

        selected_model = st.text_input(
            "Groq LLM Model",
            value=DEFAULT_GROQ_MODEL,
        )

        st.caption(
            "**Distance Threshold (Squared L2):** "
            "Lower values are stricter."
        )

        distance_threshold = st.slider(
            "Distance Cutoff",
            min_value=0.2,
            max_value=2.5,
            value=float(DEFAULT_DISTANCE_THRESHOLD),
            step=0.05,
        )

        top_k = st.slider(
            "Top Chunks (k)",
            min_value=1,
            max_value=8,
            value=4,
        )

    st.markdown("---")

    st.subheader("📊 Knowledge Base Status")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Indexed PDFs",
            st.session_state.indexed_docs_count
        )

    with col2:
        st.metric(
            "Total Chunks",
            st.session_state.indexed_chunks_count
        )

    if st.session_state.processed_files_names:

        with st.expander(
            "📋 Active Documents",
            expanded=False
        ):

            for name in st.session_state.processed_files_names:
                st.write(f"• `{name}`")

    if st.button(
        "🔄 Reload HR Policy PDFs",
        use_container_width=True
    ):

        build_knowledge_base.clear()

        st.session_state.knowledge_base_loaded = False

        st.rerun()

    if st.session_state.messages:

        if st.button(
            "🗑️ Clear Chat History",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()


# --------------------------------------------------
# MAIN CHAT INTERFACE
# --------------------------------------------------

st.title("Zyro Intelligent RAG")

st.subheader(
    "Grounded Human Resources Policy Assistant for Zyro Dynamics Employees"
)

st.markdown(
    "Ask questions about Zyro Dynamics official HR policies. "
    "Responses are generated using the policy PDFs stored with this application."
)

st.markdown("---")


# DISPLAY CHAT HISTORY

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("citations"):

            display_citations(
                message["citations"]
            )


# USER INPUT

user_query = st.chat_input(
    "Ask a question about Zyro Dynamics HR policies "
    "(e.g., 'What is the annual leave entitlement?')..."
)


# --------------------------------------------------
# CHAT PROCESSING
# --------------------------------------------------

if user_query:

    if st.session_state.vector_store is None:

        st.warning(
            "The HR policy knowledge base is not available. "
            "Please check that the PDF files are committed inside "
            "`data/hr_policies/` in the GitHub repository."
        )

        st.stop()

    if not groq_api_key:

        st.warning(
            "Please configure a valid Groq API key in Streamlit Cloud Secrets."
        )

        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):

        st.markdown(user_query)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching HR documents and generating a grounded response..."
        ):

            retrieved = retrieve_relevant_chunks(
                vector_store=st.session_state.vector_store,
                query=user_query,
                top_k=top_k,
                distance_threshold=distance_threshold,
            )

            active_model = (
                selected_model.strip()
                if selected_model and selected_model.strip()
                else DEFAULT_GROQ_MODEL
            )

            result = generate_grounded_answer(
                query=user_query,
                retrieved_docs_with_scores=retrieved,
                groq_api_key=groq_api_key,
                model_name=active_model,
            )

            response_text = result["answer"]

            citations = result.get(
                "citations",
                []
            )

            st.markdown(response_text)

            if citations:

                display_citations(citations)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "citations": citations,
        }
    )