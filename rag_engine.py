"""
rag_engine.py
=============================================================================
Zyro Intelligent RAG - Core RAG Engine

This module implements the Retrieval-Augmented Generation (RAG) pipeline:
1. PDF Loading: Reads PDF files page-by-page using PyPDFLoader.
   - Handles empty PDFs, scanned/image-only PDFs (no extractable text), and corrupted files.
2. Text Chunking: Splits pages into manageable passages using RecursiveCharacterTextSplitter.
   - Filters out blank or empty chunks.
3. Vector Embeddings: Encodes chunks using HuggingFace all-MiniLM-L6-v2.
4. FAISS Indexing: Builds an in-memory vector database for similarity search.
5. Similarity Retrieval & Score Calibration:
   - FAISS with IndexFlatL2 computes SQUARED Euclidean (L2^2) distance.
   - Explains threshold calibration clearly (values depend on embedding space, question wording,
     and document domain; a single threshold is not an absolute HR-domain classifier).
6. Grounded Answer Generation:
   - Uses Groq ChatGroq with a currently supported production model: "openai/gpt-oss-20b".
   - Model name is configurable via the DEFAULT_GROQ_MODEL constant.
   - Dual-layer safeguard: Retrieval distance filtering + LLM strict factual grounding prompt.
   - Ensures citations always correspond 1-to-1 with the exact retrieved chunks sent to the LLM.
   - Comprehensive error handling for invalid API keys, rate limits, and network issues.
=============================================================================
"""

import os
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import HuggingFaceEmbeddings with backward and forward compatibility:
# langchain-huggingface is the dedicated modern package; fallback to langchain_community.embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Default Configuration Constants
# ---------------------------------------------------------------------------
# Embedding model: 384 dimensions, light, fast and runs locally on CPU
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings: 600 characters per chunk with 100 character overlap preserves context
DEFAULT_CHUNK_SIZE: int = 600
DEFAULT_CHUNK_OVERLAP: int = 100

# Groq LLM model: "openai/gpt-oss-20b" is an active production model on Groq.
# Configurable via this constant. Other options: "llama-3.3-70b-versatile", "llama-3.1-8b-instant".
DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-20b"

# Retrieval distance threshold (FAISS Squared Euclidean / L2^2 distance):
# IMPORTANT NOTE ON DISTANCE INTERPRETATION & CALIBRATION:
# 1. FAISS with the default IndexFlatL2 computes SQUARED L2 distance: sum((u_i - v_i)^2).
# 2. For unit-normalized vectors:
#       Squared L2 = 2 * (1 - cosine_similarity)
#    Because cosine_similarity ranges from -1 to 1:
#       - Identical vectors: cosine = 1  ==> Squared L2 = 0.0
#       - Orthogonal (unrelated): cosine = 0  ==> Squared L2 = 2.0
#       - Diametrically opposed: cosine = -1 ==> Squared L2 = 4.0
#    Therefore, the score does NOT simply range from 0 to 2; it can reach up to 4.0.
# 3. A distance threshold alone is NOT an infallible HR-domain classifier. Paraphrased or
#    complex questions may have higher distances while still being valid HR inquiries.
#    Conversely, non-HR questions that use workplace-like words might accidentally yield lower scores.
# 4. The initial default (e.g. 1.20) is a starting heuristic and MUST be calibrated and validated
#    against the official 11 HR policy PDFs and evaluation question benchmark.
DEFAULT_DISTANCE_THRESHOLD: float = 1.20

# Standard refusal message for out-of-scope, missing, or unverified inquiries
REFUSAL_MESSAGE: str = (
    "I am sorry, but the provided Zyro HR policy documents do not contain information "
    "to answer this question. Please reach out directly to Zyro HR Help Desk at hr@zyrodynamics.com."
)


def load_pdf_documents(file_paths: List[str]) -> Tuple[List[Document], List[str]]:
    """
    Loads text content from a list of PDF files using LangChain's PyPDFLoader.
    Preserves document metadata including original filename and 1-based page number.

    Includes defensive checks for:
    - Non-existent files
    - Empty or 0-page PDFs
    - Scanned/image-only PDFs that contain no extractable digital text

    Args:
        file_paths: List of local paths to PDF files.

    Returns:
        Tuple:
          - List[Document]: Loaded document pages with non-empty text.
          - List[str]: List of warning messages for empty or unextractable files.
    """
    documents: List[Document] = []
    warnings: List[str] = []

    for path in file_paths:
        filename = os.path.basename(path)
        if not os.path.exists(path):
            warnings.append(f"File not found: '{filename}'")
            continue

        try:
            loader = PyPDFLoader(file_path=path)
            loaded_docs = loader.load()

            if not loaded_docs:
                warnings.append(f"'{filename}' appears to be empty (0 pages loaded).")
                continue

            extracted_text_count = 0
            for doc in loaded_docs:
                clean_text = doc.page_content.strip() if doc.page_content else ""
                if clean_text:
                    extracted_text_count += len(clean_text)
                    doc.page_content = clean_text
                    doc.metadata["source_file"] = filename
                    # PyPDFLoader pages are 0-indexed; add 1 for human-readable page numbering
                    raw_page = doc.metadata.get("page", 0)
                    doc.metadata["page_number"] = int(raw_page) + 1
                    documents.append(doc)

            # Warning if PDF has pages but no extractable text (e.g. scanned image PDF)
            if extracted_text_count == 0:
                warnings.append(
                    f"'{filename}' has {len(loaded_docs)} page(s) but contains no extractable text. "
                    "It may be a scanned/image-only PDF requiring OCR."
                )

        except Exception as e:
            warnings.append(f"Failed to read PDF '{filename}': {str(e)}")

    return documents, warnings


def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """
    Splits long PDF documents into smaller chunks using RecursiveCharacterTextSplitter.
    Respects sentence boundaries (paragraphs, newlines, spaces) to prevent cutting words.
    Explicitly filters out empty or whitespace-only chunks.

    Args:
        documents: List of raw loaded documents.
        chunk_size: Target character count per chunk.
        chunk_overlap: Number of characters overlapping between consecutive chunks.

    Returns:
        List of non-empty chunked Document objects.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    raw_chunks = text_splitter.split_documents(documents)

    # Filter out empty or whitespace-only chunks
    valid_chunks: List[Document] = []
    chunk_counter = 1
    for chunk in raw_chunks:
        text = chunk.page_content.strip()
        if text:
            chunk.page_content = text
            chunk.metadata["chunk_id"] = chunk_counter
            valid_chunks.append(chunk)
            chunk_counter += 1

    return valid_chunks


def get_embedding_function(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Initializes and returns the Hugging Face sentence transformer embedding model.
    Runs locally on CPU without requiring an external API key.
    Vectors are normalized to unit length so Euclidean distance is directly related
    to cosine similarity.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def create_vector_store(chunks: List[Document], embedding_function: HuggingFaceEmbeddings = None) -> FAISS:
    """
    Creates an in-memory FAISS vector index from document chunks and their embeddings.

    Args:
        chunks: List of split text chunks.
        embedding_function: Instance of HuggingFaceEmbeddings.

    Returns:
        FAISS vector store instance ready for similarity search.
    """
    if not chunks:
        raise ValueError("Cannot create vector store: No valid document chunks provided.")

    if embedding_function is None:
        embedding_function = get_embedding_function()

    vector_store = FAISS.from_documents(chunks, embedding_function)
    return vector_store


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = 4,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
) -> List[Tuple[Document, float]]:
    """
    Retrieves the top-k most semantically relevant chunks for a user query.

    Understanding the Distance Score:
    - vector_store.similarity_search_with_score returns (Document, squared_l2_distance).
    - For normalized vectors, Squared L2 = 2 * (1 - cosine_similarity).
    - Lower distance = higher semantic similarity.
    - If distance <= distance_threshold, the chunk is retained.
    - If no chunk satisfies the threshold, an empty list is returned.

    Note on Calibrating distance_threshold:
    - Threshold filtering is an initial coarse filter.
    - A single number cannot definitively determine whether a question is valid HR policy vs out-of-scope.
    - Therefore, the threshold is configurable, and generation relies on a dual-layer safeguard
      (threshold filtering + LLM strict factual grounding prompt).

    Args:
        vector_store: Initialized FAISS index.
        query: User's question string.
        top_k: Maximum number of relevant chunks to retrieve.
        distance_threshold: Cutoff squared L2 distance.

    Returns:
        List of tuples: (Document, squared_l2_score)
    """
    if not query or not query.strip():
        return []

    # similarity_search_with_score returns (doc, distance)
    results_with_scores = vector_store.similarity_search_with_score(query.strip(), k=top_k)

    # Filter chunks based on distance threshold
    filtered_results = [
        (doc, float(score)) for doc, score in results_with_scores if float(score) <= distance_threshold
    ]

    return filtered_results


def generate_grounded_answer(
    query: str,
    retrieved_docs_with_scores: List[Tuple[Document, float]],
    groq_api_key: str,
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Generates a factually grounded answer using Groq LLM based ONLY on retrieved chunks.
    Ensures that citations match 1-to-1 with the EXACT chunks passed to the LLM.

    Dual-Layer Safeguard:
    1. Distance filter (Layer 1): If no chunks pass the retrieval threshold, immediately refuse.
    2. Prompt grounding (Layer 2): The LLM is instructed under zero temperature to answer only
       using the provided context and emit the exact refusal text if the context does not support it.

    Robust Error Handling:
    - Validates presence of groq_api_key.
    - Catches API authentication errors, rate limits, and network connection failures.

    Args:
        query: User's question.
        retrieved_docs_with_scores: Filtered list of (Document, score) tuples.
        groq_api_key: User's Groq API key.
        model_name: Groq model name (default: "openai/gpt-oss-20b").
        temperature: LLM temperature (0.0 for deterministic, factual outputs).

    Returns:
        Dictionary with:
          - 'answer': Generated response or refusal message.
          - 'citations': List of citation dicts (source_file, page_number, snippet, score).
          - 'is_refusal': Boolean indicating if query was refused/out-of-scope.
          - 'error': Optional error message string if an exception occurred.
    """
    # Defensive Check: API Key validation
    clean_api_key = groq_api_key.strip() if groq_api_key else ""
    if not clean_api_key:
        return {
            "answer": "Groq API Key is missing. Please provide a valid key in the sidebar or secrets.toml.",
            "citations": [],
            "is_refusal": True,
            "error": "Missing API Key"
        }

    # Safeguard Layer 1: If no chunks met the retrieval threshold, refuse immediately
    if not retrieved_docs_with_scores:
        return {
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "is_refusal": True,
            "error": None
        }

    # Prepare citations strictly from the exact chunks passed to the prompt
    citations: List[Dict[str, Any]] = []
    context_blocks: List[str] = []

    for i, (doc, score) in enumerate(retrieved_docs_with_scores):
        source = doc.metadata.get("source_file", "Unknown PDF")
        page = doc.metadata.get("page_number", 1)
        content = doc.page_content.strip()

        # Build prompt context block
        context_blocks.append(f"[Excerpt {i+1} | Source: {source} | Page: {page}]\n{content}")

        # Build citation dict for frontend display
        citations.append({
            "source_file": source,
            "page_number": page,
            "score": round(float(score), 4),
            "snippet": content[:300] + ("..." if len(content) > 300 else "")
        })

    formatted_context = "\n\n".join(context_blocks)

    # Prompt Template: Strict Grounding & Anti-Hallucination
    prompt_template = PromptTemplate(
        template="""You are the official Zyro Dynamics HR Help Desk AI Assistant.
Your sole responsibility is to answer employee HR questions accurately, professionally, and strictly using ONLY the official HR context provided below.

==================== OFFICIAL HR CONTEXT ====================
{context}
=============================================================

EMPLOYEE QUESTION:
{question}

STRICT GROUNDING RULES:
1. Answer the question using ONLY facts and policies explicitly mentioned in the context above.
2. Do NOT assume, extrapolate, speculate, or bring in outside world knowledge (such as general labor laws, other companies' practices, or unrelated topics like sports).
3. If the context does NOT contain sufficient factual details to answer the question completely and accurately, or if the question is unrelated, you MUST reply with this EXACT sentence and nothing else:
"{refusal_message}"
4. Do NOT mention citation tags like "[Excerpt 1]" in your prose response.

YOUR GROUNDED ANSWER:""",
        input_variables=["context", "question", "refusal_message"]
    )

    formatted_prompt = prompt_template.format(
        context=formatted_context,
        question=query.strip(),
        refusal_message=REFUSAL_MESSAGE
    )

    try:
        # ChatGroq initialization using standard langchain-groq arguments
        llm = ChatGroq(
            groq_api_key=clean_api_key,
            model_name=model_name,
            temperature=temperature
        )

        response = llm.invoke(formatted_prompt)
        raw_answer = response.content.strip()

        # Check if the LLM triggered refusal (Safeguard Layer 2)
        is_refusal = (
            REFUSAL_MESSAGE.lower() in raw_answer.lower() or
            "do not contain information" in raw_answer.lower()
        )

        return {
            "answer": raw_answer,
            "citations": [] if is_refusal else citations,
            "is_refusal": is_refusal,
            "error": None
        }

    except Exception as e:
        error_str = str(e)
        user_friendly_error = error_str

        # Distinguish common error types for the user
        if "401" in error_str or "invalid_api_key" in error_str.lower():
            user_friendly_error = "Invalid Groq API Key. Please verify your API key at https://console.groq.com/keys."
        elif "429" in error_str or "rate_limit" in error_str.lower():
            user_friendly_error = "Groq API rate limit reached. Please wait a moment and try again."
        elif "model_not_found" in error_str.lower() or "decommissioned" in error_str.lower():
            user_friendly_error = f"Model '{model_name}' is unavailable on Groq. Please select an active model."

        return {
            "answer": f"⚠️ Groq API Error: {user_friendly_error}",
            "citations": [],
            "is_refusal": True,
            "error": error_str
        }
