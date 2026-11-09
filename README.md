# Zyro Intelligent RAG - HR Policy Assistant

An end-to-end, enterprise-ready Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **LangChain**, **HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)**, **FAISS**, and **Groq (`openai/gpt-oss-20b`)**.

Designed specifically for **Zyro Dynamics HR Help Desk** to provide factual, verifiable answers strictly grounded in official HR policy documents with exact citations (source file name, page number, and similarity score), accompanied by a strict safeguard against hallucination and out-of-scope inquiries.

---

## 🏗️ Architecture & RAG Pipeline

```
[Employee Question]
         │
         ▼
[HuggingFace Embeddings] ──► [all-MiniLM-L6-v2 (384-d, normalized)]
                                        │
                                        ▼
                             [FAISS Vector Store]
                                        │
                       [Similarity Search: Squared L2 (d²)]
                                        │
                       Is distance <= calibrated threshold?
                                   /         \
                             [YES]             [NO]
                               /                 \
        [Retrieve Top-k Chunks]             [Refusal Safeguard]
                   │                        "Documents do not contain
                   ▼                         information on this..."
         [Strict Grounding Prompt]
                   │
                   ▼
      [Groq LLM (openai/gpt-oss-20b)]
                   │
                   ▼
[Grounded Answer + Verified Citations]
```

### Core Pipeline Steps:
1. **Document Ingestion (`PyPDFLoader`)**:
   - Reads uploaded PDF files page by page.
   - Detects and warns about empty documents, corrupted files, and scanned/image-only PDFs lacking digital text.
   - Tracks clean filenames and 1-based page numbers.
2. **Text Chunking (`RecursiveCharacterTextSplitter`)**:
   - Splits text into 600-character passages with 100-character overlap.
   - Respects natural separators (`\n\n`, `\n`, sentence endings).
   - Explicitly filters out blank or empty chunks.
3. **Embedding Generation (`sentence-transformers/all-MiniLM-L6-v2`)**:
   - Encodes chunks into 384-dimensional dense vectors locally on CPU.
   - Normalizes embeddings to unit length.
4. **Vector Store (`FAISS`)**:
   - Indexes embeddings in an in-memory `IndexFlatL2`.
5. **Distance Calibration & Safeguards**:
   - **Mathematical Interpretation**: FAISS `IndexFlatL2` computes **Squared Euclidean Distance ($L_2^2$)**. For unit-normalized vectors:
     $$\text{Squared } L_2 = 2 \times (1 - \text{cosine\_similarity})$$
     Because cosine similarity ranges from $-1$ to $1$, the squared $L_2$ distance ranges from **$0.0$** (identical) up to **$4.0$** (diametrically opposed), and is approximately **$2.0$** for orthogonal vectors.
   - **Threshold Calibration**: A single distance threshold is **not** a reliable, standalone HR-domain classifier. Paraphrased valid HR questions might yield higher distances, while non-HR questions using corporate vocabulary might yield moderate distances.
   - **Dual-Layer Safeguard**:
     - *Layer 1 (Retrieval Filter)*: Filters out distant chunks above the configurable threshold.
     - *Layer 2 (Prompt Grounding)*: The LLM runs at `temperature=0.0` with strict system instructions to refuse any question not explicitly evidenced in the retrieved excerpts.
6. **Grounded Generation (`ChatGroq`)**:
   - Configurable via `DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"` (active production model on Groq).
   - Safe, native Streamlit UI citation cards displaying exact PDF names, pages, and chunk excerpts.

---

## 📂 Project Directory Structure

```
Zyro_Intelligent_RAG/
├── app.py                         # Streamlit user interface, status & native citations
├── rag_engine.py                  # Core RAG logic: ingestion, FAISS, retrieval & Groq LLM
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Comprehensive guide, setup, tests & deployment
├── .gitignore                     # Git ignore rules for venv, secrets, cache & private PDFs
├── .streamlit/
│   └── secrets.toml.example       # Example secrets template for Groq API Key
└── data/
    └── hr_policies/
        └── .gitkeep               # Directory for storing internal HR policy PDFs
```

---

## 🚀 Local Setup & Installation

### Step 1: Open your workspace terminal
```bash
cd C:\Users\user\.gemini\antigravity\scratch\Zyro_Intelligent_RAG
```

### Step 2: Create and Activate a Python Virtual Environment
*(Python 3.10 or 3.11 recommended)*
```powershell
# On Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Groq API Key
Get your free API key at [Groq Console](https://console.groq.com/keys).

**Option A (Streamlit Secrets - Recommended):**
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:
```powershell
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```
Edit `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
```

**Option B (Environment Variable):**
```powershell
$env:GROQ_API_KEY="gsk_your_actual_groq_api_key_here"
```

**Option C (UI Password Input):**
Enter the key directly into the sidebar in the browser.

---

## 🖥️ Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```
Open `http://localhost:8501`.

### User Guide:
1. Verify the Groq API key is detected in the sidebar.
2. Upload one or more HR policy PDFs.
3. Click **🚀 Process & Index Documents**.
4. Monitor the parsing warnings (if any file is scanned or empty) and verify the indexed chunk count.
5. In the chat box, ask policy questions and inspect verified citation source files and pages.

---

## 🎯 Test Scenarios & Expected Behavior

| # | Test Scenario | Question Example | Expected Behavior |
|---|---------------|------------------|-------------------|
| 1 | **Supported Leave Policy** | *"How many days of annual paid leave are employees entitled to?"* | Returns exact days allowed with verified citation (`Leave_Policy.pdf`, Page X). |
| 2 | **Supported Attendance Policy** | *"What are the core working hours and attendance expectations?"* | Returns working hour window and tracking guidelines with citation (`Attendance_Policy.pdf`, Page Y). |
| 3 | **Missing HR Policy** | *"What is Zyro's policy regarding corporate pet adoption reimbursement?"* | Triggers standard refusal message; no external knowledge hallucinated. |
| 4 | **Unrelated Out-of-Scope Query** | *"Who won the 2022 FIFA World Cup in Qatar?"* | Triggers refusal safeguard; refuses to answer sports/general trivia. |

> **Note on Testing**: To run authentic functional tests, download the official Zyro HR policy PDFs into `data/hr_policies/` and upload them to the application. Synthetic or fabricated PDFs should never be used as substitutes for authentic evaluation.

---

## 🏆 Kaggle Challenge Requirements & Submission Workflow

### Official Dataset Notice:
The official Kaggle challenge requires answering benchmark questions based on **11 official Zyro Dynamics HR policy PDFs** and generating a submission file.

> **CRITICAL RULE:**
> - Official Kaggle starter notebooks, PDFs, and evaluation datasets must be downloaded directly from the Kaggle competition page.
> - **Do not invent or fabricate** official PDFs, question IDs, or answers.
> - The final evaluation submission must be a CSV file with **exactly 20 rows** and **exactly these two columns**:
>   ```csv
>   question_id,answer
>   ```
> - The Streamlit interactive application and the Kaggle batch scoring script are maintained separately to prevent accidental submission drift.

### Kaggle Batch Evaluation Workflow:
1. Download the 11 official PDFs into `data/hr_policies/`.
2. Place the Kaggle test set (`test.csv` containing `question_id` and `question`) into `kaggle_data/`.
3. In your Kaggle notebook or batch script:
   - Index the 11 official PDFs using `load_pdf_documents()` and `create_vector_store()` from `rag_engine.py`.
   - Iterate through each `question_id`, query the RAG pipeline with `retrieve_relevant_chunks()` and `generate_grounded_answer()`.
   - Save the predictions into a `submission.csv` containing exactly 20 rows and columns: `question_id,answer`.
   - Inspect `submission.csv` before uploading to ensure 20 non-empty rows without extra index columns (`index=False`).

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "feat: Zyro Intelligent RAG HR assistant"
   git branch -M main
   git remote add origin https://github.com/<your-username>/Zyro_Intelligent_RAG.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Community Cloud**:
   - Sign in at [share.streamlit.io](https://share.streamlit.io).
   - Click **New app**, select repo `Zyro_Intelligent_RAG` and main file `app.py`.
3. **Set Secrets in Cloud**:
   - In Settings -> Secrets, add:
     ```toml
     GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
     ```
   - Click **Deploy!**
