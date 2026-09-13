# RAG Sense 🧠

> **Understand your documents. Ask anything.**

**RAG Sense** is a production-grade, modular **RAG-Based Document Learning Assistant** built with Python, LangChain, ChromaDB, Sentence Transformers, Groq LLM, and Streamlit. It delivers a conversational ChatGPT-like learning experience with strict per-user data isolation, multi-format document ingestion, persistent SQLite chat threads, and intelligent dual-tier history retrieval.

---

## ✨ Features

- **🔐 User Authentication & Session Management**: Secure user registration, login, and session persistence with cryptographic PBKDF2-HMAC-SHA256 password hashing.
- **🔒 Strict User Data Isolation**: Every document, vector embedding, chat thread, and message is strictly partitioned and inaccessible to any other user.
- **📄 Multi-Format Document Ingestion (up to 100 MB per file)**: Native LangChain document loading and chunking for:
  - **PDF** (`.pdf`) via `PyPDFLoader`
  - **Word** (`.docx`) via `Docx2txtLoader`
  - **Text** (`.txt`) via `TextLoader`
  - **CSV** (`.csv`) via `CSVLoader`
  - **Excel** (`.xlsx`, `.xls`) via `UnstructuredExcelLoader`
  - **JSON** (`.json`) via `JSONLoader`
- **🧩 Semantic Chunking & Metadata Enrichment**: LangChain `RecursiveCharacterTextSplitter` with chunk size and overlap controls, tagging each chunk with `user_id`, `document_id`, `filename`, `chunk_index`, and `page_number`.
- **🧬 Sentence Transformers Embeddings**: High-performance local embeddings with `all-MiniLM-L6-v2` integrated through LangChain `HuggingFaceEmbeddings`.
- **📦 Persistent ChromaDB Vector Store**: Durable vector storage for document knowledge bases and semantic chat history with composite metadata filtering.
- **🎯 Precision LangChain Retrievers**: User-isolated retrievers with optional document scoping (query across all documents or restrict to a single file).
- **⚡ Ultra-Fast Groq LLM**: Accelerated inference using Groq Cloud (`ChatGroq`) running models such as `llama-3.3-70b-versatile`.
- **🌊 True Token Streaming**: Real-time progressive token generation via LangChain Expression Language (LCEL) and Streamlit streaming.
- **💬 Multi-Chat Threads & Persistence**: Create, rename, switch, and delete multiple conversation threads with full SQLite persistence.
- **🧠 Intelligent Relevant History Retrieval**:
  - **Recent Dialog Window**: Includes the last $K_{recent}$ messages as immediate conversational context.
  - **Semantic Older History**: Retrieves earlier relevant conversation points via LangChain vector similarity from ChromaDB, avoiding blind history dumping.
- **📚 Rich Source References & Citations**: Transparent expandable citations displaying document filename, page number (for PDFs), chunk number, and exact context snippets.
- **🎨 Modern ChatGPT-Like Interface**: Sidebar with date grouping (`Today`, `Yesterday`, `Previous 7 days`, `Older`), active thread management, document status badges (`Ready`, `Processing`, `Failed`), and interactive suggestion chips.

---

## 🏛️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Streamlit User Interface                          │
│   (Sidebar Threads · Document Library · Chat Stream · Source Citations)     │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Application Services                            │
│           (auth/service.py · chat/service.py · rag/service.py)               │
└───────────┬───────────────────────────┬─────────────────────────┬───────────┘
            │                           │                         │
            ▼                           ▼                         ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌─────────────────────┐
│   SQLite Database     │   │ User-Partitioned Disk │   │  ChromaDB Vectors   │
│ (users, docs, chats,  │   │  data/users/<uid>/    │   │ (Docs & Chat Hist)  │
│       messages)       │   │      documents/       │   │  Scoped by user_id  │
└───────────────────────┘   └───────────────────────┘   └──────────▲──────────┘
                                                                   │
                                                                   │
┌──────────────────────────────────────────────────────────────────┴──────────┐
│                   LangChain Core AI & RAG Pipeline                          │
│                                                                             │
│   [Document Loaders]        [Recursive Splitter]       [HuggingFace Emb]    │
│   PDF, DOCX, TXT, CSV,       Chunk Size: 1000           all-MiniLM-L6-v2    │
│   XLSX, JSON                 Overlap: 200                                   │
│            │                          │                        │            │
│            └──────────────────────────┼────────────────────────┘            │
│                                       ▼                                     │
│                     [ChromaDB Vector Store & Retriever]                     │
│                        filter={"user_id": user_id}                          │
│                                       │                                     │
│                                       ▼                                     │
│                    [LangChain ChatPromptTemplate (LCEL)]                    │
│     Current Query + Document Context + Relevant History + Recent History    │
│                                       │                                     │
│                                       ▼                                     │
│                        [ChatGroq (LLaMA 3.3 70B)]                           │
│                                       │                                     │
│                                       ▼                                     │
│                       [LangChain StrOutputParser]                           │
│                            Progressive Stream                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Project Structure

```text
rag-sense/
├── app.py                  # Streamlit application bootstrap & routing
├── requirements.txt        # Pinned project dependencies
├── README.md               # Complete project documentation
├── .env.example            # Environment configuration template
├── .env                    # Local environment variables (git-ignored)
├── .gitignore              # Git ignore rules (secrets, venv, data, logs)
│
├── config/
│   ├── __init__.py
│   └── settings.py         # Centralized Pydantic Settings with env validation
│
├── core/
│   ├── __init__.py
│   ├── database.py         # SQLite connection manager & schema (users, docs, chats, msgs)
│   └── exceptions.py       # Domain exception hierarchy (RAGSenseError, LLMError)
│
├── auth/
│   ├── __init__.py
│   ├── models.py           # User & UserInDB models
│   ├── repository.py       # Parameterized SQLite persistence layer for users
│   ├── service.py          # PBKDF2 password hashing & auth business logic
│   └── session.py          # Streamlit session wrapper & user isolation
│
├── documents/
│   ├── __init__.py
│   ├── models.py           # DocumentRecord and DocumentStatus enums
│   └── repository.py       # SQLite document tracking with user isolation
│
├── ingestion/
│   ├── __init__.py
│   ├── loader.py           # LangChain loaders (PDF, DOCX, TXT, CSV, XLSX, JSON)
│   ├── splitter.py         # RecursiveCharacterTextSplitter with metadata enrichment
│   ├── embeddings.py       # LangChain HuggingFaceEmbeddings (all-MiniLM-L6-v2)
│   ├── vector_store.py     # ChromaDB vector store integration & user isolation
│   └── pipeline.py         # Document ingestion pipeline orchestrator
│
├── rag/
│   ├── __init__.py
│   ├── prompts.py          # LangChain ChatPromptTemplate with anti-hallucination rules
│   ├── retriever.py        # LangChain retriever factory with user & doc filters
│   ├── chain.py            # LangChain LCEL RAG chain (prompt | llm | StrOutputParser)
│   └── service.py          # RAGService coordinating retrieval, Groq, and streaming
│
├── chat/
│   ├── __init__.py
│   ├── models.py           # ChatThread and ChatMessage models
│   ├── repository.py       # SQLite persistence for chats and messages
│   ├── history.py          # LangChain ChromaDB semantic chat history retrieval
│   └── service.py          # ChatService orchestrating threads, messages & history
│
├── ui/
│   ├── __init__.py
│   ├── auth_view.py        # Login & Registration interface
│   ├── chat_view.py        # ChatGPT-like conversation UI with streaming & citations
│   ├── document_view.py    # Document upload, status badges, and library
│   ├── qa_view.py          # Direct document Q&A fallback view
│   └── layout.py           # Sidebar navigation, date grouping & dashboard
│
├── utils/
│   ├── __init__.py
│   ├── logger.py           # Dual logging (console + rotating file)
│   └── helpers.py          # Filesystem, diagnostic, and formatting helpers
│
├── data/                   # Persistent data directory (git-ignored)
│   ├── ragsense.db         # SQLite database
│   ├── chroma/             # Persistent ChromaDB vector collections
│   └── users/              # User-partitioned document filesystem
│
├── logs/                   # Application logs (git-ignored)
│   └── rag_sense.log       # Application log file
│
└── tests/
    ├── __init__.py
    ├── test_auth.py        # 9 tests: registration, login, logout, hashing, isolation
    ├── test_chat.py        # 9 tests: threads, messages, history retrieval, isolation
    ├── test_config.py      # 8 tests: settings, validation, helpers, exceptions
    ├── test_ingestion.py   # 14 tests: loaders, splitting, embeddings, ChromaDB
    └── test_rag.py         # 8 tests: prompts, retrievers, LCEL chain, streaming
```

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Modern type hints, dataclasses, and standard library |
| **Orchestration** | LangChain | Core framework for document loading, splitting, embeddings, retrievers, prompts, LCEL chains, and streaming |
| **LLM Provider** | Groq Cloud (`ChatGroq`) | Ultra-fast inference with `llama-3.3-70b-versatile` |
| **Vector Store** | ChromaDB (`langchain-chroma`) | Persistent local vector store with metadata filtering |
| **Embeddings** | Sentence Transformers | Local `all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` |
| **User Interface** | Streamlit | Responsive ChatGPT-like web UI with native streaming |
| **Database** | SQLite3 | Local transactional persistence with foreign key constraints |
| **Security** | PBKDF2-HMAC-SHA256 | 600,000 hash iterations with cryptographically random per-user salt |
| **Validation** | Pydantic v2 | Robust schema validation and configuration management |
| **Testing** | Pytest | Comprehensive unit test suite with mock LLM support |

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.11 or higher (Python 3.12 recommended)
- A free [Groq Cloud API Key](https://console.groq.com/keys)

### 2. Clone and Setup Environment

```bash
# Navigate to project directory
cd rag-sense

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and provide your Groq API key:

```env
APP_NAME=RAG Sense
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO
APP_VERSION=0.1.0

# Database Settings (SQLite)
DATABASE_PATH=data/ragsense.db

# Document Storage & Ingestion
DOCUMENT_STORAGE_DIRECTORY=data/users
MAX_UPLOAD_SIZE_MB=100

# Embeddings & ChromaDB Vector Store
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=data/chroma
CHROMA_COLLECTION_NAME=rag_sense_docs

# Groq LLM Provider
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
RETRIEVAL_K=4

# Chat Threads & Relevant History Settings
RECENT_HISTORY_K=6
RELEVANT_HISTORY_K=3
CHROMA_CHAT_COLLECTION_NAME=rag_sense_chat_history
```

> [!NOTE]
> Never commit your `.env` file to version control. It is automatically ignored by `.gitignore`.

### 5. Run the Application

```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 🧪 Testing

Run the full automated test suite (all 48 tests):

```bash
pytest tests/ -v
```

### Test Suite Breakdown:
- **`tests/test_auth.py`**: User registration, duplicate prevention, password hashing, authentication, session state, and cross-user isolation.
- **`tests/test_chat.py`**: Thread creation, date ordering, renaming, deleting, message persistence, recent history selection, and LangChain vector history retrieval.
- **`tests/test_config.py`**: Environment variable loading, fallback defaults, validation errors, logging setup, and exception handling.
- **`tests/test_ingestion.py`**: Verification of all 6 loaders (PDF, DOCX, TXT, CSV, XLSX, JSON), recursive text splitting, metadata enrichment, ChromaDB indexing, and user data partitioning.
- **`tests/test_rag.py`**: Prompt formatting, retriever filter construction, LCEL chain invocation and streaming, mock LLM execution, and missing API key error handling.

---

## 🔒 Security Architecture

1. **Password Hashing**: Passwords are never stored in plaintext. They are salted with 16 bytes of cryptographically secure random bytes and hashed using `PBKDF2-HMAC-SHA256` with 600,000 iterations.
2. **SQL Injection Prevention**: All SQLite queries across all repositories use parameterized queries (`?`). No SQL strings are concatenated.
3. **Strict User Isolation**:
   - Filesystem: Documents are stored under `data/users/<user_id>/documents/`.
   - Vector Store: Documents are tagged with `user_id`, and retrievers enforce `filter={"user_id": user_id}`.
   - Conversations: Every chat and message verifies ownership in the repository layer before reading, updating, or deleting.
4. **Secret Masking & Clean Errors**: Stack traces and internal paths are logged to rotating log files in `logs/rag_sense.log` and never exposed directly to users in the UI.

---

## ⚠️ Known Limitations & Future Considerations

- **Local Embeddings**: Embeddings run locally on the CPU (`all-MiniLM-L6-v2`). For extremely large documents (1,000+ pages), initial ingestion may take a few seconds. GPU acceleration can be enabled if PyTorch with CUDA is available.
- **SQLite Concurrency**: SQLite with WAL mode is ideal for local, single-node or prototyping deployments. For high-concurrency multi-instance production deployments, SQLite can be swapped for PostgreSQL via SQLAlchemy.
- **Complex Scanned PDFs**: `PyPDFLoader` handles text-based PDFs natively. For scanned image-only PDFs, an OCR pre-processing layer (such as Tesseract or Unstructured OCR) can be added.

---

## 🚢 Deployment Suggestions

1. **Docker Containerization**: Package the app with a lightweight Python 3.12 slim base image, copying `requirements.txt` and exposing port `8501`.
2. **Reverse Proxy (Nginx / Caddy)**: Place behind Nginx or Caddy with SSL termination (`https://`) and WebSocket forwarding (`Upgrade $http_upgrade; Connection "upgrade";`) for responsive Streamlit streaming.
3. **Cloud Hosting**: Ready for one-click deployment to Streamlit Community Cloud, Google Cloud Run, AWS ECS, or Render with persistent volume mounts for `data/` and `logs/`.
