

# 🚀 Enterprise RAG Gateway & Knowledge Pipeline

An enterprise-grade, multi-provider Retrieval-Augmented Generation (RAG) architecture and semantic search gateway built with **FastAPI**, **Qdrant**, **Groq (Llama-3.3)**, **OpenAI**, and **Neural Cross-Encoders**. Designed for high-density document parsing, low-latency hybrid retrieval, strict guardrailed LLM synthesis, and automatic model resilience.

---

## 🏗️ End-to-End System Architecture & Data Flow

```text
                                [ User / Web UI / MCP Client ]
                                              │
                                              ▼
                                    [ FastAPI REST Gateway ]
                                              │
                                              ▼
                             [ Enterprise Query Orchestrator ]
                                              │
      ┌───────────────────────────────┼───────────────────────────────┐
      │                               │                               │
[ Multi-Query Expander ]   [ Production Embedding ]        [ Neural Reranker ]
 (LLM Search Variants)      (Dense Vector Encoding)         (Neural Cross-Encoder)
      │                               │                               │
      └───────────────────────────────┼───────────────────────────────┘
                                              │
                                              ▼
                                   [ Qdrant Vector Cluster ]
                                              │
                                              ▼
                                 [ Parent Context Assembler ]
                                              │
                                              ▼
                                  [ Resilient LLM Engine ]
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
          [ Primary: Groq Llama-3.3 ] ──(Failover)──► [ Fallback: OpenAI GPT-4o-mini ]

⚡ Key Features & Subsystems
Dynamic Multi-Query Expansion: Automatically converts raw user prompts into multiple domain-specific search variations to drastically boost retrieval recall.

Resilient Multi-Provider LLM Gateway: Built-in auto-failover architecture. Operates primarily on Groq Cloud (Llama-3.3-70b-versatile) and seamlessly fails over to OpenAI (GPT-4o-mini) with exponential retry logic via tenacity.

Hybrid Vector Retrieval & Neural Reranking: Executes high-throughput dense vector search on Qdrant DB combined with Cross-Encoder neural reranking (bge-reranker) to filter out noise.

Parent-Child Context Assembly: Preserves deep document hierarchy by retrieving child chunks and assembling enriched parent contexts before feeding the LLM.

Strict Factual Guardrails: System prompts strictly bound responses to verified knowledge chunks to prevent hallucinations and accurately parse complex tabular metrics.

Model Context Protocol (MCP) Integration: Native MCP server compatibility (mcp/server.py) for AI agent interactions.

Modern Cyberpunk UI: Integrated glassmorphism frontend interface featuring real-time vector status indicators and response synthesis rendering.

🧰 Tech Stack
Backend Gateway: FastAPI, Uvicorn, Python 3.11+
Vector Database: Qdrant Cloud / Local Cluster
Embeddings & Reranking: HuggingFace Transformers, PyTorch, BGE Cross-Encode
LLM Synthesis Providers: Groq API (Llama 3.3 70B), OpenAI API (GPT-4o-mini)
Telemetry & Tracing: Langfuse, Custom Structured Logging
Resilience & Async Execution: Tenacity, AsyncIO, HTTPX
Frontend UI: HTML5, CSS3 (Glassmorphism), Axios, JetBrains Mono

PRODUCTION_RAG2/
├── agents/                  # AI agent orchestration & custom tools
│   ├── orchestrator.py      # Multi-step agent execution engine
│   └── tools.py             # Registered tool definitions
├── config/                  # Global environment & system configuration
├── data/                    # Raw document storage (.pdf files)
├── database/                # Vector database connectivity
│   ├── connection.py        # Qdrant client connection pool & health checks
│   └── indexer.py           # Payload structure & vector indexing engine
├── ingestion/               # Document ingestion pipeline
│   ├── chunker.py           # Parent-child chunking & sliding window strategy
│   └── parsing.py           # Deep PDF parsing & table layout extraction
├── llm/                     # Core generative AI models
│   ├── embeddings.py        # Dense embedding engine wrapper
│   ├── generator.py         # Resilient multi-provider LLM engine (Groq/OpenAI)
│   ├── query_expander.py    # Multi-query expansion generator
│   └── schemas.py           # Pydantic input/output validation models
├── mcp/                     # Model Context Protocol support
│   └── server.py            # MCP server interface for external agents
├── monitoring/              # System telemetry
│   └── tracer.py            # Langfuse tracer & performance monitoring
├── retrieval/               # Core retrieval engine
│   ├── context_builder.py   # Parent-child context assembler
│   ├── query_engine.py      # End-to-end RAG pipeline orchestrator
│   ├── reranker.py          # Neural cross-encoder document reranker
│   └── search.py            # Parallel hybrid vector search executor
├── frontend/ / index.html   # Cyberpunk Glassmorphism UI
├── batch_ingest.py          # Automated batch document ingestion script
├── ingest_pdf.py            # Individual document processing utility
├── main.py                  # FastAPI server entry point & REST endpoints
├── requirements.txt         # Python project dependencies
└── README.md                # System documentation


⚙️ Setup & Installation
1. Clone Repository & Setup Virtual Environment

git clone https://github.com/harishjaipale/Enterprise-RAG-Gateway
cd Enterprise-RAG-Gateway

# Create Virtual Environment (Python 3.11 recommended)
python -m venv venv

# Activate Environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Environment Variables (.env)
Create a .env file in the root directory and configure your credentials:

# Vector DB Setup
QDRANT_URL=[https://your-qdrant-cluster-url.qdrant.io:6333](https://your-qdrant-cluster-url.qdrant.io:6333)
QDRANT_API_KEY=your_qdrant_api_key

# Primary LLM Provider (Groq)
GROQ_API_KEY=gsk_your_groq_api_key_here

# Fallback LLM Provider (OpenAI)
OPENAI_API_KEY=sk-your_openai_api_key_here

# Telemetry (Optional)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key

🚀 Execution Guide
### 1: Ingest Documents into Vector DB
Place your target PDF files in the data/ directory and execute the ingestion script:
python batch_ingest.py

### 2. Run the application

uvicorn main:app --reload --port 8000

### 3. Access the Web UI

Open your browser and navigate to: `http://localhost:8000`

---

## 📦 Project Structure

```text
├── main.py             # FastAPI entry point, routes & middleware
├── index.html           # Interactive Cyberpunk Frontend Web UI
├── requirements.txt    # Project dependencies
└── README.md            # Documentation
```

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.