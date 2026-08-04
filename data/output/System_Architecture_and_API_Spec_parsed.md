# SYSTEM ARCHITECTURE SPECIFICATION: RAG ENGINE

| Component | Tech Stack         | Throughput   | P99 Latency |
| --------- | ------------------ | ------------ | ----------- |
| Ingestion | PyPDF / LlamaIndex | 45 docs/min  | 1200 ms     |
| Embedding | text-embedding-3   | 850 chunks/s | 180 ms      |
| Vector DB | Qdrant / Milvus    | 2,400 q/sec  | 35 ms       |
| LLM Gen   | vLLM / Llama-3     | 42 tokens/s  | 450 ms      |

Hybrid Search strategy combining Dense Vector Similarity with Sparse Keyword Matching (BM25).

Base Image: nvidia/cuda:12.2.0-runtime-ubuntu22.04

Shared Memory: 16GB