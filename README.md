# Multi-Agent Academic Research Assistant

> A multi-agent AI system that analyzes scientific papers and produces structured literature reviews.

## 🏗️ Architecture

```
                         ┌─────────────┐
                         │  User Query │
                         └──────┬──────┘
                                │
                    ┌───────────▼───────────┐
                    │   🔍 Agent 1:         │
                    │   Collector (ReAct)    │  ← RAG retrieval from ChromaDB
                    │   Tools: RAGQuery,    │
                    │   KeywordExtractor    │
                    └───────────┬───────────┘
                                │ excerpts
                    ┌───────────▼───────────┐
                    │   📊 Agent 2:         │
                    │   Analyst (CoT)       │  ← Trend analysis & gap detection
                    │   Tools: TrendAnalyzer│
                    │   ComparisonTool      │
                    └───────────┬───────────┘
                                │ analysis
                    ┌───────────▼───────────┐
                    │   ✍️ Agent 3:          │
                    │   Writer              │  ← Generates Markdown review
                    │   Tools: MdFormatter, │
                    │   CitationBuilder     │
                    └───────────┬───────────┘
                                │ draft
                    ┌───────────▼───────────┐
                    │   ✅ Agent 4:         │
                    │   Validator           │  ← Quality check & feedback
                    │   Tools: Consistency, │
                    │   GrammarChecker      │
                    └───────┬───────┬───────┘
                            │       │
                     approved    rejected
                            │       │
                    ┌───────▼─┐  ┌──▼──────────┐
                    │  END    │  │ Writer retry │
                    │  (save) │  │ (max 2x)    │
                    └─────────┘  └─────────────┘
```

### RAG Pipeline (LlamaIndex + ChromaDB)

| Component | Technology | Details |
|-----------|-----------|---------|
| Ingestion | `SimpleDirectoryReader` + `PyMuPDFReader` | Loads PDFs from `data/papers/` |
| Chunking | `SentenceSplitter` | chunk_size=512, overlap=64 |
| Embeddings | `BAAI/bge-small-en-v1.5` | Free, local, 384-dim |
| Vector Store | ChromaDB | Persisted to disk |
| Retrieval | Semantic search | top_k=5 |

### Orchestration (LangGraph)

The pipeline uses LangGraph's `StateGraph` with a shared `PipelineState` dictionary:
- Sequential flow: `Collector → Analyst → Writer → Validator`
- Conditional retry: Validator can send the draft back to Writer (max 2 retries)
- State keys: `query`, `excerpts`, `analysis`, `draft`, `final`, `feedback`, `scores`

---

## 🚀 Setup

### Prerequisites
- **Python 3.10+**
- **Ollama** (for local LLM) — [install here](https://ollama.ai)

### 1. Clone & Install

```bash
cd aimultiagentproject

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Ollama

```bash
# Install Ollama (if not already)
# Download from: https://ollama.ai

# Pull the model (Llama 3.1 — best tool-calling support)
ollama pull llama3.1

# Verify it's running
ollama list
```

### 3. Configure Environment

```bash
# Copy the template
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac

# Edit .env if needed (defaults work out of the box)
```

### 4. Download Research Papers

```bash
python data/download_papers.py
```

This downloads 15 curated arXiv papers on **few-shot learning in NLP**.

---

## 🎮 Running the Demo

```bash
python demo_run.py
```

### Expected Output

1. **Collector** retrieves ~5 relevant passages from the corpus
2. **Analyst** identifies 3+ trends and 2+ research gaps
3. **Writer** produces a ~500-word structured literature review
4. **Validator** scores and approves (or requests revision)
5. Final review is saved to `output/literature_review.md`

### Demo Query
> "What are the main approaches to few-shot learning in NLP, and what are the open challenges?"

---

## 📁 Project Structure

```
aimultiagentproject/
├── config/
│   └── settings.py           # Centralized configuration
├── rag/
│   ├── ingestion.py          # PDF loading
│   ├── indexer.py            # Chunking + embedding + ChromaDB
│   └── retriever.py          # Semantic retrieval
├── tools/
│   ├── rag_tools.py          # RAGQuery, KeywordExtractor
│   ├── analysis_tools.py     # TrendAnalyzer, Comparison
│   ├── writing_tools.py      # MarkdownFormatter, CitationBuilder
│   └── validation_tools.py   # ConsistencyChecker, Grammar
├── agents/
│   ├── collector.py          # Agent 1 — RAG retrieval
│   ├── analyst.py            # Agent 2 — Trend analysis
│   ├── writer.py             # Agent 3 — Literature review
│   └── validator.py          # Agent 4 — Quality validation
├── orchestrator/
│   ├── state.py              # PipelineState TypedDict
│   └── main.py               # LangGraph pipeline
├── data/
│   ├── papers/               # Scientific PDFs (arXiv)
│   └── download_papers.py    # Paper downloader
├── output/                   # Generated reviews
├── demo_run.py               # End-to-end demo
├── requirements.txt          # Dependencies
├── .env.example              # Config template
└── README.md                 # This file
```

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | Ollama (llama3.1) | Local, free inference |
| Embeddings | HuggingFace BGE-small | Local, free embeddings |
| RAG | LlamaIndex | Ingestion, indexing, retrieval |
| Vector DB | ChromaDB | Persistent vector storage |
| Agents | LangChain | Tool definitions, prompts |
| Orchestration | LangGraph | Multi-agent pipeline |
| Output | Rich | Console formatting |

---

## 📝 License

Academic project — Distributed AI & Multi-Agent Systems course.
