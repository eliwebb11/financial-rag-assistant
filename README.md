# Financial Document Research Assistant

A RAG-powered financial research tool that lets you ask natural language questions
about SEC filings and get answers grounded in actual documents with source citations.

**[Live Demo →](https://financial-research-assistant-bot.streamlit.app/)**

---

## What it does

- **Search SEC Filings** — Ask any question about a company's financials, strategy,
  risks, or operations. Answers are grounded in actual 10-K filings with citations
  showing exactly which document and filing date the answer came from.

- **Summarize Documents** — Generate structured summaries of a company's most recent
  annual report, focused on a specific area like revenue growth, risk factors, or
  capital allocation strategy.

- **Compare Companies** — Query the same question across multiple companies
  simultaneously and get a side-by-side comparative analysis.

- **Ingest New Companies** — Add any publicly traded company on demand by entering
  its ticker symbol. The app downloads filings directly from SEC EDGAR and indexes
  them in real time.

---

## Architecture
```
Streamlit UI (app.py)
      ↓
MCP Server (mcp_server/server.py)
      ↓
LlamaIndex RAG Engine (rag/query_engine.py)
      ↓  ↑
ChromaDB Vector Store ← Ingestion Pipeline (ingestion/pipeline.py)
                                ↓
                        SEC EDGAR API
```

**Data flow:**
1. SEC filings are downloaded from EDGAR as `.htm` files
2. Documents are chunked into 512-token pieces with 64-token overlap using `SentenceSplitter`
3. Each chunk is embedded using `BAAI/bge-small-en-v1.5` (free, runs locally)
4. Embeddings and text are stored in a persistent ChromaDB collection
5. At query time, the top-8 most semantically similar chunks are retrieved
6. Chunks scoring below 0.4 cosine similarity are filtered out to prevent hallucination
7. Retrieved chunks are sent to Claude with a strict citation prompt
8. The answer and source citations are returned to the UI

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Anthropic Claude Haiku | Fast, cost-efficient for RAG synthesis |
| RAG Framework | LlamaIndex | Purpose-built for document Q&A, native citation support |
| Vector Store | ChromaDB | Free, local, no infrastructure required |
| Embeddings | BAAI/bge-small-en-v1.5 | Free local model, strong semantic similarity |
| MCP Server | Anthropic MCP SDK | Exposes tools to any MCP-compatible client |
| UI | Streamlit | Rapid Python-native UI, free public deployment |
| Data Source | SEC EDGAR API | Free, official source for all public filings |

---

## MCP Server Tools

The project exposes an MCP server that any MCP-compatible client can connect to:

| Tool | Description |
|---|---|
| `search_filings` | Natural language search across ingested SEC filings |
| `summarize_document` | Structured summary of a company's most recent 10-K |
| `compare_companies` | Side-by-side comparison across multiple tickers |
| `list_available_companies` | Lists all ingested companies |
| `ingest_company` | Downloads and indexes filings for a new ticker |

---

## Running Locally

**Prerequisites:** Python 3.11, Git
```bash
# Clone the repo
git clone https://github.com/eliwebb11/financial-rag-assistant.git
cd financial-rag-assistant

# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Add your Anthropic API key
cp .env.example .env
# Edit .env and add your key: ANTHROPIC_API_KEY=sk-ant-...

# Ingest some companies
python -m ingestion.sec_fetcher  # downloads AAPL by default
python -m ingestion.pipeline     # embeds and indexes into ChromaDB

# Run the app
streamlit run app.py
```

---

## Key Design Decisions

**Why LlamaIndex over LangChain?**
RAG over documents is LlamaIndex's core design purpose. The abstractions map
directly to the problem — `SentenceSplitter`, `VectorStoreIndex`, and metadata
filtering all work exactly as needed without the plumbing overhead LangChain
would require for a document-first use case.

**Why local embeddings?**
Using `BAAI/bge-small-en-v1.5` instead of a paid embedding API means ingestion
is completely free regardless of document volume. Claude is only called at query
time, keeping API costs proportional to actual usage.

**Why a similarity cutoff?**
Financial data requires strict hallucination prevention. Chunks scoring below 0.4
cosine similarity are filtered before reaching Claude — if no relevant chunks exist,
the system returns "insufficient information" rather than generating a plausible but
incorrect answer. This is critical for a finance context where wrong numbers have
real consequences.

**Why separate ingestion and query layers?**
Ingestion is a one-time or periodic job. Keeping it separate from the query engine
means the Streamlit app never accidentally re-indexes documents, and the pipeline
can be run independently via CLI for batch processing.

---

## Project Structure
```
financial-rag-assistant/
├── app.py                    # Streamlit UI
├── ingestion/
│   ├── sec_fetcher.py        # Downloads filings from SEC EDGAR
│   └── pipeline.py           # Chunks, embeds, stores into ChromaDB
├── rag/
│   ├── query_engine.py       # LlamaIndex RAG engine + citation logic
│   └── index_manager.py      # Loads and manages ChromaDB index
└── mcp_server/
    └── server.py             # MCP server exposing research tools
```

---

## Author

**Eli Webb** — BS Computer Science, Oklahoma State University

[GitHub](https://github.com/eliwebb11) · [LinkedIn](https://www.linkedin.com/in/eli-webb1/)
