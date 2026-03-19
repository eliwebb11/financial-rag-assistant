import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

#Configuration
RAW_DATA_DIR = Path("data/raw")
CHROMA_DB_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_documents"

# Free local embedding model
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking settings
# chunk_size: how many tokens per chunk
# chunk_overlap: how many tokens bleed between adjacent chunks (preserves context at boundaries)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

#Embedding model
def get_embed_model():
    #Loads the local HuggingFace embedding model.
    #First run downloads ~130MB and caches it. Subsequent runs are instant.
    print("Loading embedding model...")
    return HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

#ChromaDB
def get_chroma_collection():
    #Connects to (or creates) the local ChromaDB collection.
    #ChromaDB persists to disk at chroma_db/ so the index survives restarts.
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return client, collection

#Core ingestion
def ingest_ticker(ticker: str, embed_model=None) -> int:
    #Ingests all documents in data/raw/<ticker>/ into ChromaDB.
    #Returns the number of chunks stored.
    ticker = ticker.upper()
    ticker_dir = RAW_DATA_DIR / ticker

    if not ticker_dir.exists():
        raise FileNotFoundError(
            f"No data found for {ticker}. "
            f"Run sec_fetcher.py first to download filings."
        )

    print(f"\nIngesting documents for {ticker}...")
    print(f"  Source: {ticker_dir}")

    # Load all documents from the ticker's folder.
    # SimpleDirectoryReader handles .htm, .html, .txt, .pdf automatically.
    # required_exts filters to just the file types we want.
    documents = SimpleDirectoryReader(
        input_dir=str(ticker_dir),
        required_exts=[".htm", ".html", ".txt", ".pdf"],
        recursive=True,
    ).load_data()

    print(f"  Loaded {len(documents)} document(s)")

    # Attach ticker metadata to every document so we can filter by company later.
    # This is what powers the compare_companies tool in the MCP layer.
    for doc in documents:
        doc.metadata["ticker"] = ticker
        # Extract date from filename if present (e.g. AAPL_10-K_2024-11-01.htm)
        filename = Path(doc.metadata.get("file_name", "")).stem
        parts = filename.split("_")
        if len(parts) >= 3:
            doc.metadata["filing_date"] = parts[-1]
            doc.metadata["form_type"] = parts[-2]

    # Split documents into chunks.
    # SentenceSplitter is smarter than a naive character splitter —
    # it tries to break at sentence boundaries to keep context coherent.
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"  Split into {len(nodes)} chunks")

    # Load embedding model if not passed in
    if embed_model is None:
        embed_model = get_embed_model()

    # Connect to ChromaDB
    chroma_client, chroma_collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index this embeds every chunk and stores vectors + text in ChromaDB
    print(f"  Embedding and storing chunks (this may take a minute)...")
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print(f"  Done. {len(nodes)} chunks stored for {ticker}.")
    return len(nodes)

#Batch ingestion
def ingest_all_tickers() -> dict[str, int]:
    #Ingests every ticker folder found in data/raw/.
    #Reuses the same embedding model across all tickers for efficiency.
    embed_model = get_embed_model()
    results = {}

    ticker_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]

    if not ticker_dirs:
        print("No ticker folders found in data/raw/. Run sec_fetcher.py first.")
        return results

    print(f"Found {len(ticker_dirs)} ticker(s): {[d.name for d in ticker_dirs]}")

    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name
        try:
            count = ingest_ticker(ticker, embed_model=embed_model)
            results[ticker] = count
        except Exception as e:
            print(f"  Error ingesting {ticker}: {e}")
            results[ticker] = 0

    print(f"\nIngestion complete: {results}")
    return results

#Entry point
if __name__ == "__main__":
    ingest_all_tickers()