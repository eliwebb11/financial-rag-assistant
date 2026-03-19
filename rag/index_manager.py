from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

CHROMA_DB_DIR = Path("chroma_db")
COLLECTION_NAME = "financial_documents"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_index(embed_model=None) -> VectorStoreIndex:
    """
    Loads the existing ChromaDB-backed index.
    This does NOT re-embed anything — it just connects to the already
    populated vector store so we can query it.
    """
    if embed_model is None:
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index


def get_available_tickers() -> list[str]:
    """
    Returns a list of tickers that have been ingested into ChromaDB.
    Useful for the UI to show users what companies are available.
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)

    # Peek at stored metadata to find unique tickers
    results = collection.get(limit=collection.count(), include=["metadatas"])
    tickers = set()
    for metadata in results["metadatas"]:
        if "ticker" in metadata:
            tickers.add(metadata["ticker"])

    return sorted(list(tickers))