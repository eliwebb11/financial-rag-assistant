import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rag.index_manager import load_index

load_dotenv()

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# How many chunks to retrieve per query.
# Higher = more context for Claude but costs more tokens.
TOP_K = 8

# Minimum similarity score to include a chunk (0.0 to 1.0).
# Chunks below this threshold get filtered out before sending to Claude.
# This is a key hallucination prevention mechanism.
SIMILARITY_CUTOFF = 0.4


def build_query_engine(api_key: str, index: VectorStoreIndex = None):
    #Builds and returns a query engine backed by Claude.
    #api_key: Anthropic API key (passed in from UI or .env)
    #index: optional pre-loaded index (pass one in to avoid reloading)

    # Configure LlamaIndex to use Claude as the LLM
    llm = Anthropic(
        model="claude-haiku-4-5-20251001",
        api_key=api_key,
        max_tokens=1024,
    )

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    # Set global LlamaIndex settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Load index if not provided
    if index is None:
        index = load_index(embed_model=embed_model)

    # Retriever: finds the top K most semantically similar chunks
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )

    # Postprocessor: filters out low-confidence chunks before they reach Claude
    similarity_filter = SimilarityPostprocessor(
        similarity_cutoff=SIMILARITY_CUTOFF
    )

    # Build the full query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[similarity_filter],
    )

    return query_engine


def query(question: str, api_key: str, ticker_filter: str = None) -> dict:
    #Main entry point for querying the RAG system.
    
    #question: natural language question from the user
    #api_key: Anthropic API key
    #ticker_filter: optional ticker to restrict search (e.g. 'AAPL')

    """
    Returns a dict with:
        answer: Claude's response grounded in documents
        sources: list of source citations
        has_answer: whether relevant documents were found
    """
    engine = build_query_engine(api_key=api_key)

    # Wrap the question with instructions that enforce citation behavior
    # and honest "I don't know" responses
    prompt = f"""You are a financial research assistant analyzing SEC filings.

Answer the following question using ONLY the information provided in the context.

Rules:
- Always cite which document and section your answer comes from
- If the context does not contain enough information to answer, say exactly: "The documents do not contain enough information to answer this question."
- Never make up numbers, dates, or financial figures
- Be specific and quote exact figures when available
- If asked about a specific company, focus only on that company's data

Question: {question}"""

    if ticker_filter:
        prompt += f"\n\nFocus only on documents for ticker: {ticker_filter.upper()}"

    response = engine.query(prompt)

    # Extract source citations from the retrieved nodes
    sources = []
    if hasattr(response, "source_nodes") and response.source_nodes:
        for node in response.source_nodes:
            metadata = node.node.metadata
            source = {
                "ticker":       metadata.get("ticker", "Unknown"),
                "form_type":    metadata.get("form_type", "Unknown"),
                "filing_date":  metadata.get("filing_date", "Unknown"),
                "filename":     metadata.get("file_name", "Unknown"),
                "score":        round(node.score, 3) if node.score else None,
                "text_preview": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
            }
            # Avoid duplicate sources (same file cited multiple times)
            if source["filename"] not in [s["filename"] for s in sources]:
                sources.append(source)

    # Detect "I don't know" responses to flag them clearly in the UI
    answer_text = str(response)
    has_answer = "do not contain enough information" not in answer_text.lower()

    return {
        "answer":     answer_text,
        "sources":    sources,
        "has_answer": has_answer,
    }


def compare_companies(question: str, tickers: list[str], api_key: str) -> dict:
    #Queries the same question across multiple tickers and returns
    #a structured comparison. Powers the compare_companies MCP tool.
    results = {}
    for ticker in tickers:
        results[ticker] = query(question, api_key=api_key, ticker_filter=ticker)

    # Build a comparison summary using Claude
    llm = Anthropic(
        model="claude-haiku-4-5-20251001",
        api_key=api_key,
        max_tokens=1024,
    )

    individual_answers = "\n\n".join([
        f"{ticker}:\n{results[ticker]['answer']}"
        for ticker in tickers
    ])

    comparison_prompt = f"""Given these individual answers about different companies, 
write a concise comparative analysis highlighting key similarities and differences.

Question asked: {question}

Individual answers:
{individual_answers}

Provide a structured comparison with specific figures where available."""

    from anthropic import Anthropic as AnthropicClient
    client = AnthropicClient(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": comparison_prompt}]
    )

    comparison_text = message.content[0].text

    return {
        "comparison":   comparison_text,
        "individual":   results,
        "tickers":      tickers,
    }


# test
if __name__ == "__main__":
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY in your .env file")

    print("Testing RAG query engine...")
    result = query(
        question="What was Apple's total revenue in the most recent fiscal year?",
        api_key=api_key,
    )

    print("\n── Answer ──────────────────────────────────────────")
    print(result["answer"])
    print("\n── Sources ─────────────────────────────────────────")
    for s in result["sources"]:
        print(f"  [{s['ticker']}] {s['form_type']} filed {s['filing_date']} (score: {s['score']})")
    print(f"\nHas answer: {result['has_answer']}")