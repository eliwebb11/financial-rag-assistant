import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path so we can import from rag/
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from rag.query_engine import query, compare_companies
from rag.index_manager import get_available_tickers
from ingestion.sec_fetcher import fetch_filings_for_ticker
from ingestion.pipeline import ingest_ticker

load_dotenv()

#Server setup 
server = Server("financial-rag-assistant")


def get_api_key() -> str:
    #Gets the Anthropic API key from environment.
    #In the MCP context the key comes from .env on the server side.
    #In the Streamlit UI context it's passed in from the user.
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to your .env file or pass it as an environment variable."
        )
    return api_key

#Tool definitions
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    #Declares all tools this MCP server exposes.
    #This is what MCP clients call to discover what your server can do.
    return [
        types.Tool(
            name="search_filings",
            description=(
                "Search through SEC filings (10-K annual reports, earnings transcripts) "
                "using a natural language question. Returns an answer grounded in the "
                "actual documents with source citations. Use this to answer questions "
                "about a company's financials, strategy, risks, or operations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The natural language question to answer from SEC filings",
                    },
                    "ticker": {
                        "type": "string",
                        "description": (
                            "Optional stock ticker to restrict search to one company "
                            "(e.g. 'AAPL'). If omitted, searches all ingested companies."
                        ),
                    },
                },
                "required": ["question"],
            },
        ),

        types.Tool(
            name="summarize_document",
            description=(
                "Generate a structured summary of a specific company's most recent "
                "SEC filing. Returns key financial metrics, business highlights, "
                "risk factors, and management commentary."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL')",
                    },
                    "focus": {
                        "type": "string",
                        "description": (
                            "Optional specific area to focus the summary on. "
                            "Examples: 'revenue and growth', 'risk factors', "
                            "'competitive position', 'capital allocation'"
                        ),
                    },
                },
                "required": ["ticker"],
            },
        ),

        types.Tool(
            name="compare_companies",
            description=(
                "Compare multiple companies side by side on a specific question or metric. "
                "Queries each company's SEC filings independently and produces a structured "
                "comparative analysis. Use this for competitive analysis or sector research."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or metric to compare across companies",
                    },
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock tickers to compare (e.g. ['AAPL', 'MSFT'])",
                        "minItems": 2,
                        "maxItems": 5,
                    },
                },
                "required": ["question", "tickers"],
            },
        ),

        types.Tool(
            name="list_available_companies",
            description=(
                "Lists all companies whose SEC filings have been ingested and are "
                "available to query. Call this first to know what data is available "
                "before running search_filings or compare_companies."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),

        types.Tool(
            name="ingest_company",
            description=(
                "Downloads and indexes SEC filings for a new company ticker. "
                "Run this to add a company to the research database before querying it. "
                "Downloads the two most recent 10-K annual reports by default."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol to ingest (e.g. 'MSFT')",
                    },
                    "filing_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filing types to download. Defaults to ['10-K']",
                    },
                },
                "required": ["ticker"],
            },
        ),
    ]

#Tool implementations
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    #Routes tool calls to the appropriate handler and returns results.
    #All tools return TextContent — plain text that the MCP client renders.
    try:
        if name == "search_filings":
            return await handle_search_filings(arguments)
        elif name == "summarize_document":
            return await handle_summarize_document(arguments)
        elif name == "compare_companies":
            return await handle_compare_companies(arguments)
        elif name == "list_available_companies":
            return await handle_list_available_companies()
        elif name == "ingest_company":
            return await handle_ingest_company(arguments)
        else:
            return [types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Error running tool '{name}': {str(e)}"
        )]

#Individual tool handlers
async def handle_search_filings(arguments: dict) -> list[types.TextContent]:
    question = arguments["question"]
    ticker = arguments.get("ticker")
    api_key = get_api_key()

    # Run blocking RAG query in a thread so we don't block the async event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: query(question, api_key=api_key, ticker_filter=ticker)
    )

    # Format the response with citations
    output = []
    output.append(f"## Answer\n{result['answer']}")

    if result["sources"]:
        output.append("\n## Sources")
        for s in result["sources"]:
            output.append(
                f"- **{s['ticker']}** {s['form_type']} "
                f"(filed {s['filing_date']}, relevance: {s['score']})"
            )
            output.append(f"  > {s['text_preview']}")

    if not result["has_answer"]:
        output.append(
            "\nThe indexed documents did not contain enough information "
            "to answer this question confidently."
        )

    return [types.TextContent(type="text", text="\n".join(output))]

async def handle_summarize_document(arguments: dict) -> list[types.TextContent]:
    ticker = arguments["ticker"].upper()
    focus = arguments.get("focus", "overall business performance and financials")
    api_key = get_api_key()

    question = (
        f"Provide a comprehensive summary of {ticker}'s most recent 10-K filing, "
        f"focusing on: {focus}. "
        f"Include specific numbers, dates, and figures where available. "
        f"Structure your response with clear sections."
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: query(question, api_key=api_key, ticker_filter=ticker)
    )

    output = [f"# {ticker} Filing Summary\n"]
    output.append(result["answer"])

    if result["sources"]:
        output.append("\n---\n**Sources:**")
        for s in result["sources"]:
            output.append(
                f"- {s['form_type']} filed {s['filing_date']}"
            )

    return [types.TextContent(type="text", text="\n".join(output))]

async def handle_compare_companies(arguments: dict) -> list[types.TextContent]:
    question = arguments["question"]
    tickers = [t.upper() for t in arguments["tickers"]]
    api_key = get_api_key()

    # Check that all requested tickers are actually ingested
    available = get_available_tickers()
    missing = [t for t in tickers if t not in available]
    if missing:
        return [types.TextContent(
            type="text",
            text=(
                f"The following tickers have not been ingested yet: {missing}. "
                f"Use the ingest_company tool first, then retry. "
                f"Currently available: {available}"
            )
        )]

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: compare_companies(question, tickers=tickers, api_key=api_key)
    )

    output = [f"# Comparison: {', '.join(tickers)}\n"]
    output.append(f"**Question:** {question}\n")
    output.append("## Comparative Analysis")
    output.append(result["comparison"])
    output.append("\n---\n## Individual Answers")

    for ticker in tickers:
        individual = result["individual"][ticker]
        output.append(f"\n### {ticker}")
        output.append(individual["answer"])

    return [types.TextContent(type="text", text="\n".join(output))]

async def handle_list_available_companies() -> list[types.TextContent]:
    tickers = get_available_tickers()

    if not tickers:
        return [types.TextContent(
            type="text",
            text=(
                "No companies have been ingested yet. "
                "Use the ingest_company tool to add companies to the database."
            )
        )]

    output = ["## Available Companies\n"]
    output.append(f"The following {len(tickers)} company/companies are ready to query:\n")
    for ticker in tickers:
        output.append(f"- **{ticker}**")
    output.append(
        "\nUse search_filings or compare_companies to query these companies."
    )

    return [types.TextContent(type="text", text="\n".join(output))]

async def handle_ingest_company(arguments: dict) -> list[types.TextContent]:
    ticker = arguments["ticker"].upper()
    filing_types = arguments.get("filing_types", ["10-K"])

    output = [f"## Ingesting {ticker}\n"]

    try:
        # Step 1: Download filings
        output.append("**Step 1:** Downloading filings from SEC EDGAR...")
        paths = fetch_filings_for_ticker(ticker, filing_types=filing_types)

        if not paths:
            return [types.TextContent(
                type="text",
                text=f"No filings found for {ticker}. Check that the ticker is valid."
            )]

        output.append(f"Downloaded {len(paths)} filing(s).")

        # Step 2: Ingest into ChromaDB
        output.append("**Step 2:** Embedding and indexing into ChromaDB...")
        chunk_count = ingest_ticker(ticker)
        output.append(f"Indexed {chunk_count} chunks.")
        output.append(
            f"\n✅ {ticker} is now available for querying. "
            f"Use search_filings or summarize_document to query it."
        )

    except ValueError as e:
        output.append(f"Error: {str(e)}")
    except Exception as e:
        output.append(f"Unexpected error: {str(e)}")

    return [types.TextContent(type="text", text="\n".join(output))]

#Entry point
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())