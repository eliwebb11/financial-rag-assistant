import os
import streamlit as st
from dotenv import load_dotenv

from rag.query_engine import query, compare_companies
from rag.index_manager import get_available_tickers
from ingestion.sec_fetcher import fetch_filings_for_ticker
from ingestion.pipeline import ingest_ticker

load_dotenv()

#Page config
st.set_page_config(
    page_title="Financial Research Assistant",
    page_icon="🔶",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Custom CSS
st.markdown("""
<style>
    .source-card {
        background-color: #111111;
        border-left: 3px solid #00c853;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.85em;
    }
    .no-answer {
        background-color: #1a0a00;
        border-left: 3px solid #00c853;
        padding: 10px 15px;
        border-radius: 0 4px 4px 0;
    }
    .answer-box {
        background-color: #0d0d0d;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    [data-testid="stSidebar"] {
        min-width: 220px;
        max-width: 220px;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

#Session state initialization
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("ANTHROPIC_API_KEY", "")
if "query_history" not in st.session_state:
    st.session_state.query_history = []

#Helper functions
def get_api_key() -> str | None:
    #Returns the API key from session state or None if not set.
    key = st.session_state.api_key.strip()
    return key if key else None

def render_sources(sources: list[dict]):
    #Renders source citation cards below an answer.
    if not sources:
        return
    st.markdown("**Sources**")
    for s in sources:
        st.markdown(
            f"""<div class="source-card">
                <strong>[{s['ticker']}]</strong> {s['form_type']} 
                &nbsp;·&nbsp; Filed {s['filing_date']} 
                &nbsp;·&nbsp; Relevance: {s['score']}<br>
                <em>{s['text_preview']}</em>
            </div>""",
            unsafe_allow_html=True,
        )

def render_no_answer():
    #Renders a warning when documents don't contain the answer.
    st.markdown(
        """<div class="no-answer">
            <strong>Insufficient information</strong> — 
            the indexed documents do not contain enough information 
            to answer this question. Try rephrasing or adding 
            additional filings.
        </div>""",
        unsafe_allow_html=True,
    )

#Legal Disclaimer
@st.dialog("Legal Disclaimer")
def show_disclaimer():
    st.caption(
        "This tool is for informational and research purposes only and does not "
        "constitute financial, investment, or legal advice. Answers are generated "
        "by an AI model and may be inaccurate, incomplete, or outdated. All "
        "information is sourced from publicly available SEC filings — always verify "
        "against the original source documents before making any decisions. "
        "Do not enter any private, sensitive, or non-public information. "
        "By using this tool, you acknowledge that outputs may contain errors and "
        "that no reliance should be placed on them without independent verification."
    )

#Sidebar
with st.sidebar:
    st.title("Financial Research Assistant")
    st.caption("SEC EDGAR + Claude")

    st.divider()

    # API key input
    st.subheader("API Key")
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-ant-...",
        help="Your key is never stored or transmitted anywhere except directly to Anthropic's API.",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    if not get_api_key():
        st.warning("Enter your API key to enable querying.")
    else:
        st.success("API key set.")

    st.divider()

    # Available companies
    st.subheader("Available Companies")
    try:
        available_tickers = get_available_tickers()
        if available_tickers:
            for ticker in available_tickers:
                st.markdown(f"- **{ticker}**")
        else:
            st.warning("No companies added yet. Go to the **Add Company** tab to get started.")
    except Exception:
        available_tickers = []
        st.warning("No companies added yet. Go to the **Add Company** tab to get started.")

    st.divider()

    #Query history
    if st.session_state.query_history:
        st.subheader("Recent Queries")
        for item in reversed(st.session_state.query_history[-5:]):
            st.caption(f"• {item[:60]}{'...' if len(item) > 60 else ''}")

    if st.button("Legal Disclaimer", key="disclaimer_btn", type="tertiary"):
        show_disclaimer()

#Main content tabs
tab_search, tab_summarize, tab_compare, tab_ingest = st.tabs([
    "Search Filings",
    "Summarize",
    "Compare Companies",
    "Add New Company",
])


#Tab 1: Search Filings
with tab_search:
    st.header("Search SEC Filings")
    st.caption(
        "Ask any question about a company's financials, strategy, risks, or operations. Answers are grounded in actual SEC filings with source citations."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_input(
            "Your question",
            placeholder="How much did revenue grow year over year?",
            key="search_question",
        )

    with col2:
        ticker_options = ["All companies"] + available_tickers
        selected_ticker = st.selectbox(
            "Filter by company",
            options=ticker_options,
            key="search_ticker",
        )

    ticker_filter = None if selected_ticker == "All companies" else selected_ticker

    # Example questions
    if st.button("Search", type="primary", key="search_btn", disabled=not get_api_key()):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.query_history.append(question)
            with st.spinner("Searching filings and generating answer..."):
                try:
                    result = query(
                        question=question,
                        api_key=get_api_key(),
                        ticker_filter=ticker_filter,
                    )

                    if result["has_answer"]:
                        st.markdown("### Answer")
                        st.markdown(
                            f'<div class="answer-box">{result["answer"]}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        render_no_answer()
                        st.markdown(result["answer"])

                    render_sources(result["sources"])

                except Exception as e:
                    st.error(f"Query failed: {str(e)}")

#Tab 2: Summarize
with tab_summarize:
    st.header("Summarize Filing")
    st.caption(
        "Generate a structured summary of a company's most recent 10-K filing. "
        "Focus the summary on a specific area or get a broad overview."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        summarize_ticker = st.selectbox(
            "Company",
            options=available_tickers if available_tickers else ["No companies added"],
            key="summarize_ticker",
        )

    with col2:
        focus_options = {
            "Overall business performance": "overall business performance and key financial metrics",
            "Revenue and growth": "revenue trends, growth drivers, and segment performance",
            "Risk factors": "key risk factors and how management plans to address them",
            "Competitive position": "competitive landscape, market share, and strategic advantages",
            "Capital allocation": "capital allocation, dividends, buybacks, and investment strategy",
            "Custom...": "custom",
        }
        selected_focus = st.selectbox(
            "Focus area",
            options=list(focus_options.keys()),
            key="summarize_focus",
        )

    custom_focus = ""
    if selected_focus == "Custom...":
        custom_focus = st.text_input(
            "Describe your focus",
            placeholder="e.g. AI and machine learning investments",
            key="custom_focus",
        )

    focus_text = (
        custom_focus if selected_focus == "Custom..." and custom_focus
        else focus_options.get(selected_focus, selected_focus)
    )

    if st.button(
        "Generate Summary",
        type="primary",
        key="summarize_btn",
        disabled=not get_api_key() or not available_tickers,
    ):
        with st.spinner(f"Summarizing {summarize_ticker} filing..."):
            try:
                summary_question = (
                    f"Provide a comprehensive summary of {summarize_ticker}'s most recent "
                    f"10-K filing, focusing on: {focus_text}. "
                    f"Include specific numbers, dates, and figures. "
                    f"Use clear sections with headers."
                )
                result = query(
                    question=summary_question,
                    api_key=get_api_key(),
                    ticker_filter=summarize_ticker,
                )

                st.markdown(f"### {summarize_ticker} — {selected_focus}")
                st.markdown(result["answer"])
                render_sources(result["sources"])

            except Exception as e:
                st.error(f"Summary failed: {str(e)}")

#Tab 3: Compare Companies
with tab_compare:
    st.header("Compare Companies")
    st.caption(
        "Ask the same question across multiple companies and get a side-by-side "
        "comparative analysis grounded in their SEC filings."
    )

    compare_question = st.text_input(
        "Comparison question",
        placeholder="How does revenue growth compare across these companies?",
        key="compare_question",
    )

    st.markdown("**Select companies to compare** (2–5)")
    if available_tickers:
        selected_for_compare = st.multiselect(
            "Companies",
            options=available_tickers,
            default=available_tickers[:2] if len(available_tickers) >= 2 else available_tickers,
            key="compare_tickers",
        )
    else:
        st.info("No companies added yet. Use the Add Companies tab to add companies.")
        selected_for_compare = []

    compare_ready = (
        get_api_key()
        and len(selected_for_compare) >= 2
        and compare_question.strip()
    )

    if st.button(
        "Compare",
        type="primary",
        key="compare_btn",
        disabled=not compare_ready,
    ):
        with st.spinner(
            f"Querying {', '.join(selected_for_compare)} — this may take a moment..."
        ):
            try:
                result = compare_companies(
                    question=compare_question,
                    tickers=selected_for_compare,
                    api_key=get_api_key(),
                )

                st.markdown("### Comparative Analysis")
                st.markdown(result["comparison"])

                st.divider()
                st.markdown("### Individual Answers")

                cols = st.columns(len(selected_for_compare))
                for i, ticker in enumerate(selected_for_compare):
                    with cols[i]:
                        st.markdown(f"**{ticker}**")
                        individual = result["individual"][ticker]
                        st.markdown(individual["answer"])
                        if individual["sources"]:
                            with st.expander("Sources"):
                                render_sources(individual["sources"])

            except Exception as e:
                st.error(f"Comparison failed: {str(e)}")


#Tab 4: Ingest New Company
with tab_ingest:
    st.header("Add a New Company")
    st.caption(
        "Download and index SEC filings for a new company. "
        "Once added, the company will be available in all other tabs."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        new_ticker = st.text_input(
            "Stock ticker symbol",
            placeholder="MSFT",
            key="new_ticker",
        ).upper().strip()

    with col2:
        filing_type_options = st.multiselect(
            "Filing types to download",
            options=["10-K", "8-K"],
            default=["10-K"],
            key="filing_types",
        )

    st.caption(
        "This will download the 2 most recent filings of each selected type from SEC EDGAR and index them into ChromaDB. This may take a few minutes."
    )

    if st.button(
        "Add Company",
        type="primary",
        key="ingest_btn",
        disabled=not new_ticker or not filing_type_options,
    ):
        progress = st.empty()
        status = st.empty()

        with st.spinner(f"Fetching and indexing {new_ticker}..."):
            try:
                progress.markdown("**Step 1 of 2:** Downloading filings from SEC EDGAR...")
                paths = fetch_filings_for_ticker(
                    new_ticker,
                    filing_types=filing_type_options,
                )

                if not paths:
                    status.error(
                        f"No filings found for {new_ticker}. "
                        "Check that the ticker symbol is correct."
                    )
                else:
                    progress.markdown(
                        f"**Step 2 of 2:** Embedding and indexing "
                        f"{len(paths)} filing(s) into ChromaDB..."
                    )
                    chunk_count = ingest_ticker(new_ticker)
                    progress.empty()
                    status.success(
                        f"✅ {new_ticker} added successfully! "
                        f"{chunk_count} chunks indexed. "
                        f"Refresh the page to see it in the company list."
                    )

            except ValueError as e:
                progress.empty()
                status.error(f"{str(e)}")
            except Exception as e:
                progress.empty()
                status.error(f"Unexpected error: {str(e)}")