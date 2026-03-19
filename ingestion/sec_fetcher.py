import os
import time
import requests
from pathlib import Path

# SEC requires a User-Agent header identifying who is making requests.
# This is their official policy — without it, requests get blocked.
HEADERS = {
    "User-Agent": "EliWebb/financial-rag-assistant eliwebb456@gmail.com"
}

# Where downloaded filings will be saved
RAW_DATA_DIR = Path("data/raw")


def get_cik_for_ticker(ticker: str) -> str:
    # Converts a stock ticker (e.g. 'AAPL') to a CIK number.
    # CIK (Central Index Key) is how SEC EDGAR identifies companies internally.
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()

    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            # CIK must be zero-padded to 10 digits for EDGAR API calls
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR.")


def get_recent_filings(cik: str, filing_type: str = "10-K", count: int = 3) -> list[dict]:
    #Returns metadata for the most recent filings of a given type for a company.
    #filing_type can be '10-K' (annual report) or '8-K' (earnings/events).
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    filings = data["filings"]["recent"]

    # Zip the parallel arrays from EDGAR into a list of dicts
    all_filings = [
        {
            "accessionNumber": filings["accessionNumber"][i],
            "filingDate":      filings["filingDate"][i],
            "form":            filings["form"][i],
            "primaryDocument": filings["primaryDocument"][i],
        }
        for i in range(len(filings["form"]))
    ]

    # Filter to the filing type we want and take the most recent N
    matching = [f for f in all_filings if f["form"] == filing_type]
    return matching[:count]


def download_filing(cik: str, filing: dict, ticker: str) -> Path:
    #Downloads a single filing's primary document and saves it to data/raw/.
    #Returns the path to the saved file.
    accession = filing["accessionNumber"].replace("-", "")
    primary_doc = filing["primaryDocument"]
    filing_date = filing["filingDate"]
    form_type = filing["form"].replace("/", "-")  # e.g. 10-K/A -> 10-K-A

    # Build the EDGAR URL for this specific document
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    # Create a folder per ticker to keep things organized
    ticker_dir = RAW_DATA_DIR / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # Filename includes ticker, form type, and date for easy identification
    filename = f"{ticker.upper()}_{form_type}_{filing_date}{Path(primary_doc).suffix}"
    filepath = ticker_dir / filename

    with open(filepath, "wb") as f:
        f.write(response.content)

    print(f"  Downloaded: {filename}")
    return filepath


def fetch_filings_for_ticker(ticker: str, filing_types: list[str] = None) -> list[Path]:
    #Main entry point. Downloads recent filings for a ticker and returns
    #a list of file paths for the ingestion pipeline to process.
    if filing_types is None:
        filing_types = ["10-K"]

    print(f"\nFetching filings for {ticker.upper()}...")

    cik = get_cik_for_ticker(ticker)
    print(f"  CIK: {cik}")

    downloaded_paths = []

    for filing_type in filing_types:
        print(f"  Looking for {filing_type} filings...")
        filings = get_recent_filings(cik, filing_type=filing_type, count=2)

        if not filings:
            print(f"  No {filing_type} filings found.")
            continue

        for filing in filings:
            try:
                path = download_filing(cik, filing, ticker)
                downloaded_paths.append(path)
                # SEC rate limit: be polite, wait between requests
                time.sleep(0.5)
            except Exception as e:
                print(f"  Warning: could not download {filing['accessionNumber']}: {e}")

    print(f"  Done. {len(downloaded_paths)} files saved to data/raw/{ticker.upper()}/")
    return downloaded_paths


# for testing
if __name__ == "__main__":
    # Run this file directly to test: python -m ingestion.sec_fetcher
    paths = fetch_filings_for_ticker("AAPL", filing_types=["10-K"])
    print("\nFiles ready for ingestion:")
    for p in paths:
        print(f"  {p}")