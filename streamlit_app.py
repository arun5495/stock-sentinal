import streamlit as st
import plotly.express as px
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="StockSentinel", layout="wide")
st.title("📈 StockSentinel: AI-Powered Market Sentiment Analyzer")

# ------------------------------------------------------------
# YOUR NEWSAPI KEY
# ------------------------------------------------------------
API_KEY = "151eb78228084f2fb633e9aacb91ba96"  # Replace with your real key

# ------------------------------------------------------------
# CACHED FUNCTIONS
# ------------------------------------------------------------

@st.cache_resource
def load_finbert():
    """Load FinBERT safely on CPU (avoid meta tensor issues)."""
    torch.set_default_device("cpu")
    model_name = "ProsusAI/finbert"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
    return pipe


@st.cache_data(ttl=3600)
def fetch_stock_data(tickers):
    """Fetch market data using yfinance, flatten columns to single level."""
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers,
        period="30d",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        multi_level_index=False,  # ✅ Works for yfinance ≥ 0.2.51
    )

    data.reset_index(inplace=True)

    # Fallback flattening (for older yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        if len(tickers) == 1:
            data.columns = data.columns.get_level_values(0)
        else:
            data.columns = [f"{a}_{b}" if b else a for a, b in data.columns]

    data = data.loc[:, ~data.columns.duplicated()]
    data = data.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)
    return data


@st.cache_data(ttl=3600)
def fetch_news(query):
    """Collect latest news about selected company via NewsAPI."""
    try:
        newsapi = NewsApiClient(api_key=API_KEY)
        articles = newsapi.get_everything(
            q=query, language="en", sort_by="publishedAt", page_size=20
        )
        df = pd.DataFrame(
            [
                {
                    "title": i["title"],
                    "description": i["description"],
                    "publishedAt": i["publishedAt"],
                    "source": i["source"]["name"],
                }
                for i in articles["articles"]
            ]
        )
        return df
    except Exception as e:
        st.warning(f"News fetch failed for {query}: {e}")
        return pd.DataFrame()


def analyze_sentiment(df):
    """Run FinBERT sentiment analysis on news titles."""
    finbert = load_finbert()
    sentiments = []
    for text in df["title"]:
        if isinstance(text, str):
            try:
                output = finbert(text[:512])[0]
                sentiments.append(output["label"])
            except Exception:
                sentiments.append("NEUTRAL")
        else:
            sentiments.append("NEUTRAL")
    df["sentiment"] = sentiments
    summary = df["sentiment"].value_counts().to_dict()
    return df, summary


# ------------------------------------------------------------
# MAIN INPUT AND UI
# ------------------------------------------------------------
tickers = st.multiselect(
    "Select Companies to Analyze:",
    ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "NFLX"],
)

# ------------------------------------------------------------
# PIPELINE EXECUTION
# ------------------------------------------------------------
if tickers:
    for ticker in tickers:
        st.header(f"Results for {ticker}")

        try:
            stock_data = fetch_stock_data(ticker)
            news_data = fetch_news(ticker)

            # --- Validate Data ---
            if stock_data.empty:
                st.warning(f"No stock data found for {ticker}.")
                continue
            if news_data.empty:
                st.warning(f"No news found for {ticker}.")
                continue

            analyzed_df, sentiment_summary = analyze_sentiment(news_data)

            # --- STOCK CHART ---
            close_col = f"Close_{ticker}" if f"Close_{ticker}" in stock_data.columns else "Close"
            if "Date" in stock_data.columns and close_col in stock_data.columns:
                fig = px.line(
                    stock_data,
                    x="Date",
                    y=close_col,
                    title=f"{ticker} – Closing Prices (Last 30 Days)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"‘Close’ column missing for {ticker}.")

            # --- SENTIMENT PIE CHART ---
            if sentiment_summary:
                fig_pie = px.pie(
                    names=list(sentiment_summary.keys()),
                    values=list(sentiment_summary.values()),
                    title=f"{ticker} – News Sentiment",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- NEWS TABLE ---
            st.subheader(f"Recent News Articles for {ticker}")
            st.dataframe(
                analyzed_df[["title", "source", "publishedAt", "sentiment"]],
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Error while processing {ticker}: {e}")

else:
    st.info("Please select at least one company to start analysis.")
