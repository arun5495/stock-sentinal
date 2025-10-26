import streamlit as st
import plotly.express as px
from data_fetcher import fetch_stock_data, fetch_news_data
from sentiment_model import analyze_sentiment
import os

st.set_page_config(page_title="StockSentinel", layout="wide")
st.title("📈 StockSentinel: AI-Powered Market Sentiment Analyzer")


api_key = "151eb78228084f2fb633e9aacb91ba96" 


stock_data = None
news_data = None
sentiment_summary = {}
analyzed_data = None

# --- Input field for stock ticker ---
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA, AAPL):")

# --- Main Logic ---
if ticker:
    with st.spinner("Fetching latest data..."):
        try:
            # Fetch stock and news data
            stock_data = fetch_stock_data(ticker)
            news_data = fetch_news_data(ticker, api_key)

            if stock_data is not None and not stock_data.empty and news_data is not None and not news_data.empty:
                analyzed_data, sentiment_summary = analyze_sentiment(news_data)
            else:
                st.warning("No data found for this ticker. Try a different one.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Stock Chart Visualization ---
if stock_data is not None and not stock_data.empty:
    st.subheader("Stock Price Over Time")
    if "Date" in stock_data.columns and "Close" in stock_data.columns:
        try:
            price_chart = px.line(
                stock_data, x="Date", y="Close",
                title=f"{ticker.upper()} Closing Prices (Last 30 Days)"
            )
            st.plotly_chart(price_chart, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render stock chart: {e}")
    else:
        st.warning("Stock data missing 'Date' or 'Close' columns.")

# --- Sentiment Distribution Visualization ---
if sentiment_summary:
    st.subheader("Sentiment Distribution")
    try:
        fig = px.pie(
            names=list(sentiment_summary.keys()),
            values=list(sentiment_summary.values()),
            title="Market News Sentiment Share"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Error rendering sentiment chart: {e}")

# --- News Data Table ---
if analyzed_data is not None and not analyzed_data.empty:
    st.subheader("Latest Analyzed News")
    st.dataframe(
        analyzed_data[["title", "source", "publishedAt", "sentiment"]],
        use_container_width=True
    )
