import streamlit as st
import plotly.express as px
from data_fetcher import fetch_stock_data, fetch_news_data
from sentiment_model import analyze_sentiment
import os

st.set_page_config(page_title="StockSentinel", layout="wide")
st.title("📈 StockSentinel: AI-Powered Market Sentiment Analyzer")

user_api_key = st.text_input("151eb78228084f2fb633e9aacb91ba96")
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA, AAPL):")

if ticker and user_api_key:
    st.write("Fetching latest data...")
    stock_data = fetch_stock_data(ticker)
    news_data = fetch_news_data(ticker, user_api_key)
    analyzed_data, sentiment_summary = analyze_sentiment(news_data)
    
    st.subheader("Sentiment Distribution")
    fig = px.pie(names=list(sentiment_summary.keys()), values=list(sentiment_summary.values()), title="News Sentiment Share")
    st.plotly_chart(fig)
    
    st.subheader("Recent Market News")
    st.dataframe(analyzed_data[['title', 'source', 'publishedAt', 'sentiment']])
    
    st.subheader("Stock Price Over Time")
    price_chart = px.line(stock_data, x='Date', y='Close', title=f'{ticker} Closing Prices (30 Days)')
    st.plotly_chart(price_chart)
st.subheader("Stock Price Over Time")
if 'Date' in stock_data.columns and 'Close' in stock_data.columns:
    price_chart = px.line(stock_data, x='Date', y='Close',
                          title=f'{ticker} Closing Prices (30 Days)')
    st.plotly_chart(price_chart, use_container_width=True)
else:
    st.warning("Stock data not available or incorrectly formatted.")
