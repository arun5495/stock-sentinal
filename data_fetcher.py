import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=1800)
def fetch_stock_data(tickers):
    data = yf.download(tickers, period="30d", interval="0d", group_by="ticker")

    data.reset_index(inplace=True)

    # Flatten MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in data.columns
        ]

    return data
