import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers):
    # Fetch multiple tickers
    data = yf.download(tickers, period="30d", interval="1d", group_by='ticker')

    # Reset index to make Date a proper column
    data.reset_index(inplace=True)

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in data.columns
        ]

    return data
