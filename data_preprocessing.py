import yfinance as yf
import pandas as pd

def load_stock_data(ticker='AAPL', start='2015-01-01', end='2024-12-31'):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df[['Close']]

if __name__ == "__main__":
    df = load_stock_data()
    print(df.head())
