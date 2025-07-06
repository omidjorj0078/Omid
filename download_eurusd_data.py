import yfinance as yf
import pandas as pd

eurusd = yf.Ticker("EURUSD=X")
data = eurusd.history(period="max", interval="1m")

data.to_csv("eurusd_1m_historical_data.csv")
print("EURUSD 1-minute historical data downloaded and saved to eurusd_1m_historical_data.csv")

