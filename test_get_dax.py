import yfinance as yf
import pandas as pd
from datetime import datetime

# Daten abrufen (DAX Ticker: ^GDAXI)
ticker = "^GDAXI"
# Hole Daten für ca. 10 Jahre (Puffer eingebaut)
df = yf.download(ticker, start="2015-01-01", interval="1mo", progress=False)

# MultiIndex bereinigen (yfinance gibt oft (Price, Ticker) zurück)
if isinstance(df.columns, pd.MultiIndex):
    try:
        df = df.xs('Close', axis=1, level=0, drop_level=True)
    except:
        df = df.iloc[:, 0] # Fallback
elif 'Close' in df.columns:
    df = df['Close']

# Store as CSV (use ";" as separator for compatibility)
df = df.reset_index()
df.to_csv("dax_monthly.csv", sep=";", index=False)