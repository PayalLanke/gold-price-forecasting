import pandas as pd
import os

def run_preprocessing():
    filepath = "data/gold_price_data.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    print(f"Loading data from {filepath}...")
    
    try:
        # Load raw data. yfinance sometimes exports with 2-level headers if saved naively
        df = pd.read_csv(filepath)
        if len(df) > 0 and ("Ticker" in str(df.iloc[0].values) or "GC=F" in str(df.iloc[0].values)):
            df = pd.read_csv(filepath, header=[0, 1], index_col=0)
            df.columns = df.columns.droplevel(1) # Drop the ticker name
            df.index.name = 'Date'
            df.reset_index(inplace=True)
    except Exception as e:
        print("Warning during loading:", e)
        df = pd.read_csv(filepath)

    # Standardize Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
    elif df.index.name in ['Date', 'Price']:
        df.index = pd.to_datetime(df.index, errors='coerce')
        df.index.name = 'Date'
        
    df.sort_index(inplace=True)
    
    # Ensure numeric types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Forward and Backward Fill missing data
    df = df.ffill().bfill()

    if 'Close' not in df.columns:
        print("Error: 'Close' column missing. Columns:", list(df.columns))
        return
        
    # Feature Engineering
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_3'] = df['Close'].shift(3)
    
    # Drop rows with NaN due to rolling/lag operations
    df.dropna(inplace=True)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_gold_data.csv")
    print(f"Preprocessed data saved to data/processed_gold_data.csv. Total ready rows: {len(df)}")
    
if __name__ == "__main__":
    run_preprocessing()
