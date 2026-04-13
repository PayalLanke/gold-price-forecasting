import yfinance as yf
import pandas as pd
import os

def collect_data():
    os.makedirs("data", exist_ok=True)
    print("Downloading gold data...")
    gold_data = yf.download("GC=F", start="2015-01-01", end="2026-01-01")
    
    # Save the raw data
    output_path = "data/gold_price_data.csv"
    gold_data.to_csv(output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    collect_data()
