import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

def run_eda():
    filepath = "data/processed_gold_data.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
    os.makedirs("outputs", exist_ok=True)
    print("Generating EDA visualizations...")
    
    # 1. Trend Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='gold', alpha=0.8)
    plt.plot(df.index, df['MA_30'], label='30-Day MA', color='red', linestyle='--')
    plt.title('Gold Price Trend with 30-Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/trend_visualization.png", bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_7', 'MA_30', 'EMA_14', 'Lag_1', 'Lag_3']
    existing_cols = [c for c in cols if c in df.columns]
    corr = df[existing_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig("outputs/correlation_heatmap.png", bbox_inches='tight')
    plt.close()
    
    # 3. Time Series Decomposition (Seasonality Analysis)
    try:
        # Resample to monthly means for clearer seasonality
        df_monthly = df['Close'].resample('ME').mean().dropna()
        decomposition = seasonal_decompose(df_monthly, model='additive', period=12) 
        
        fig = decomposition.plot()
        fig.set_size_inches(14, 10)
        fig.suptitle('Time Series Decomposition (Monthly)', fontsize=16)
        plt.tight_layout()
        plt.savefig("outputs/time_series_decomposition.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not perform decomposition: {e}")
        
    print("EDA Visualizations successfully saved to outputs/")

if __name__ == "__main__":
    run_eda()
