import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

def generate_future():
    print("Generating comprehensive future forecasts...")
    filepath = "data/processed_gold_data.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Train Prophet on FULL historical data to allow valid extrapolation
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=90, freq='B') # Business days
    forecast = model.predict(future)
    
    os.makedirs("outputs", exist_ok=True)
    
    for days in [30, 60, 90]:
        plt.figure(figsize=(12, 6))
        plt.plot(prophet_df['ds'].tail(365), prophet_df['y'].tail(365), label='Last 1 Year Actuals', color='#facc15')
        
        future_mask = forecast['ds'] > prophet_df['ds'].max()
        extrapolation = forecast[future_mask].head(days)
        
        plt.plot(extrapolation['ds'], extrapolation['yhat'], label=f'{days}-Day Projection', color='#ef4444', linestyle='--')
        plt.fill_between(extrapolation['ds'], extrapolation['yhat_lower'], extrapolation['yhat_upper'], color='#ef4444', alpha=0.1)
        
        plt.title(f'Gold Price {days}-Day Future Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"outputs/future_forecast_{days}.png", bbox_inches='tight')
        plt.close()
        
    print("Saved comprehensive future forecasting visualizations.")

if __name__ == "__main__":
    generate_future()
