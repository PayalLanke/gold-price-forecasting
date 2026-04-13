import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pickle

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, rmse, mape

def train_prophet():
    filepath = "data/processed_gold_data.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Prophet explicitly requires columns to be named 'ds' and 'y'
    prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Train-test split
    train = prophet_df[prophet_df['ds'] <= '2023-12-31']
    test = prophet_df[prophet_df['ds'] >= '2024-01-01']
    
    print(f"Training Prophet on {len(train)} samples. Testing on {len(test)} samples.")
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train)
    
    os.makedirs("models", exist_ok=True)
    with open("models/prophet_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("Prophet model trained and saved successfully.")
        
    if len(test) > 0:
        # Create future dataframe matching the test index size
        future = model.make_future_dataframe(periods=len(test), freq='B')
        forecast = model.predict(future)
        
        # Merge forecast with test to align precise matching dates 
        test_eval = test.merge(forecast[['ds', 'yhat']], on='ds', how='inner')
        
        if len(test_eval) > 0:
            mae, rmse, mape = evaluate_metrics(test_eval['y'], test_eval['yhat'])
            print(f"\n[Prophet Validation Metrics]")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAPE: {mape:.2f}%")
            
            pd.DataFrame([{'Model': 'Prophet', 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}]).to_csv("outputs/prophet_metrics.csv", index=False)
            
            plt.figure(figsize=(10, 5))
            plt.plot(train['ds'], train['y'], label='Train')
            plt.plot(test_eval['ds'], test_eval['y'], label='Validation Actual')
            plt.plot(test_eval['ds'], test_eval['yhat'], label='Validation Predict', color='red', linestyle='--')
            plt.title('Facebook Prophet Forecasting (Train-Test Split)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            os.makedirs("outputs", exist_ok=True)
            plt.savefig("outputs/prophet_validation.png", bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    train_prophet()
