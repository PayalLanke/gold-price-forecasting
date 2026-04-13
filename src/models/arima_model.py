import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pickle
import warnings
warnings.filterwarnings("ignore")

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, rmse, mape

def train_arima():
    filepath = "data/processed_gold_data.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('B') # Filter to business days since markets are typically closed on weekends
    df['Close'] = df['Close'].ffill().bfill()
    
    # Train-test split
    train = df[:'2023']['Close']
    test = df['2024':]['Close']
    
    print(f"Training ARIMA on {len(train)} samples. Testing on {len(test)} samples.")
    
    # Baseline SARIMA model
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    os.makedirs("models", exist_ok=True)
    with open("models/arima_model.pkl", "wb") as f:
        pickle.dump(results, f)
        
    print("ARIMA model trained and saved successfully.")
    
    if len(test) > 0:
        predictions = results.get_forecast(steps=len(test))
        y_pred = predictions.predicted_mean
        
        mae, rmse, mape = evaluate_metrics(test.values, y_pred.values)
        print(f"\n[ARIMA Validation Metrics]")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # Save metrics to a file to be shown in UI later
        pd.DataFrame([{'Model': 'ARIMA', 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}]).to_csv("outputs/arima_metrics.csv", index=False)
        
        os.makedirs("outputs", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test, label='Validation Actual')
        plt.plot(test.index, y_pred, label='Validation Predict', color='red', linestyle='--')
        plt.title('ARIMA Forecasting (Train-Test Split)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("outputs/arima_validation.png", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    train_arima()
