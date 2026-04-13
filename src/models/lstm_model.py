import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set memory growth for tensorflow to avoid aggressive memory allocation crashes
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, rmse, mape

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def train_lstm():
    filepath = "data/processed_gold_data.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    close_prices = df[['Close']]
    
    # Chronological Train-test split
    train_df = close_prices[:'2023']
    test_df = close_prices['2024':]
    
    print(f"Training LSTM on {len(train_df)} samples. Testing on {len(test_df)} samples.")
    
    # Scale Data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train = scaler.fit_transform(train_df.values)
    
    os.makedirs("models", exist_ok=True)
    with open("models/lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    seq_length = 60 # Using 60 days to predict the next
    X_train, y_train = create_sequences(scaled_train, seq_length)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Deep Learning Architecture
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=32, epochs=30, callbacks=[early_stop], verbose=1)
    
    model.save("models/lstm_model.h5")
    print("LSTM model trained and saved successfully.")
    
    # Evaluation
    if len(test_df) > 0:
        total_data = pd.concat((train_df, test_df), axis=0)
        # We grab the sequences ending right where test data starts 
        inputs = total_data[len(total_data) - len(test_df) - seq_length:].values
        inputs = scaler.transform(inputs)
        
        X_test, y_test_scaled = create_sequences(inputs, seq_length)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predicted_prices = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        
        y_test_actual = test_df.values
        min_len = min(len(predicted_prices), len(y_test_actual))
        y_test_actual = y_test_actual[:min_len]
        predicted_prices = predicted_prices[:min_len]
        
        mae, rmse, mape = evaluate_metrics(y_test_actual, predicted_prices)
        print(f"\n[LSTM Validation Metrics]")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        pd.DataFrame([{'Model': 'LSTM', 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}]).to_csv("outputs/lstm_metrics.csv", index=False)
        
        os.makedirs("outputs", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(train_df.index, train_df.values, label='Train')
        plt.plot(test_df.index[:min_len], y_test_actual, label='Validation Actual')
        plt.plot(test_df.index[:min_len], predicted_prices, label='Validation Predict', color='red', linestyle='--')
        plt.title('LSTM Forecasting (Train-Test Split)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("outputs/lstm_validation.png", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    train_lstm()
