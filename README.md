# Gold Price Forecasting using Time Series Analysis

## Problem Statement
Develop a forecasting model that predicts future gold prices using historical gold market data. The system analyzes trends, seasonality, and patterns to generate accurate and justifiable predictions of future gold prices.

## Dataset Explanation
Historical gold price data is fetched dynamically using the Yahoo Finance API for the ticker `GC=F` (Gold Futures). The dataset covers trading data from 2015 to 2026, capturing critical attributes like Open, High, Low, Close, and Volume.

## Data Preprocessing Steps
- Interpolation and forward-filling of missing values.
- Conversion of `Date` columns to explicit chronological datetime indices.
- Feature derivation including Moving Averages (MA, EMA), rolling statistics, and specific historical event lags.
- Normalization mapping optimal for sequential neural network input structure.

## Models Used
- **ARIMA / SARIMA:** Autoregressive Integrated Moving Average, a powerful baseline for extracting structured short-term linear dependencies.
- **Facebook Prophet:** An additive framework exceptionally tuned to capture nonlinear trends along with weekly/yearly seasonality effects.
- **LSTM (Long Short-Term Memory):** A deep learning recurrent neural network trained on sequence batches to extrapolate deeper, temporally linked insights.

## Evaluation Metrics and Results
All predictive structures undergo rigorous testing via chronological Train-Test Splits (*train on historical pre-2023 data, test on recent post-2024 records*). Model efficacy is evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

*Metrics breakdown will be detailed post-evaluation sweep.*

## Forecasting Explanation
Outputs forecast trajectories for **30**, **60**, and **90-day** horizons.
> **Note:** Predictions formulated herein rely heavily on historically observed trajectories and internal statistical consistency, however financial instruments (like Gold) are highly susceptible to real-world exogenous factors (economic policy, geopolitical tensions).

## Steps to Run the Project
1. **Set up Virtual Environment & Install Dependencies:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
2. **Collect Data & Validate Models:**
```bash
python src/data_collection.py
# Execute additional scripts as per the upcoming code
```
3. **Launch the Real-time Dashboard:**
```bash
streamlit run app/main.py
```
