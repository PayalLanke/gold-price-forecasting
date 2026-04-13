import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Gold Price Engine", page_icon="🪙", layout="wide")

def get_theme_css(theme):
    if theme == "Dark Mode":
        return """
        <style>
            .stApp { background-color: #0f172a; color: #f8fafc; }
            h1, h2, h3, h4 { color: #facc15 !important; font-family: 'Inter', sans-serif; }
            .metric-container { background: #1e293b; padding: 20px; border-radius: 12px; border-left: 4px solid #facc15; box-shadow: 0 4px 6px rgba(0,0,0,0.3); text-align: left; margin-bottom: 20px;}
            .val { font-size: 2rem; font-weight: 700; color: #ffffff; margin-top: 5px;}
            .lbl { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
            .insight-box { background: #1e293b; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 8px; margin-top: 20px;}
            .info-box { background: #1e293b; border-left: 4px solid #facc15; padding: 15px; border-radius: 8px; margin-top: 20px;}
            .disclaimer { font-size: 0.8rem; color: #64748b; text-align: center; margin-top: 50px; }
            [data-testid="stMetricValue"] { font-size: 2.8rem !important; font-weight: 900 !important; color: #ffffff !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
            [data-testid="stMetricDelta"] { font-size: 1.3rem !important; font-weight: 800 !important; }
            [data-testid="stMetricLabel"] p { font-size: 1.1rem !important; font-weight: 800 !important; color: #facc15 !important; }
        </style>
        """
    else:
        return """
        <style>
            .stApp { background-color: #f8fafc; color: #0f172a; }
            h1, h2, h3, h4 { color: #b45309 !important; font-family: 'Inter', sans-serif; }
            .metric-container { background: #ffffff; padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: left; margin-bottom: 20px;}
            .val { font-size: 2rem; font-weight: 700; color: #0f172a; margin-top: 5px;}
            .lbl { font-size: 0.9rem; color: #475569; text-transform: uppercase; letter-spacing: 1px; }
            .insight-box { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 8px; margin-top: 20px; color: #0f172a;}
            .info-box { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 15px; border-radius: 8px; margin-top: 20px; color: #0f172a;}
            .disclaimer { font-size: 0.8rem; color: #64748b; text-align: center; margin-top: 50px; }
            [data-testid="stMetricValue"] { font-size: 2.8rem !important; font-weight: 900 !important; color: #0f172a !important; text-shadow: none; }
            [data-testid="stMetricDelta"] { font-size: 1.3rem !important; font-weight: 800 !important; }
            [data-testid="stMetricLabel"] p { font-size: 1.1rem !important; font-weight: 800 !important; color: #3b82f6 !important; }
        </style>
        """

@st.cache_data
def load_data():
    if os.path.exists("data/processed_gold_data.csv"):
        df = pd.read_csv("data/processed_gold_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

@st.cache_data
def load_future():
    if os.path.exists("outputs/ui_future_forecasts.csv"):
        df = pd.read_csv("outputs/ui_future_forecasts.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

@st.cache_data
def load_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def dashboard():
    st.title("🪙 Premium Gold Market Forecaster")
    
    df = load_data()
    future_df = load_future()
    
    if df is None or future_df is None:
        st.error("Processed or forecast data missing. Please ensure the backend scripts are run.")
        return
        
    if 'forecast_clicked' not in st.session_state:
        st.session_state.forecast_clicked = False
        
    MODEL_MAP = {
        "📈 Trend-Based Forecast": "prophet",
        "📊 Statistical Forecast": "arima",
        "🤖 AI Deep Learning Forecast": "lstm"
    }

    st.sidebar.title("🎛️ Configuration")
    
    ui_theme = st.sidebar.radio("🌗 Select Theme", ["Dark Mode", "Light Mode"])
    st.markdown(get_theme_css(ui_theme), unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    chosen_model_label = st.sidebar.selectbox("Forecast Type", list(MODEL_MAP.keys()))
    horizon = st.sidebar.slider("🔮 Select Forecast Days (7–90)", 7, 90, 30, step=1)
    time_range = st.sidebar.radio("📅 Time Range Selector", ["Last 1 Year", "Last 5 Years", "Full Data"])
    show_ma = st.sidebar.toggle("Show 30-Day Moving Average", value=True)
    
    if st.sidebar.button("🔍 Generate Forecast", use_container_width=True, type="primary"):
        st.session_state.forecast_clicked = True
    
    model_key = MODEL_MAP[chosen_model_label]

    last_date = df['Date'].max()
    if time_range == "Last 1 Year":
        df_filtered = df[df['Date'] >= (last_date - pd.DateOffset(years=1))]
    elif time_range == "Last 5 Years":
        df_filtered = df[df['Date'] >= (last_date - pd.DateOffset(years=5))]
    else:
        df_filtered = df

    future_plot = future_df.head(horizon)
    
    if not st.session_state.forecast_clicked:
        st.info("👈 Please select your parameters from the sidebar and click **Generate Forecast** to view the analytical projections.")
        return

    # Introduce Multi-Tab Architecture
    tab1, tab2, tab3 = st.tabs(["📊 Live Forecasting Engine", "🗃️ Historical Data Browser", "⚙️ AI Explanatory Engine"])

    with tab1:
        # ---------------------------------------------------------
        # 1. PREDICTION SUMMARY USING st.metric and st.success
        # ---------------------------------------------------------
        st.markdown("### 📦 Prediction Summary")
        
        last_actual_price = df['Close'].iloc[-1]
        final_predicted_price_usd = future_plot[model_key].iloc[-1]
        price_diff_usd = final_predicted_price_usd - last_actual_price
        trend_up = price_diff_usd > 0
        pct_change = (price_diff_usd / last_actual_price) * 100
        
        INR_CONVERSION_RATE = 83.50 
        final_predicted_price_inr = final_predicted_price_usd * INR_CONVERSION_RATE
        price_diff_inr = price_diff_usd * INR_CONVERSION_RATE
        
        trend_arrow = "⬆️" if trend_up else "⬇️"
        trend_word = "Increasing" if trend_up else "Decreasing"
        
        st.success(f"{trend_arrow} **Forecast generated successfully! The market trend points towards {trend_word} prices within the next {horizon} days.**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"Predicted Value ({horizon} Days) [USD]", 
                value=f"${final_predicted_price_usd:,.2f}", 
                delta=f"${price_diff_usd:,.2f} ({pct_change:+.2f}%)",
                delta_color="normal"
            )
        with col2:
            st.metric(
                label=f"Predicted Value ({horizon} Days) [INR Approx]", 
                value=f"₹{final_predicted_price_inr:,.2f}", 
                delta=f"₹{price_diff_inr:,.2f}",
                delta_color="normal"
            )
        
        # ---------------------------------------------------------
        # 2. INTERACTIVE PLOTLY GRAPH
        # ---------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Interactive Forecast Graph")
        
        fig = go.Figure()
        
        # Historical Curve
        fig.add_trace(go.Scatter(
            x=df_filtered['Date'], y=df_filtered['Close'], 
            name='Actual Price', 
            line=dict(color='#fbd38d', width=2),
            hovertemplate='<b>Date</b>: %{x}<br><b>Actual Price</b>: $%{y:.2f}'
        ))
        
        if show_ma:
            fig.add_trace(go.Scatter(
                x=df_filtered['Date'], y=df_filtered['MA_30'], 
                name='30-Day MA', 
                line=dict(color='#60a5fa', dash='dash', width=1.5),
                hovertemplate='<b>Date</b>: %{x}<br><b>30-Day MA</b>: $%{y:.2f}'
            ))
            
        fig.add_trace(go.Scatter(
            x=future_plot['Date'], y=future_plot[model_key], 
            name=f"Predicted ({chosen_model_label})", 
            line=dict(color='#34d399' if trend_up else '#f87171', width=3),
            hovertemplate='<b>Date</b>: %{x}<br><b>Predicted Price</b>: $%{y:.2f}'
        ))
        
        if ui_theme == "Dark Mode":
            plot_bg, plot_font = '#0f172a', '#ffffff'
        else:
            plot_bg, plot_font = '#f8fafc', '#000000'
            
        fig.update_layout(
            plot_bgcolor=plot_bg, paper_bgcolor=plot_bg, font=dict(color=plot_font, family="Inter", size=14), 
            hovermode="x unified",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color=plot_font, size=15)
            )
        )
        
        # Activate Native Range Slider
        fig.update_xaxes(rangeslider_visible=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download Forecast Button
        download_df = future_plot[['Date', model_key]].rename(columns={model_key: "Predicted_Price_USD"})
        download_df['Predicted_Price_INR'] = download_df['Predicted_Price_USD'] * INR_CONVERSION_RATE
        csv_data = load_csv(download_df)
        
        st.download_button(
            label="📥 Download Forecasting Dataset (CSV)",
            data=csv_data,
            file_name=f"gold_forecast_{horizon}days.csv",
            mime='text/csv',
            help="Export these exact AI predictions to a CSV file for financial modelling.",
            type="primary"
        )

        # ---------------------------------------------------------
        # 3. AUTO-GENERATED INSIGHTS
        # ---------------------------------------------------------
        st.markdown("### 🧠 Market Insights")
        if trend_up and pct_change > 2:
            insight_text = f"Gold prices show a **strong upward trend of {pct_change:.2f}%**, indicating **bullish market behavior**. This is commonly driven by macroeconomic hedging strategies."
        elif trend_up:
            insight_text = "Gold prices are projecting a **steady, subtle upward movement**. Safe-haven accumulation appears stable with minimized short-term volatility."
        elif not trend_up and pct_change < -2:
            insight_text = f"Analysis projects a **downward trajectory of {pct_change:.2f}%**, indicating **bearish behavior** in the short-to-medium term. Recent fluctuations suggest a correction in premium pricing."
        else:
            insight_text = "Prices are expected to experience a **minor correction or stabilization**. Structural resistance thresholds maintain market balance."

        st.markdown(f"<div class='insight-box'>💡 {insight_text}</div>", unsafe_allow_html=True)

    with tab2:
        # ---------------------------------------------------------
        # 4. HISTORICAL DATA BROWSER
        # ---------------------------------------------------------
        st.markdown("### 🗃️ Historical Data Browser")
        st.markdown("Explore the raw, chronological market data fetched natively from Yahoo Finance.")
        st.dataframe(df_filtered[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values(by="Date", ascending=False), use_container_width=True, height=500)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📌 Factors Affecting Gold Prices")
        st.markdown("""
        <div class='info-box'>
            <ul style="margin-bottom: 0;">
                <li><b>Inflation Rates:</b> Gold is traditionally viewed as a reliable asset preservation hedge against rapid currency inflation.</li>
                <li><b>Global Currency Value:</b> Because gold is dollar-denominated, a weaker global U.S. dollar structure naturally pushes gold prices up.</li>
                <li><b>Central Bank Reserves:</b> Large-scale purchasing or liquidations by governmental central banks affect mass supply-demand equilibriums.</li>
                <li><b>Geopolitical Tensions:</b> World-stage uncertainty natively drives safe-haven allocations into long-term bullion storage.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with tab3:
        # ---------------------------------------------------------
        # 5. MODEL PERFORMANCE METRICS
        # ---------------------------------------------------------
        st.markdown("### 📊 AI Model Architecture & Performance")
        st.markdown("Evaluating the mathematical accuracy of the selected algorithm using historical hold-out data.")
        
        metric_file = f"outputs/{model_key}_metrics.csv"
        if os.path.exists(metric_file):
            m_df = pd.read_csv(metric_file)
            mae, rmse, mape = m_df['MAE'].values[0], m_df['RMSE'].values[0], m_df['MAPE'].values[0]
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-container'><div class='lbl'>Mean Abs Error (MAE)</div><div class='val'>{mae:.2f}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-container'><div class='lbl'>Root Mean Sq Error</div><div class='val'>{rmse:.2f}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-container'><div class='lbl'>Percentage Error (MAPE)</div><div class='val'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
            
            st.caption(f"ℹ️ **Interpretation**: A MAPE of {mape:.2f}% {'indicates excellent prediction mathematically, maintaining less than a 10% error margin.' if mape < 10 else 'indicates a decent macro directional guideline but potentially contains localized fluctuation variance.'}")
        else:
            st.warning("Performance metrics not located for the chosen model.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='disclaimer'>⚠️ Predictions are computed autonomously based on mathematical historical structures and may not actively account for real-time unpredicted economic shifts.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    dashboard()
