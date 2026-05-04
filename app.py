import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from model_logic import prepare_data, build_lstm_model, predict_future

# 1. Page Config
st.set_page_config(page_title="AI Stock Predictor", layout="wide", page_icon="📈")

# 2. Earthy Professional Styling
st.markdown("""
    <style>
    .stApp { background-color: #FAF9F6 !important; }
    section[data-testid="stSidebar"] { background-color: #EFEBE3 !important; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 2px solid #A0522D !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #A0522D !important; }
    div, p, label, .stMetricLabel, .stMetricValue { color: #2F4F4F !important; }
    div.stButton > button {
        background-color: #A0522D !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar
st.sidebar.title("🛠️ Control Panel")
ticker_input = st.sidebar.text_input("Stock Symbol", "AAPL")
seq_len = st.sidebar.slider("Lookback (Days)", 30, 90, 60)
pred_len = st.sidebar.slider("Prediction Horizon", 1, 14, 7)

# 4. Data Logic - Optimized for Cloud Deployment
@st.cache_data(ttl=3600)
def get_data(symbol):
    try:
        clean_symbol = symbol.strip().upper()
        ticker_obj = yf.Ticker(clean_symbol)
        
        # Using history() is more stable than download() on cloud servers
        df = ticker_obj.history(period="2y", interval="1d", auto_adjust=True)
        
        if df.empty:
            return pd.DataFrame()
            
        # Select required columns and remove timezone for compatibility
        df = df[['Open', 'High', 'Low', 'Close']]
        df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# 5. UI Layout
st.title("🚀 AI-Powered Stock Forecasting")
st.markdown("### 🧠 Multistep-Multivariate LSTM Neural Network")

try:
    with st.spinner(f"Fetching data for {ticker_input}..."):
        data = get_data(ticker_input)
    
    if data.empty:
        st.error(f"❌ No data found for '{ticker_input}'. Please check the symbol or try again later.")
        st.info("💡 Hint: Try 'AAPL', 'TSLA', or 'BTC-USD'.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Market Overview")
            last_price = data['Close'].iloc[-1].item()
            st.metric(f"Latest {ticker_input} Close", f"${last_price:.2f}")
            st.write(data.tail(5))
            
        with col2:
            st.subheader("📈 Historical Trend (Last 100 Days)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index[-100:], 
                y=data['Close'][-100:], 
                name="Close Price", 
                line=dict(color='#A0522D')
            ))
            fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🚀 Run AI Prediction", type="primary"):
            # Ensure we have enough data for the chosen lookback
            if len(data) < seq_len:
                st.error(f"❌ Not enough data! Need {seq_len} days, but only found {len(data)}.")
            else:
                with st.spinner("Training LSTM (Optimized for Free Tier)..."):
                    # Prepare data
                    X, y, scaler = prepare_data(data, seq_len, pred_len)
                    
                    # Build and train model
                    # Note: We use 2 epochs and a small subset to prevent Memory Errors on Render
                    model = build_lstm_model(seq_len, pred_len)
                    model.fit(X[-50:], y[-50:], epochs=2, batch_size=16, verbose=0)
                    
                    # Run inference
                    last_seq = data.values[-seq_len:]
                    prediction = predict_future(model, last_seq, scaler)
                    
                    st.success("✅ Prediction generated!")
                    st.subheader(f"🔮 {pred_len}-Day Forecast")
                    
                    # Generate forecast dates
                    pred_dates = pd.date_range(start=data.index[-1], periods=pred_len+1)[1:]
                    
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=pred_dates, 
                        y=prediction, 
                        mode='lines+markers', 
                        name="Forecast",
                        line=dict(color='#6B8E23', width=3)
                    ))
                    fig_pred.update_layout(template="plotly_white", xaxis_title="Future Dates", yaxis_title="Predicted Price")
                    st.plotly_chart(fig_pred, use_container_width=True)

except Exception as e:
    st.error(f"❌ An unexpected error occurred: {e}")